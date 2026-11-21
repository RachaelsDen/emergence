"""One filesystem destination and one retrying consumer of the SQLite outbox."""

import errno
import os
import re
import stat
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Thread
from time import monotonic
from typing import Final, Literal

from anyio import Event, Lock, RunFinishedError, fail_after, from_thread, sleep, to_thread
from anyio.lowlevel import current_token
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from .archive import Spool, SpoolError
from .database import Database
from .gateway_state import request_capacity
from .models import archive_outbox, archive_references, request_events
from .repositories import ArchiveReference

_DIRECTORY: Final = os.O_DIRECTORY | os.O_NOFOLLOW | os.O_RDONLY
_IO_TIMEOUT: Final = 5
_POLL: Final = 1


def _mount_id(fd: int) -> int:
    """Linux mount identity of the *open object*, including bind mounts."""
    for line in Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines():
        if line.startswith("mnt_id:"):
            return int(line.split()[1])
    raise SpoolError("archive_mount_unknown")


class MountedArchive:
    def __init__(self, path: Path, mount: Path, local: Path) -> None:
        self.path, self.mount = path.absolute(), mount.absolute()
        self._local = local
        if (
            ".." in self.path.parts
            or ".." in self.mount.parts
            or self.mount == Path("/")
            or not self.path.is_relative_to(self.mount)
            or self.path.is_relative_to(local)
            or local.is_relative_to(self.mount)
        ):
            raise SpoolError("unsafe_archive_path")
        self._slot = Lock()
        self._running: Event | None = None
        self._started = 0.0

    @property
    def overdue(self) -> bool:
        return (
            self._running is not None
            and not self._running.is_set()
            and monotonic() - self._started >= _IO_TIMEOUT
        )

    async def io[T, **P](self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """One daemon syscall lane. Cancellation never releases an occupied lane."""
        async with self._slot:
            if self._running is not None:
                await self._running.wait()
            token = current_token()
            finished = Event()
            result: Future[T] = Future()

            def work() -> None:
                try:
                    result.set_result(fn(*args, **kwargs))
                except BaseException as error:
                    # Thread entry point: transport exceptions intact to result.result().
                    # Only the delivery loop classifies expected I/O and database failures.
                    result.set_exception(error)
                finally:
                    with suppress(RunFinishedError):
                        from_thread.run_sync(finished.set, token=token)

            self._started, self._running = monotonic(), finished
            Thread(target=work, name="archive-io", daemon=True).start()
            # A caller can leave without losing ownership of a still-running syscall.
            await finished.wait()
            return result.result()

    @contextmanager
    def directory(self, *, create: bool) -> Generator[int]:
        with ExitStack() as stack:
            fd = os.open("/", _DIRECTORY)
            stack.callback(os.close, fd)
            parent = fd
            for part in self.mount.parts[1:]:
                parent = fd
                fd = os.open(part, _DIRECTORY, dir_fd=parent)
                stack.callback(os.close, fd)
            mount_id = _mount_id(fd)
            if mount_id == _mount_id(parent):
                raise SpoolError("archive_mount_absent")
            local = self._local.stat()
            if os.path.samestat(os.fstat(fd), local):
                raise SpoolError("unsafe_archive_path")
            for part in self.path.relative_to(self.mount).parts:
                if create:
                    with suppress(FileExistsError):
                        os.mkdir(part, 0o700, dir_fd=fd)
                    # Also persist a child left by an interrupted earlier mkdir.
                    os.fsync(fd)
                fd = os.open(part, _DIRECTORY, dir_fd=fd)
                stack.callback(os.close, fd)
                if _mount_id(fd) != mount_id or os.path.samestat(os.fstat(fd), local):
                    raise SpoolError("unsafe_archive_path")
            yield fd

    def probe(self) -> None:
        with self.directory(create=True) as fd:
            if os.fstatvfs(fd).f_flag & os.ST_RDONLY:
                raise SpoolError("archive_read_only")

    @staticmethod
    def _verified(fd: int, ref: ArchiveReference, *, sync: bool) -> bytes:
        if not re.fullmatch(r"[a-f0-9]{64}", ref.record_id):
            raise SpoolError("invalid_record_id")
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        with os.fdopen(os.open(ref.record_id, flags, dir_fd=fd), "rb") as stream:
            meta = os.fstat(stream.fileno())
            if meta.st_mode != stat.S_IFREG | 0o400 or _mount_id(stream.fileno()) != _mount_id(fd):
                raise SpoolError("unsafe_archive_record")
            if meta.st_size != ref.size_bytes:
                raise SpoolError("archive_checksum_mismatch")
            data = stream.read(ref.size_bytes + 1)
            if len(data) != ref.size_bytes or sha256(data).hexdigest() != ref.checksum:
                raise SpoolError("archive_checksum_mismatch")
            if sync:
                os.fsync(stream.fileno())
        return data

    def _read(self, ref: ArchiveReference) -> bytes:
        with self.directory(create=False) as fd:
            return self._verified(fd, ref, sync=False)

    async def read(self, ref: ArchiveReference) -> bytes:
        try:
            with fail_after(_IO_TIMEOUT):
                return await self.io(self._read, ref)
        except TimeoutError:
            raise SpoolError("archive_io_timeout") from None

    def publish(self, ref: ArchiveReference, data: bytes) -> None:
        with self.directory(create=True) as fd:
            stage = f".stage-{ref.record_id}"
            try:
                self._verified(fd, ref, sync=True)
            except FileNotFoundError:
                # Only this single writer owns staging names on this destination.
                self._discard_stage(fd, stage)
                flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_CREAT | os.O_EXCL
                with os.fdopen(os.open(stage, flags, 0o600, dir_fd=fd), "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fchmod(stream.fileno(), 0o400)
                    os.fsync(stream.fileno())
                with suppress(FileExistsError):
                    os.link(
                        stage, ref.record_id, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False
                    )
                self._verified(fd, ref, sync=True)
            self._discard_stage(fd, stage)
            os.fsync(fd)

    @staticmethod
    def _discard_stage(fd: int, stage: str) -> None:
        with suppress(FileNotFoundError):
            os.unlink(stage, dir_fd=fd)


@dataclass(frozen=True, slots=True)
class ArchiveStatus:
    enabled: bool
    state: Literal["disabled", "pending", "healthy", "unavailable"]
    last_error: str | None = None


async def _pinned(session: AsyncSession, record_id: str) -> bool:
    owner = select(request_capacity.c.request_id).where(
        (request_capacity.c.input_record_id == record_id)
        | (request_capacity.c.output_record_id == record_id)
    )
    request_id = await session.scalar(owner)
    if request_id is None:
        return False
    latest = await session.scalar(
        select(request_events.c.kind)
        .where(request_events.c.request_id == request_id)
        .order_by(request_events.c.sequence.desc())
        .limit(1)
    )
    return latest not in {"completed", "failed", "resolved"}


class ArchiveDelivery:
    def __init__(self, spool: Spool, database: Database) -> None:
        self.spool, self.database = spool, database
        self._status = ArchiveStatus(
            spool.archive is not None, "disabled" if spool.archive is None else "pending"
        )

    async def status(self) -> ArchiveStatus:
        if self.spool.archive is not None and self.spool.archive.overdue:
            return ArchiveStatus(True, "unavailable", "archive_io_timeout")
        return self._status

    async def run(self) -> None:
        archive = self.spool.archive
        if archive is None:
            return
        while True:
            try:
                work = await self._deliver(archive)
                self._status = ArchiveStatus(True, "pending" if work else "healthy")
                if work:
                    continue
            except SpoolError as error:
                self._status = ArchiveStatus(True, "unavailable", error.code)
            except OSError as error:
                code = {
                    errno.EROFS: "archive_read_only",
                    errno.ENOSPC: "archive_full",
                    errno.EDQUOT: "archive_full",
                }.get(error.errno or 0, "archive_io_error")
                self._status = ArchiveStatus(True, "unavailable", code)
            except SQLAlchemyError:
                self._status = ArchiveStatus(True, "unavailable", "archive_database_error")
            await sleep(_POLL)

    async def _deliver(self, archive: MountedArchive) -> bool:
        joined = select(archive_references).join(archive_outbox)
        async with self.database.session() as session:
            pending = (
                await session.execute(
                    joined.where(archive_outbox.c.state == "pending")
                    .order_by(archive_references.c.record_id)
                    .limit(32)
                )
            ).all()
        for row in pending:
            ref = ArchiveReference(row.record_id, row.checksum, row.size_bytes)
            await archive.io(archive.publish, ref, await self.spool.read_local(ref))
            async with self.database.write() as session:
                await session.execute(
                    update(archive_outbox)
                    .where(archive_outbox.c.record_id == ref.record_id)
                    .values(state="archived")
                )
            await self._reclaim(ref)
        if pending:
            return True

        # Revisit local survivors of an ACK/unlink crash or a previously live claim.
        # Iterate the bounded local spool, not the unbounded remote history.
        ids = sorted(await self.spool.local_ids())
        for i in range(0, len(ids), 128):
            async with self.database.session() as session:
                rows = (
                    await session.execute(
                        joined.where(
                            archive_outbox.c.state == "archived",
                            archive_outbox.c.record_id.in_(ids[i : i + 128]),
                        )
                    )
                ).all()
                reclaimable = [r for r in rows if not await _pinned(session, r.record_id)]
            for row in reclaimable:
                ref = ArchiveReference(row.record_id, row.checksum, row.size_bytes)
                await archive.io(archive.publish, ref, await self.spool.read_local(ref))
                await self._reclaim(ref)
        await archive.io(archive.probe)
        return False

    async def _reclaim(self, ref: ArchiveReference) -> None:
        async with self.spool.capacity_write() as session:
            if not await _pinned(session, ref.record_id):
                await to_thread.run_sync(self.spool.reclaim, ref)
