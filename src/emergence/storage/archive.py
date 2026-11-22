"""Protected immutable local records and deterministic SQLite outbox recovery."""

import json
import os
import re
import shutil
import stat
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final
from weakref import WeakValueDictionary, finalize

from anyio import Lock, to_thread
from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.telemetry.logging import SECRET_FIELD, Redactor

from .capacity import live_claims
from .database import Database
from .models import archive_outbox, archive_references
from .repositories import (
    ArchiveReference,
    list_archive_references,
    register_archive_reference,
)

_RECORD: Final = re.compile(r"[a-f0-9]{64}")
_STAGING: Final = re.compile(r"\.stage-[a-zA-Z0-9_-]+")
_FORMAT: Final = b"emergence-spool-v1"
type _DirectoryVersion = tuple[int, int, int, int]


class _DirectoryUsage:
    def __init__(self) -> None:
        self.lock = Lock()
        self.snapshot: tuple[_DirectoryVersion, int] | None = None
        self.identity: tuple[int, int] | None = None
        self.lost = False

    def bind(self, path: Path, metadata: os.stat_result) -> None:
        if self.identity is not None:
            return
        # Pin the inode so a removed directory cannot be replaced with the
        # same inode number while the core still owns this spool.
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            pinned = os.fstat(fd)
            identity = (metadata.st_dev, metadata.st_ino)
            if (pinned.st_dev, pinned.st_ino) != identity:
                raise SpoolError("spool_directory_lost")
        except BaseException:
            os.close(fd)
            raise
        finalize(self, os.close, fd)
        self.identity = identity


_DIRECTORY_USAGE: Final[WeakValueDictionary[Path, _DirectoryUsage]] = WeakValueDictionary()


class SpoolError(RuntimeError):
    """A protected spool invariant failed; recovery must fail closed."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RecordKey:
    """Opaque stable identity, record kind and zero-based ordinal."""

    source_id: str
    kind: str
    ordinal: int = 0

    @property
    def record_id(self) -> str:
        identity = json.dumps([self.source_id, self.kind, self.ordinal], ensure_ascii=True)
        return sha256(_FORMAT + b"\0" + identity.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class _Record:
    reference: ArchiveReference
    data: bytes


@dataclass(frozen=True, slots=True)
class SpoolStats:
    used_bytes: int
    records: int
    free_bytes: int


def _scrub(redactor: Redactor, payload: JsonValue) -> JsonValue:
    # Also protect JSON property names, which the logging redactor preserves.
    match payload:
        case dict():
            output: dict[str, JsonValue] = {}
            for key, value in payload.items():
                safe_key = redactor.text(key)
                if safe_key in output:
                    raise SpoolError("redacted_key_collision")
                output[safe_key] = (
                    "[REDACTED]" if SECRET_FIELD.search(key) else (_scrub(redactor, value))
                )
            return output
        case list():
            return [_scrub(redactor, value) for value in payload]
        case _:
            return redactor.value(payload)


def _stamp(record_id: str, body: bytes) -> bytes:
    return sha256(_FORMAT + b"\0" + record_id.encode("ascii") + b"\0" + body).hexdigest().encode()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Spool:
    """Single local spool per database; pass a redactor configured with known secrets."""

    def __init__(
        self,
        path: Path,
        database: Database,
        redactor: Redactor,
        *,
        limit_bytes: int = 256 * 1024 * 1024,
        headroom_bytes: int = 64 * 1024 * 1024,
        archive_path: Path | None = None,
        archive_mount_path: Path | None = None,
    ) -> None:
        from .delivery import MountedArchive

        self._path = path.absolute()
        self._db = database
        self._redactor = redactor
        self._limit = limit_bytes
        self._headroom = headroom_bytes
        # Share accounting across handles to a protected directory in this core.
        self._usage = _DIRECTORY_USAGE.setdefault(self._path, _DirectoryUsage())
        self.archive = (
            MountedArchive(archive_path, archive_mount_path or archive_path, self._path)
            if archive_path is not None
            else None
        )

    def preview(self, key: RecordKey, payload: JsonValue) -> ArchiveReference:
        """Compute exact redacted input storage before financial admission."""
        return self._prepare(key, payload).reference

    async def stats(self) -> SpoolStats:
        return await to_thread.run_sync(self._stats)

    async def readiness(self) -> bool:
        """Check the local spool without creating it or contacting the archive."""
        try:
            return await to_thread.run_sync(self._directory_usable)
        except (OSError, SpoolError):
            return False

    def _directory_usable(self) -> bool:
        self._check_directory()
        return os.access(self._path, os.R_OK | os.W_OK | os.X_OK, effective_ids=True)

    @asynccontextmanager
    async def _capacity_guard(self) -> AsyncGenerator[None]:
        async with self._usage.lock:
            await to_thread.run_sync(self._refresh_usage)
            yield

    @asynccontextmanager
    async def capacity_write(self) -> AsyncGenerator[AsyncSession]:
        """Refresh the usage cache before taking the SQLite writer."""
        async with self._capacity_guard(), self._db.write() as session:
            yield session

    async def ensure_capacity(
        self,
        session: AsyncSession,
        additional_bytes: int,
        *,
        claim_request_id: str | None = None,
        record_id: str | None = None,
        record_size: int = 0,
    ) -> None:
        """Check current claims and disk usage while holding the spool guard."""
        claims = await live_claims(session)
        if any(claim.remaining_bytes < 0 for claim in claims):
            raise SpoolError("spool_claim_exceeded")
        reduction = 0
        if claim_request_id is not None:
            own = next((claim for claim in claims if claim.request_id == claim_request_id), None)
            if own is None or record_id not in (own.input_record_id, own.output_record_id):
                raise SpoolError("spool_claim_missing")
            if record_size > own.remaining_bytes:
                raise SpoolError("spool_claim_exceeded")
            reduction = record_size
        remaining = sum(claim.remaining_bytes for claim in claims) - reduction
        used, free = await to_thread.run_sync(self._capacity_usage)
        if used + remaining + additional_bytes > self._limit:
            raise SpoolError("spool_capacity_exhausted")
        if free < self._headroom + remaining + additional_bytes:
            raise SpoolError("local_disk_headroom_exhausted")

    async def publish(
        self, key: RecordKey, payload: JsonValue, *, claim_request_id: str | None = None
    ) -> ArchiveReference:
        """Redact and durably publish an immutable JSON record, then commit its reference."""
        record = self._prepare(key, payload)
        reference = record.reference
        # Admissions, reclamation, and other publications use the same guard.
        # They cannot spend this capacity between the check and reference commit.
        async with self._capacity_guard():
            present = False
            async with self._db.session() as session:
                existing = (
                    await session.execute(
                        select(archive_references).where(
                            archive_references.c.record_id == reference.record_id
                        )
                    )
                ).one_or_none()
                if existing is None:
                    present = await to_thread.run_sync((self._path / reference.record_id).exists)
                    await self.ensure_capacity(
                        session,
                        0 if present else reference.size_bytes,
                        claim_request_id=claim_request_id,
                        record_id=reference.record_id,
                        record_size=reference.size_bytes,
                    )
            if existing is not None:
                if (existing.checksum, existing.size_bytes) != (
                    reference.checksum,
                    reference.size_bytes,
                ):
                    raise SpoolError("record_identity_conflict")
                try:
                    await self.read_local(reference)
                except SpoolError as error:
                    # An identical record may already have been delivered and reclaimed.
                    if error.code != "missing_record":
                        raise
                    async with self._db.session() as session:
                        if not await _archived(session, reference):
                            raise
            else:
                await to_thread.run_sync(self._publish_counted, record, present)
            async with self._db.write() as session:
                await register_archive_reference(session, reference)
        return reference

    async def read(self, reference: ArchiveReference) -> bytes:
        """Return the verified on-disk bytes covered by the reference's SHA-256."""
        try:
            return await self.read_local(reference)
        except SpoolError as error:
            if error.code != "missing_record" or self.archive is None:
                raise
            async with self._db.session() as session:
                if not await _archived(session, reference):
                    raise
        return await self.archive.read(reference)

    async def read_local(self, reference: ArchiveReference) -> bytes:
        return await to_thread.run_sync(self._read_reference, reference)

    async def local_ids(self) -> set[str]:
        return await to_thread.run_sync(
            lambda: {p.name for p in self._path.iterdir() if _RECORD.fullmatch(p.name)}
        )

    def reclaim(self, reference: ArchiveReference) -> None:
        """Local-only; caller holds the writer after verified delivery and committed ACK."""
        if not _RECORD.fullmatch(reference.record_id):
            raise SpoolError("invalid_record_id")
        used = self._indexed_bytes()
        self._usage.snapshot = None
        path = self._path / reference.record_id
        size = path.lstat().st_size
        path.unlink()
        _fsync_directory(self._path)
        self._usage.snapshot = self._directory_version(), used - size

    def _read_reference(self, reference: ArchiveReference) -> bytes:
        self._check_directory()
        record = self._read(reference.record_id)
        if record.reference != reference:
            raise SpoolError("reference_mismatch")
        return record.data

    async def recover(self) -> None:
        """Reconcile durable files and rows; do not silently drop corrupt or missing data."""
        async with self._usage.lock:
            self._usage.snapshot = None
            async with self._db.session() as session:
                known = await list_archive_references(session)
                archived = set(
                    await session.scalars(
                        select(archive_outbox.c.record_id).where(
                            archive_outbox.c.state == "archived"
                        )
                    )
                )
            recovered, version = await to_thread.run_sync(self._scan, known, archived)
            async with self._db.write() as session:
                for reference in recovered:
                    await register_archive_reference(session, reference)
            self._usage.snapshot = version, sum(reference.size_bytes for reference in recovered)

    def _ensure_directory(self) -> None:
        if self._usage.identity is not None:
            self._check_directory()
            return
        try:
            # Create only the leaf. Provision its local parent before launching.
            self._path.mkdir(mode=0o700)
        except FileExistsError:
            self._check_directory()
        _fsync_directory(self._path.parent)

    def _check_directory(self) -> os.stat_result:
        if self._usage.lost:
            raise SpoolError("spool_directory_lost")
        identity = self._usage.identity
        try:
            metadata = self._path.lstat()
        except FileNotFoundError:
            if identity is None:
                raise
            self._usage.lost = True
            raise SpoolError("spool_directory_lost") from None
        if identity is not None and identity != (metadata.st_dev, metadata.st_ino):
            self._usage.lost = True
            raise SpoolError("spool_directory_lost")
        if metadata.st_uid != os.geteuid() or metadata.st_mode != stat.S_IFDIR | 0o700:
            raise SpoolError("unsafe_spool_directory")
        return metadata

    def _directory_version(self) -> _DirectoryVersion:
        metadata = self._check_directory()
        return metadata.st_dev, metadata.st_ino, metadata.st_ctime_ns, metadata.st_mtime_ns

    def _refresh_usage(self) -> None:
        if self._usage.identity is None:
            self._ensure_directory()
            self._usage.bind(self._path, self._check_directory())
        version = self._directory_version()
        snapshot = self._usage.snapshot
        if snapshot is not None and snapshot[0] == version:
            return
        self._usage.snapshot = None
        scanned = self._stats()
        if self._directory_version() != version:
            raise SpoolError("spool_contents_changed")
        self._usage.snapshot = version, scanned.used_bytes

    def _indexed_bytes(self) -> int:
        snapshot = self._usage.snapshot
        if snapshot is None or snapshot[0] != self._directory_version():
            raise SpoolError("spool_contents_changed")
        return snapshot[1]

    def _capacity_usage(self) -> tuple[int, int]:
        return self._indexed_bytes(), shutil.disk_usage(self._path).free

    def _publish_counted(self, record: _Record, present: bool) -> None:
        used = self._indexed_bytes()
        self._usage.snapshot = None
        self._publish(record)
        self._usage.snapshot = (
            self._directory_version(),
            used + (0 if present else record.reference.size_bytes),
        )

    def _read(self, record_id: str) -> _Record:
        if _RECORD.fullmatch(record_id) is None:
            raise SpoolError("invalid_record_id")
        try:
            fd = os.open(self._path / record_id, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            raise SpoolError("missing_record") from None
        with os.fdopen(fd, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if metadata.st_uid != os.geteuid() or metadata.st_mode != stat.S_IFREG | 0o400:
                raise SpoolError("unsafe_record")
            data = stream.read()
        version, _, remainder = data.partition(b"\n")
        checksum, _, content = remainder.partition(b"\n")
        if version != _FORMAT or not content or checksum != _stamp(record_id, content):
            raise SpoolError("record_checksum_mismatch")
        return _Record(ArchiveReference(record_id, sha256(data).hexdigest(), len(data)), data)

    def _prepare(self, key: RecordKey, payload: JsonValue) -> _Record:
        scrubbed = _scrub(self._redactor, payload)
        try:
            body = json.dumps(
                scrubbed, ensure_ascii=True, sort_keys=True, allow_nan=False, separators=(",", ":")
            ).encode("ascii")
        except ValueError:
            raise SpoolError("invalid_json_payload") from None
        record_id = key.record_id
        data = b"\n".join((_FORMAT, _stamp(record_id, body), body))
        reference = ArchiveReference(record_id, sha256(data).hexdigest(), len(data))
        return _Record(reference, data)

    def _stats(self) -> SpoolStats:
        self._check_directory()
        total = 0
        records = 0
        for entry in self._path.iterdir():
            try:
                metadata = entry.lstat()
            except FileNotFoundError:
                # An unlocked status snapshot may race publication or reclamation.
                continue
            if not stat.S_ISREG(metadata.st_mode) or not (
                _RECORD.fullmatch(entry.name) or _STAGING.fullmatch(entry.name)
            ):
                raise SpoolError("unexpected_spool_entry")
            total += metadata.st_size
            records += int(_RECORD.fullmatch(entry.name) is not None)
        return SpoolStats(total, records, shutil.disk_usage(self._path).free)

    def _publish(self, record: _Record) -> None:
        reference, data = record.reference, record.data
        record_id = reference.record_id
        self._ensure_directory()
        if (self._path / record_id).exists():
            if self._read(record_id).reference != reference:
                raise SpoolError("record_identity_conflict")
            _fsync_directory(self._path)
            return
        with NamedTemporaryFile(dir=self._path, prefix=".stage-", delete=False) as staged:
            try:
                staged.write(data)
                staged.flush()
                os.fchmod(staged.fileno(), 0o400)
                os.fsync(staged.fileno())
                try:
                    os.link(staged.name, self._path / record_id)
                except FileExistsError:
                    if self._read(record_id).reference != reference:
                        raise SpoolError("record_identity_conflict") from None
                # A previous interrupted publication may have linked but not synced.
                _fsync_directory(self._path)
            finally:
                Path(staged.name).unlink()
                _fsync_directory(self._path)

    def _scan(
        self, known: tuple[ArchiveReference, ...], archived: set[str]
    ) -> tuple[tuple[ArchiveReference, ...], _DirectoryVersion]:
        # Startup may provision a new spool only when there is no local-only evidence.
        try:
            metadata = self._check_directory()
        except FileNotFoundError:
            if any(reference.record_id not in archived for reference in known):
                raise SpoolError("missing_record") from None
            self._ensure_directory()
            metadata = self._check_directory()
        self._usage.bind(self._path, metadata)
        discovered: dict[str, ArchiveReference] = {}
        stages: list[Path] = []
        for entry in sorted(self._path.iterdir()):
            if _RECORD.fullmatch(entry.name):
                discovered[entry.name] = self._read(entry.name).reference
            elif _STAGING.fullmatch(entry.name) and stat.S_ISREG(entry.lstat().st_mode):
                stages.append(entry)
            else:
                raise SpoolError("unexpected_spool_entry")
        for reference in known:
            observed = discovered.get(reference.record_id)
            if observed is None:
                if reference.record_id in archived:
                    continue
                raise SpoolError("missing_record")
            if observed != reference:
                raise SpoolError("reference_mismatch")
        for stage in stages:
            stage.unlink()
        _fsync_directory(self._path)
        return tuple(discovered.values()), self._directory_version()


async def _archived(session: AsyncSession, reference: ArchiveReference) -> bool:
    return (
        await session.scalar(
            select(archive_outbox.c.state)
            .join(archive_references)
            .where(
                archive_references.c.record_id == reference.record_id,
                archive_references.c.checksum == reference.checksum,
                archive_references.c.size_bytes == reference.size_bytes,
            )
        )
    ) == "archived"
