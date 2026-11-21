import errno
import os
from pathlib import Path
from threading import Event

import pytest
from anyio import Path as AsyncPath
from anyio import create_task_group, fail_after, sleep, to_thread
from sqlalchemy import select, text
from test_migrations import migrated_path as migrated_path

from emergence.storage import delivery
from emergence.storage.archive import RecordKey, Spool, SpoolError
from emergence.storage.database import Database
from emergence.storage.delivery import ArchiveDelivery, ArchiveStatus
from emergence.storage.models import archive_outbox
from emergence.telemetry.logging import Redactor


def fd_path(fd: int) -> Path:
    return Path(os.readlink(f"/proc/self/fd/{fd}"))


def mounted(mp: pytest.MonkeyPatch, mount: Path) -> None:
    def mount_id(fd: int) -> int:
        return 2 if fd_path(fd).is_relative_to(mount) else 1

    mp.setattr(delivery, "_mount_id", mount_id)


async def run_until(worker: ArchiveDelivery, state: str) -> ArchiveStatus:
    with fail_after(5):
        async with create_task_group() as tasks:
            tasks.start_soon(worker.run)
            while True:
                if (await worker.status()).state == state:
                    break
                await sleep(0.01)
            tasks.cancel_scope.cancel()
    return await worker.status()


async def states(db: Database) -> dict[str, str]:
    async with db.session() as session:
        rows = await session.execute(select(archive_outbox))
        return {row.record_id: row.state for row in rows}


@pytest.mark.anyio
async def test_unmounted_writable_directory_never_receives_bytes_and_remount_recovers(
    migrated_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount, local = tmp_path / "nas", tmp_path / "spool"
    await AsyncPath(mount).mkdir()
    dest = mount / "history"
    key = RecordKey("operator-configuration", "fixture")
    async with Database(migrated_path) as db:
        spool = Spool(local, db, Redactor(), archive_path=dest, archive_mount_path=mount)
        ref = await spool.publish(key, "config")
        envelope = await spool.read(ref)
        worker = ArchiveDelivery(spool, db)
        assert (await run_until(worker, "unavailable")).last_error == "archive_mount_absent"
        assert not await AsyncPath(dest).exists()
        assert await states(db) == {ref.record_id: "pending"}
        mounted(monkeypatch, mount)
        await run_until(worker, "healthy")
        assert await states(db) == {ref.record_id: "archived"}
        assert (await spool.stats()).used_bytes == 0
        assert (
            await spool.read(ref) == envelope == await AsyncPath(dest / ref.record_id).read_bytes()
        )

    monkeypatch.undo()  # Lose the mount; the file below MUST NOT serve as a fallback.
    async with Database(migrated_path) as db:
        spool = Spool(local, db, Redactor(), archive_path=dest, archive_mount_path=mount)
        await spool.recover()
        assert await db.readiness()
        assert await spool.publish(key, "config") == ref
        with pytest.raises(SpoolError, match="archive_mount_absent"):
            await spool.read(ref)
        mounted(monkeypatch, mount)
        assert await spool.read(ref) == envelope
        await AsyncPath(dest / ref.record_id).chmod(0o600)
        await AsyncPath(dest / ref.record_id).write_bytes(b"x" * len(envelope))
        await AsyncPath(dest / ref.record_id).chmod(0o400)
        with pytest.raises(SpoolError, match="archive_checksum_mismatch"):
            await spool.read(ref)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("failure", "code"),
    [
        ("readonly", "archive_read_only"),
        ("full", "archive_full"),
        ("dirsync", "archive_io_error"),
        ("corrupt", "archive_checksum_mismatch"),
        ("ack", "archive_database_error"),
        ("unlink", "archive_io_error"),
    ],
)
async def test_fault_boundaries_keep_original_and_resume_without_replacing_valid_copy(
    migrated_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, code: str
) -> None:
    local, mount = tmp_path / "spool", tmp_path / "nas"
    await AsyncPath(mount).mkdir()
    mounted(monkeypatch, mount)
    async with Database(migrated_path) as db:
        spool = Spool(local, db, Redactor(), archive_path=mount)
        ref = await spool.publish(RecordKey("q", "report"), "durable history")
        original = await spool.read(ref)
        dest = AsyncPath(mount / ref.record_id)
        with monkeypatch.context() as fault:
            fsync = os.fsync
            unlink = os.unlink

            def sync(fd: int) -> None:
                path = fd_path(fd)
                if path.is_relative_to(mount):
                    if failure == "readonly":
                        raise OSError(errno.EROFS, "private diagnostic")
                    if failure == "full":
                        raise OSError(errno.ENOSPC, "private diagnostic")
                    if failure == "dirsync" and path == mount:
                        raise OSError(errno.EIO, "private diagnostic")
                fsync(fd)

            def remove(path: str | Path, *, dir_fd: int | None = None) -> None:
                if failure == "unlink" and path == local / ref.record_id:
                    raise OSError(errno.EIO, "private diagnostic")
                unlink(path, dir_fd=dir_fd)

            fault.setattr(os, "fsync", sync)
            fault.setattr(os, "unlink", remove)
            if failure == "corrupt":
                await dest.write_bytes(b"X" * len(original))
                await dest.chmod(0o400)
            if failure == "ack":
                async with db.write() as session:
                    await session.execute(
                        text(
                            "CREATE TRIGGER stop_ack BEFORE UPDATE ON archive_outbox "
                            "BEGIN SELECT RAISE(ABORT, 'delivered but unacknowledged'); END"
                        )
                    )
            worker = ArchiveDelivery(spool, db)
            assert (await run_until(worker, "unavailable")).last_error == code
        assert await states(db) == {ref.record_id: "archived" if failure == "unlink" else "pending"}
        assert await AsyncPath(local / ref.record_id).read_bytes() == original
        if failure == "corrupt":
            assert await dest.read_bytes() == b"X" * len(original)
            await dest.unlink()  # Explicit operator remediation; never automatic.
        if failure == "ack":
            async with db.write() as session:
                await session.execute(text("DROP TRIGGER stop_ack"))
        inode = (await dest.stat()).st_ino if failure in {"ack", "dirsync", "unlink"} else None

    async with Database(migrated_path) as db:
        spool = Spool(local, db, Redactor(), archive_path=mount)
        await spool.recover()
        await run_until(ArchiveDelivery(spool, db), "healthy")
        assert await states(db) == {ref.record_id: "archived"}
        assert await spool.read(ref) == original and (await spool.stats()).records == 0
        if inode is not None:
            assert (await dest.stat()).st_ino == inode
        assert [p.name async for p in AsyncPath(mount).iterdir()] == [ref.record_id]


@pytest.mark.anyio
async def test_mount_and_record_symlinks_cannot_redirect_publication(
    migrated_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount, outside, local = tmp_path / "nas", tmp_path / "outside", tmp_path / "spool"
    await AsyncPath(mount).mkdir()
    await AsyncPath(outside).mkdir()
    mounted(monkeypatch, mount)
    async with Database(migrated_path) as db:
        with pytest.raises(SpoolError, match="unsafe_archive_path"):
            Spool(
                local, db, Redactor(), archive_path=mount / "../outside", archive_mount_path=mount
            )
        dest = AsyncPath(mount / "escape")
        await dest.symlink_to(outside)
        for root in (mount, Path(dest)):
            spool = Spool(local, db, Redactor(), archive_path=Path(dest), archive_mount_path=root)
            ref = await spool.publish(RecordKey("a", "b"), "secret")
            await run_until(ArchiveDelivery(spool, db), "unavailable")
            assert not await AsyncPath(outside / ref.record_id).exists()
        await dest.unlink()
        await dest.mkdir()
        await (dest / ref.record_id).symlink_to(outside / "victim")
        spool = Spool(local, db, Redactor(), archive_path=Path(dest), archive_mount_path=mount)
        await run_until(ArchiveDelivery(spool, db), "unavailable")
        assert not await AsyncPath(outside / "victim").exists()
        assert await states(db) == {ref.record_id: "pending"}


@pytest.mark.anyio
async def test_hung_syscall_times_out_without_holding_writer_or_starting_another(
    migrated_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount = tmp_path / "nas"
    await AsyncPath(mount).mkdir()
    mounted(monkeypatch, mount)
    monkeypatch.setattr(delivery, "_IO_TIMEOUT", 0.15)
    monkeypatch.setattr(delivery, "_POLL", 0.01)
    entered, release = Event(), Event()
    blocked = 0
    fsync = os.fsync

    def sync(fd: int) -> None:
        nonlocal blocked
        if fd_path(fd).is_relative_to(mount) and not release.is_set():
            blocked += 1
            entered.set()
            if not release.wait(5):
                raise AssertionError("worker shutdown blocked")
        fsync(fd)

    monkeypatch.setattr(os, "fsync", sync)
    async with Database(migrated_path) as db:
        spool = Spool(tmp_path / "spool", db, Redactor(), archive_path=mount)
        ref = await spool.publish(RecordKey("a", "b"), "original")
        worker = ArchiveDelivery(spool, db)
        try:
            with fail_after(2):
                async with create_task_group() as group:
                    group.start_soon(worker.run)
                    assert await to_thread.run_sync(entered.wait, 1)
                    async with db.write() as session:
                        await session.execute(text("SELECT 1"))
                    await sleep(0.35)  # Cross two status deadlines while the same syscall is held.
                    assert (await worker.status()).last_error == "archive_io_timeout"
                    group.cancel_scope.cancel()
            assert blocked == 1
            assert await states(db) == {ref.record_id: "pending"}
        finally:
            release.set()
        await run_until(worker, "healthy")
        assert (await spool.stats()).records == 0
