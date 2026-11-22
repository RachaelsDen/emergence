import json
import os
import stat
from hashlib import sha256
from pathlib import Path
from threading import Event

import pytest
from anyio import Path as AsyncPath
from anyio import create_task_group, fail_after, to_thread
from pydantic import JsonValue
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from test_migrations import migrated_path as migrated_path

from emergence.storage.archive import RecordKey, Spool, SpoolError, SpoolStats
from emergence.storage.database import Database
from emergence.storage.repositories import list_archive_references
from emergence.telemetry.logging import Redactor


@pytest.mark.anyio
@pytest.mark.parametrize("replace", [False, True])
async def test_lost_live_spool_is_never_recreated(
    migrated_path: Path, tmp_path: Path, replace: bool
) -> None:
    directory = tmp_path / "spool"
    async with Database(migrated_path) as db:
        spool = Spool(directory, db, Redactor())
        await spool.recover()
        await spool.publish(RecordKey("retained", "fixture"), "unarchived")
        directory.rename(tmp_path / "missing-spool")
        if replace:
            directory.mkdir(mode=0o700)

        # Publication must detect this even when readiness has not run yet.
        with pytest.raises(SpoolError, match="spool_directory_lost"):
            await spool.publish(RecordKey("another", "fixture"), "must not appear")
        assert not await spool.readiness()
        with pytest.raises(SpoolError, match="spool_directory_lost"):
            async with spool.capacity_write():
                pytest.fail("a lost spool must not admit work")
        assert directory.exists() is replace
        if replace:
            assert not tuple(directory.iterdir())


@pytest.mark.anyio
async def test_spool_scan_runs_outside_writer_and_publish_updates_usage(
    migrated_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = Event(), Event()
    directory = tmp_path / "spool"
    directory.mkdir(mode=0o700)
    (directory / ".stage-incomplete").write_bytes(b"s" * 160)
    async with Database(migrated_path) as db:
        spool = Spool(directory, db, Redactor(), limit_bytes=400, headroom_bytes=0)
        stats = spool._stats  # pyright: ignore[reportPrivateUsage]
        scans = 0

        def paused_scan() -> SpoolStats:
            nonlocal scans
            scans += 1
            entered.set()
            if not release.wait(4):
                raise AssertionError("scan was not released")
            return stats()

        async def publish() -> None:
            await spool.publish(RecordKey("first", "fixture"), "x" * 150)

        monkeypatch.setattr(spool, "_stats", paused_scan)
        with fail_after(5):
            async with create_task_group() as tasks:
                tasks.start_soon(publish)
                try:
                    assert await to_thread.run_sync(entered.wait, 2)
                    async with db.write() as session:
                        await session.execute(text("INSERT INTO providers VALUES ('p', 'zai')"))
                finally:
                    release.set()
        # Both the staged bytes and the committed file count against the limit.
        with pytest.raises(SpoolError, match="spool_capacity_exhausted"):
            await spool.publish(RecordKey("second", "fixture"), "small")
        assert scans == 1

        (directory / ".stage-incomplete").unlink()
        await spool.publish(RecordKey("second", "fixture"), "small")
    assert scans == 2


@pytest.mark.anyio
@pytest.mark.parametrize("sync_directory", [False, True])
async def test_publication_fsync_does_not_hold_writer_and_precedes_reference(
    migrated_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sync_directory: bool
) -> None:
    entered, release = Event(), Event()
    directory = tmp_path / "spool"
    key = RecordKey("record", "fixture")
    path = directory / key.record_id
    async with Database(migrated_path) as db:
        spool = Spool(directory, db, Redactor())
        await spool.recover()
        fsync = os.fsync

        def pause(fd: int) -> None:
            metadata = os.fstat(fd)
            should_pause = (
                metadata.st_ino == directory.stat().st_ino and path.exists()
                if sync_directory
                else stat.S_ISREG(metadata.st_mode)
            )
            if should_pause and not entered.is_set():
                entered.set()
                if not release.wait(4):
                    raise AssertionError("publication was not released")
            fsync(fd)

        async def publish() -> None:
            await spool.publish(key, "durable only after fsync")

        monkeypatch.setattr(os, "fsync", pause)
        with fail_after(5):
            async with create_task_group() as tasks:
                tasks.start_soon(publish)
                try:
                    assert await to_thread.run_sync(entered.wait, 2)
                    async with db.write() as session:
                        assert await list_archive_references(session) == ()
                        await session.execute(text("INSERT INTO providers VALUES ('p', 'zai')"))
                finally:
                    release.set()
        async with db.session() as session:
            (reference,) = await list_archive_references(session)
        assert reference.record_id == key.record_id
        assert (await spool.read(reference)).endswith(b'"durable only after fsync"')


@pytest.mark.anyio
async def test_publish_redacts_keys_and_values_and_never_replaces_a_record(
    migrated_path: Path, tmp_path: Path
) -> None:
    folder = AsyncPath(tmp_path / "spool")
    key = RecordKey("opaque/key", "request")
    sentinel = "configured-canary-3751"
    payload: JsonValue = {
        "plain": "retained",
        sentinel: [sentinel],
        "Authorization": "recognizable-canary",
    }
    async with Database(migrated_path) as database:
        spool = Spool(Path(folder), database, Redactor((sentinel,)))

        first = await spool.publish(key, payload)
        assert await spool.publish(key, payload) == first
        with pytest.raises(SpoolError, match="record_identity_conflict"):
            await spool.publish(key, {"plain": "different"})

        stored_file = folder / first.record_id
        assert (await folder.stat()).st_mode & 0o777 == 0o700
        assert (await stored_file.stat()).st_mode & 0o777 == 0o400
        stored = await stored_file.read_bytes()
        assert sentinel.encode() not in stored
        assert b"recognizable-canary" not in stored
        verified = await spool.read(first)
        assert sha256(verified).hexdigest() == first.checksum
        assert len(verified) == first.size_bytes
        assert json.loads(verified.split(b"\n", maxsplit=2)[2]) == {
            "plain": "retained",
            "[REDACTED]": ["[REDACTED]"],
            "Authorization": "[REDACTED]",
        }
        async with database.session() as session:
            assert await list_archive_references(session) == (first,)


@pytest.mark.anyio
async def test_recovery_ingests_orphan_file_and_removes_interrupted_staging(
    migrated_path: Path, tmp_path: Path
) -> None:
    directory = tmp_path / "spool"
    key = RecordKey("request-1", "response", 1)
    async with Database(migrated_path) as database:
        spool = Spool(directory, database, Redactor())
        async with database.write() as session:
            await session.execute(
                text(
                    "CREATE TRIGGER interrupt_outbox BEFORE INSERT ON archive_outbox "
                    "BEGIN SELECT RAISE(ABORT, 'simulated interruption'); END"
                )
            )
        with pytest.raises(IntegrityError, match="simulated interruption"):
            await spool.publish(key, "durable response")
        orphan = AsyncPath(directory / key.record_id)
        assert await orphan.is_file()
        staged = AsyncPath(directory / ".stage-test0001")
        await staged.write_bytes(b"incomplete")
        async with database.session() as session:
            assert await list_archive_references(session) == ()
        async with database.write() as session:
            await session.execute(text("DROP TRIGGER interrupt_outbox"))

    async with Database(migrated_path) as database:
        spool = Spool(directory, database, Redactor())
        await spool.recover()
        await spool.recover()

        assert not await staged.exists()
        async with database.session() as session:
            (reference,) = await list_archive_references(session)
            assert (await session.execute(text("SELECT * FROM archive_outbox"))).all() == [
                (key.record_id, "pending")
            ]
        assert (await spool.read(reference)).split(b"\n", maxsplit=2)[2] == b'"durable response"'


@pytest.mark.anyio
async def test_recovery_rebuilds_missing_projection_without_resetting_existing_state(
    migrated_path: Path, tmp_path: Path
) -> None:
    async with Database(migrated_path) as database:
        spool = Spool(tmp_path / "spool", database, Redactor())
        archived = await spool.publish(RecordKey("r", "input"), "input")
        missing = await spool.publish(RecordKey("r", "output"), "output")
        async with database.write() as session:
            await session.execute(
                text("UPDATE archive_outbox SET state = 'archived' WHERE record_id = :id"),
                {"id": archived.record_id},
            )
            await session.execute(
                text("DELETE FROM archive_outbox WHERE record_id = :id"),
                {"id": missing.record_id},
            )

        await spool.recover()

        async with database.session() as session:
            rows = await session.execute(text("SELECT record_id, state FROM archive_outbox"))
            assert set(rows.all()) == {
                (archived.record_id, "archived"),
                (missing.record_id, "pending"),
            }


@pytest.mark.anyio
@pytest.mark.parametrize("damage", ["corrupt", "missing", "wrong_path"])
async def test_recovery_fails_closed_when_authoritative_bytes_are_damaged(
    migrated_path: Path, tmp_path: Path, damage: str
) -> None:
    directory = tmp_path / "spool"
    async with Database(migrated_path) as database:
        spool = Spool(directory, database, Redactor())
        reference = await spool.publish(RecordKey("request", "input"), "original")
        record = AsyncPath(directory / reference.record_id)
        if damage == "corrupt":
            await record.chmod(0o600)
            await record.write_bytes(b"corrupt")
            await record.chmod(0o400)
        elif damage == "missing":
            await record.unlink()
        else:
            await record.rename(directory / ("f" * 64))

        code = "missing_record" if damage == "missing" else "record_checksum_mismatch"
        with pytest.raises(SpoolError, match=code):
            await spool.recover()
        async with database.session() as session:
            assert await list_archive_references(session) == (reference,)
