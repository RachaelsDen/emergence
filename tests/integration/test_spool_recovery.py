import json
from hashlib import sha256
from pathlib import Path

import pytest
from anyio import Path as AsyncPath
from pydantic import JsonValue
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from test_migrations import migrated_path as migrated_path

from emergence.storage.archive import RecordKey, Spool, SpoolError
from emergence.storage.database import Database
from emergence.storage.repositories import list_archive_references
from emergence.telemetry.logging import Redactor


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
