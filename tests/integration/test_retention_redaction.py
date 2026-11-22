import json
from pathlib import Path

import pytest
from anyio import Path as AsyncPath
from test_archive_delivery import mounted, run_until, states
from test_budget_concurrency import NOW, admit, configuration, prepared
from test_migrations import migrated_path as migrated_path

from emergence.storage.archive import RecordKey, Spool
from emergence.storage.database import Database
from emergence.storage.delivery import ArchiveDelivery
from emergence.storage.gateway_state import request_capacity
from emergence.storage.models import request_events
from emergence.telemetry.logging import Redactor


@pytest.mark.anyio
async def test_unreconciled_records_survive_ack_and_redacted_history_survives_resolution(
    migrated_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local, mount = tmp_path / "spool", tmp_path / "mount"
    await AsyncPath(mount).mkdir()
    mounted(monkeypatch, mount)
    request = "uncertain"
    incoming, outgoing = RecordKey(request, "input"), RecordKey(request, "output")
    sentinel = "test-secret-f67d"
    async with Database(migrated_path) as db:
        ledger = await prepared(db, configuration())
        spool = Spool(local, db, Redactor((sentinel,)), archive_path=mount)
        async with db.write() as session:
            await admit(session, ledger, request, 3)
            await session.execute(
                request_capacity.insert().values(
                    request_id=request,
                    input_record_id=incoming.record_id,
                    output_record_id=outgoing.record_id,
                    allocated_bytes=8192,
                    input_tokens_bound=50,
                    output_tokens_bound=3,
                    created_at=NOW.isoformat(),
                )
            )
        for sequence, state in enumerate(("admitted", "dispatched", "ambiguous", "resolved")):
            async with db.write() as session:
                await session.execute(
                    request_events.insert().values(
                        request_id=request,
                        sequence=sequence,
                        kind=state,
                        occurred_at=NOW.isoformat(),
                    )
                )
            if state == "admitted":
                await spool.publish(
                    incoming, {"text": f"keep {sentinel}", "authorization": "canary"}
                )
            if state == "dispatched":
                await spool.publish(outgoing, {"answer": ["complete", sentinel]})
            await run_until(ArchiveDelivery(spool, db), "healthy")
            ids = (
                (incoming.record_id,)
                if state == "admitted"
                else (incoming.record_id, outgoing.record_id)
            )
            assert await states(db) == dict.fromkeys(ids, "archived")
            for record_id in ids:
                copy = await AsyncPath(mount / record_id).read_bytes()
                assert sentinel.encode() not in copy and b"canary" not in copy
                payload = json.loads(copy.split(b"\n", 2)[2])
                assert payload == (
                    {"text": "keep [REDACTED]", "authorization": "[REDACTED]"}
                    if record_id == incoming.record_id
                    else {"answer": ["complete", "[REDACTED]"]}
                )
                original = AsyncPath(local / record_id)
                if state == "resolved":
                    assert not await original.exists()
                else:
                    assert await original.read_bytes() == copy
