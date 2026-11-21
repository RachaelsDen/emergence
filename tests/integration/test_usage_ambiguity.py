"""Malformed wire usage must not settle or redispatch a bounded request."""

from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from _gateway_fixture import NOW, gateway_for, invocation, settings_for
from sqlalchemy import select
from test_migrations import migrated_path as migrated_path

from emergence.inference.gateway import ArchivedOutcome, RequestState
from emergence.storage.archive import RecordKey
from emergence.storage.models import archive_references
from emergence.storage.repositories import ArchiveReference


@pytest.mark.anyio
@pytest.mark.parametrize("malformed_usage", [True, False], ids=["usage", "compressed-body"])
async def test_invalid_wire_response_preserves_hold_and_replays_without_dispatch(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch, malformed_usage: bool
) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-b1-inference")
    payload = {
        "model": "configured-model",
        "choices": [{"message": {"content": "synthetic response"}}],
        "usage": {"prompt_tokens": False, "completion_tokens": False},
    }
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if malformed_usage:
            return httpx.Response(200, json=payload)
        return httpx.Response(
            200, headers={"Content-Encoding": "gzip"}, stream=httpx.ByteStream(b"invalid gzip")
        )

    settings = settings_for()
    async with gateway_for(migrated_path, settings, respond) as (gateway, ledger, database, spool):
        first = await gateway.infer(invocation(), "malformed-usage")
        assert first.state is RequestState.AMBIGUOUS
        assert first.error_code == (
            "provider_schema_error" if malformed_usage else "provider_transport_error"
        )
        assert first.result is None
        record_id = RecordKey(first.request_id, "output").record_id
        async with database.session() as session:
            row = (
                await session.execute(
                    select(archive_references).where(archive_references.c.record_id == record_id)
                )
            ).one()
        record = await spool.read(ArchiveReference(record_id, row.checksum, row.size_bytes))
        outcome = ArchivedOutcome.model_validate_json(record.split(b"\n", 2)[2])
        assert outcome.failure is not None
        if malformed_usage:
            assert outcome.failure.raw_response == payload
        balance = await ledger.balance("p", NOW)
        assert (balance.reserved, balance.spent) == (Decimal("0.03"), Decimal("0"))
        immediate = await gateway.infer(invocation(), "malformed-usage")
        assert immediate.replayed and immediate.state is RequestState.AMBIGUOUS
        assert immediate.error_code == first.error_code

    async with gateway_for(migrated_path, settings, respond) as (gateway, ledger, _, _):
        replay = await gateway.infer(invocation(), "malformed-usage")
        assert replay.replayed and replay.request_id == first.request_id
        assert replay.state is first.state and replay.error_code == first.error_code
        balance = await ledger.balance("p", NOW)
        assert (balance.reserved, balance.spent) == (Decimal("0.03"), Decimal("0"))
    assert calls == ["/v1/chat/completions"]
