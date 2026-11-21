"""Malformed wire usage must not settle or redispatch a bounded request."""

from decimal import Decimal
from pathlib import Path
from typing import Literal

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
@pytest.mark.parametrize(
    ("defect", "response_model"),
    [
        ("usage", "configured-model"),
        ("compressed-body", "configured-model"),
        ("model", "other-model"),
        ("model", "m"),
        ("model", "Configured-model"),
        ("model", "configured-model "),
    ],
)
async def test_invalid_wire_response_preserves_hold_and_replays_without_dispatch(
    migrated_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: Literal["usage", "compressed-body", "model"],
    response_model: str,
) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-b1-inference")
    payload = {
        "model": response_model,
        "choices": [{"message": {"content": "synthetic response"}}],
        "usage": (
            {"prompt_tokens": False, "completion_tokens": False}
            if defect == "usage"
            else {"prompt_tokens": 5, "completion_tokens": 3}
        ),
    }
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if defect != "compressed-body":
            return httpx.Response(200, json=payload)
        return httpx.Response(
            200, headers={"Content-Encoding": "gzip"}, stream=httpx.ByteStream(b"invalid gzip")
        )

    settings = settings_for()
    async with gateway_for(migrated_path, settings, respond) as (gateway, ledger, database, spool):
        first = await gateway.infer(invocation(), "malformed-usage")
        assert first.state is RequestState.AMBIGUOUS
        assert (
            first.error_code
            == {
                "usage": "provider_schema_error",
                "compressed-body": "provider_transport_error",
                "model": "provider_model_mismatch",
            }[defect]
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
        if defect != "compressed-body":
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
