"""Malformed wire usage must not settle or redispatch a bounded request."""

from decimal import Decimal
from pathlib import Path
from typing import Literal

import httpx
import pytest
from _gateway_fixture import NOW, gateway_for, invocation, settings_for
from pydantic import JsonValue
from sqlalchemy import select
from test_migrations import migrated_path as migrated_path

from emergence.api.app import create_app
from emergence.inference.errors import GatewayError
from emergence.inference.gateway import ArchivedOutcome, RequestState
from emergence.storage.archive import RecordKey
from emergence.storage.models import archive_references
from emergence.storage.repositories import ArchiveReference


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("defect", "response_model", "wire_usage"),
    [
        ("usage", "configured-model", {"prompt_tokens": False, "completion_tokens": False}),
        ("compressed-body", "configured-model", {"prompt_tokens": 5, "completion_tokens": 3}),
        ("duplicate-usage", "configured-model", {"prompt_tokens": 5, "completion_tokens": 3}),
        ("model", "other-model", {"prompt_tokens": 5, "completion_tokens": 3}),
        ("model", "m", {"prompt_tokens": 5, "completion_tokens": 3}),
        ("model", "Configured-model", {"prompt_tokens": 5, "completion_tokens": 3}),
        ("model", "configured-model ", {"prompt_tokens": 5, "completion_tokens": 3}),
        ("usage", "configured-model", {"prompt_tokens": 2**63, "completion_tokens": 3}),
        ("usage", "configured-model", {"prompt_tokens": 5, "completion_tokens": 2**63}),
        (
            "usage",
            "configured-model",
            {
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 2**63},
            },
        ),
        (
            "usage",
            "configured-model",
            {
                "prompt_tokens": 5,
                "completion_tokens_details": {"reasoning_tokens": 2**63},
            },
        ),
        (
            "usage",
            "configured-model",
            {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 2**63},
        ),
    ],
)
async def test_invalid_wire_response_preserves_hold_and_replays_without_dispatch(
    migrated_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: Literal["usage", "compressed-body", "duplicate-usage", "model"],
    response_model: str,
    wire_usage: dict[str, JsonValue],
) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-b1-inference")
    payload = {
        "model": response_model,
        "choices": [{"message": {"content": "synthetic response"}}],
        "usage": wire_usage,
        "credential_echo": "synthetic-b1-inference",
    }
    duplicate_usage = (
        '{"model":"configured-model","choices":[{"message":{"content":"answer"}}],'
        '"usage":{"prompt_tokens":1,"prompt_tokens":5,"completion_tokens":3},'
        '"credential_echo":"synthetic-b1-inference"}'
    )
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if defect == "duplicate-usage":
            return httpx.Response(200, content=duplicate_usage.encode())
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
                "compressed-body": "provider_response_encoding_unsupported",
                "duplicate-usage": "provider_schema_error",
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
        if defect == "duplicate-usage":
            assert outcome.failure.raw_response == duplicate_usage.replace(
                "synthetic-b1-inference", "[REDACTED]"
            )
            assert b"synthetic-b1-inference" not in record
        elif defect != "compressed-body":
            assert outcome.failure.raw_response == {**payload, "credential_echo": "[REDACTED]"}
            assert b"synthetic-b1-inference" not in record
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

    app = create_app(
        settings.model_copy(
            update={
                "database_path": migrated_path,
                "spool_path": migrated_path.parent / "gateway-spool",
            }
        )
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/health/ready")).json() == {"status": "ready"}


@pytest.mark.anyio
async def test_storage_maximum_still_enforces_billable_bound_policy(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-b1-inference")
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "model": "configured-model",
                "choices": [{"message": {"content": "unreasonable usage"}}],
                "usage": {
                    "prompt_tokens": 2**63 - 1,
                    "completion_tokens": 0,
                    "prompt_tokens_details": {"cached_tokens": 2**63 - 1},
                },
            },
        )

    async with gateway_for(migrated_path, settings_for(), respond) as (gateway, _, _, _):
        result = await gateway.infer(invocation(), "storage-maximum")
        assert result.error_code == "billable_bound_violation"
        with pytest.raises(GatewayError, match="billable_bound_unverified"):
            await gateway.infer(invocation(), "another-request")
    assert calls == ["/v1/chat/completions"]
