"""Public retries reconcile durable output after a rolled-back finalization."""

from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from _gateway_fixture import NOW, answer, gateway_for, invocation, settings_for
from anyio import create_task_group, fail_after
from pydantic import JsonValue
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from test_migrations import migrated_path as migrated_path

from emergence.api.app import create_app
from emergence.inference.gateway import GatewayError, GatewayResult, RequestState
from emergence.storage.archive import RecordKey
from emergence.storage.capacity import live_claims
from emergence.storage.models import (
    archive_references,
    quota_observations,
    request_events,
    requests,
    reservation_events,
    usage_transactions,
)
from emergence.storage.repositories import ArchiveReference


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("response", "state", "spent", "held", "observations"),
    [
        (answer(), RequestState.COMPLETED, "0.008", "0", 0),
        (answer(None, None), RequestState.AMBIGUOUS, "0", "0.03", 0),
        (
            httpx.Response(401, json={"error": {"code": "invalid_api_key"}}),
            RequestState.FAILED,
            "0",
            "0",
            0,
        ),
        (
            httpx.Response(429, json={"error": {"code": "insufficient_quota"}}),
            RequestState.FAILED,
            "0",
            "0",
            1,
        ),
        (
            httpx.Response(200, json={"model": "configured-model", "usage": False}),
            RequestState.AMBIGUOUS,
            "0",
            "0.03",
            0,
        ),
    ],
)
async def test_replays_reconcile_archived_output_once_after_finish_rollback(
    migrated_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
    state: RequestState,
    spent: str,
    held: str,
    observations: int,
) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-replay-key")
    calls: list[str] = []
    settings = settings_for()
    settings = settings.model_copy(
        update={"providers": {"p": settings.providers["p"].model_copy(update={"concurrency": 1})}}
    )

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return response

    async with gateway_for(migrated_path, settings, respond) as (gateway, ledger, db, _):
        async with db.write() as session:
            await session.execute(
                text(
                    "CREATE TRIGGER finish_fault BEFORE INSERT ON request_events "
                    "WHEN NEW.kind IN ('completed', 'failed', 'ambiguous') "
                    "BEGIN SELECT RAISE(ABORT, 'finish transaction rolled back'); END"
                )
            )
        with pytest.raises(IntegrityError, match="finish transaction rolled back"):
            await gateway.infer(invocation(), "archived")
        async with db.write() as session:
            request_id: str = (await session.execute(select(requests.c.id))).scalar_one()
            assert await session.scalar(select(func.count()).select_from(archive_references)) == 2
            assert await session.scalar(select(func.count()).select_from(usage_transactions)) == 0
            assert len(await live_claims(session)) == 1
            await session.execute(text("DROP TRIGGER finish_fault"))
        assert (await gateway.get(request_id)).state is RequestState.DISPATCHED

        replays: list[GatewayResult] = []

        async def replay() -> None:
            replays.append(await gateway.infer(invocation(), "archived"))

        with fail_after(10):
            async with create_task_group() as group:
                for _ in range(4):
                    group.start_soon(replay)
        assert len(replays) == 4
        assert all(r.replayed and r.request_id == request_id and r.state is state for r in replays)
        assert all(r.result == replays[0].result for r in replays)
        assert all(r.error_code == replays[0].error_code for r in replays)
        balance = await ledger.balance("p", NOW)
        assert (balance.spent, balance.reserved) == (Decimal(spent), Decimal(held))
        async with db.session() as session:
            events = tuple(await session.scalars(select(request_events.c.kind)))
            reservations = tuple(await session.scalars(select(reservation_events.c.kind)))
            assert events == ("admitted", "dispatched", state.value)
            assert await session.scalar(select(func.count()).select_from(quota_observations)) == (
                observations
            )
            assert await session.scalar(select(func.count()).select_from(usage_transactions)) == (
                1 if state is RequestState.COMPLETED else 0
            )
            assert len(await live_claims(session)) == (1 if held != "0" else 0)
        if state is RequestState.AMBIGUOUS:
            with pytest.raises(GatewayError, match="provider_concurrency_exhausted"):
                await gateway.infer(invocation(), "blocked")
            if replays[0].result is not None:
                await gateway.resolve(request_id, None, "confirmed no charge", "operator")
                state = RequestState.RESOLVED
                resolved = await gateway.infer(invocation(), "archived")
                assert resolved.state is RequestState.RESOLVED and resolved.replayed

    async with gateway_for(migrated_path, settings, respond) as (gateway, _, db, _):
        replayed = await gateway.infer(invocation(), "archived")
        assert replayed.state is state and replayed.replayed
        async with db.session() as session:
            assert tuple(await session.scalars(select(request_events.c.kind))) == (
                events + ("resolved",) if state is RequestState.RESOLVED else events
            )
            later_reservations = tuple(await session.scalars(select(reservation_events.c.kind)))
            assert later_reservations[: len(reservations)] == reservations
            assert len(later_reservations) == len(reservations) + (
                1 if state is RequestState.RESOLVED else 0
            )
            assert await session.scalar(select(func.count()).select_from(quota_observations)) == (
                observations
            )
            assert await session.scalar(select(func.count()).select_from(usage_transactions)) == (
                1 if state is RequestState.COMPLETED else 0
            )
    assert calls == ["/v1/chat/completions"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "defect",
    [
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "overflow_and_bad_content",
        "overflow_and_bad_overlap",
        "boolean",
        "negative",
    ],
)
async def test_startup_recovers_only_legacy_normalized_integer_overflow(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-legacy-key")
    settings = settings_for()
    calls: list[str] = []
    references: list[ArchiveReference] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    async with gateway_for(migrated_path, settings, respond) as (gateway, _, _, spool):
        original_publish = spool.publish

        async def publish_old(
            key: RecordKey, payload: JsonValue, *, claim_request_id: str | None = None
        ) -> ArchiveReference:
            if key.kind == "output":
                assert isinstance(payload, dict) and isinstance(payload["result"], dict)
                result = payload["result"]
                result["usage"] = (
                    {defect: 2**63}
                    if defect.endswith("_tokens")
                    else {"input_tokens": 2**63, "cached_input_tokens": None}
                )
                if defect == "overflow_and_bad_content":
                    result["content"] = []
                if defect == "overflow_and_bad_overlap":
                    result["usage"] = {"input_tokens": 2**63, "cached_input_tokens": 2**63 + 1}
                if defect in {"boolean", "negative"}:
                    result["usage"] = {"input_tokens": True if defect == "boolean" else -1}
                result["raw_usage"] = {"legacy_count": 2**63}
                result["raw_response"] = {"historic": 2**63, "body": "legacy evidence"}
            reference = await original_publish(key, payload, claim_request_id=claim_request_id)
            if key.kind == "output":
                references.append(reference)
                raise RuntimeError("crashed after historic output publication")
            return reference

        monkeypatch.setattr(spool, "publish", publish_old)
        with pytest.raises(RuntimeError, match="crashed after historic"):
            await gateway.infer(invocation(), "legacy")
        original_bytes = await spool.read(references[0])
    valid_legacy = defect.endswith("_tokens")
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
        assert (await client.get("/health/ready")).status_code == (200 if valid_legacy else 503)
    if valid_legacy:
        async with gateway_for(migrated_path, settings, respond) as (gateway, ledger, db, spool):
            replay = await gateway.infer(invocation(), "legacy")
            assert replay.state is RequestState.AMBIGUOUS and replay.replayed
            assert replay.error_code == "provider_schema_error" and replay.result is None
            balance = await ledger.balance("p", NOW)
            assert (balance.reserved, balance.spent) == (Decimal("0.03"), 0)
            assert await spool.read(references[0]) == original_bytes
            async with db.session() as session:
                assert tuple(await session.scalars(select(request_events.c.kind))) == (
                    "admitted",
                    "dispatched",
                    "ambiguous",
                )
                assert (
                    await session.scalar(select(func.count()).select_from(usage_transactions)) == 0
                )
                assert (
                    await session.scalar(
                        select(request_events.c.archive_record_id).where(
                            request_events.c.kind == "ambiguous"
                        )
                    )
                    == references[0].record_id
                )
    assert calls == ["/v1/chat/completions"]
