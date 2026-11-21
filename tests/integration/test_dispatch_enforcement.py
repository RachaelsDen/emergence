"""Drive actual adapter dispatch through durable admission and crash boundaries."""

from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from _gateway_fixture import answer, gateway_for, invocation, settings_for
from anyio import Event, create_task_group, fail_after, sleep_forever
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from test_migrations import migrated_path as migrated_path

from emergence.inference.gateway import GatewayError, GatewayResult, RequestState
from emergence.storage.archive import SpoolError
from emergence.storage.models import (
    request_events,
    request_reservations,
    requests,
    reservation_events,
)
from emergence.storage.repositories import ArchiveReference


@pytest.fixture(autouse=True)
def gateway_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-e005-inference-key")


@pytest.mark.anyio
async def test_timeout_and_restart_preserve_identity_and_hold_without_redispatch(
    migrated_path: Path,
) -> None:
    calls: list[str] = []

    def timeout(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    settings = settings_for()
    async with gateway_for(migrated_path, settings, timeout) as (gateway, ledger, _, _):
        first = await gateway.infer(invocation(), "same-key")
        assert first.state is RequestState.AMBIGUOUS
        assert (await ledger.balance("p")).reserved == Decimal("0.03")
    async with gateway_for(migrated_path, settings, timeout) as (gateway, ledger, _, _):
        replay = await gateway.infer(invocation(), "same-key")
        assert replay.request_id == first.request_id
        assert replay.replayed
        assert replay.state is RequestState.AMBIGUOUS
        with pytest.raises(GatewayError, match="idempotency_conflict"):
            await gateway.infer(invocation(content="different"), "same-key")
        assert (await ledger.balance("p")).reserved == Decimal("0.03")
    assert calls == ["/v1/chat/completions"]


@pytest.mark.anyio
async def test_known_free_rejection_releases_funds_and_replays_failure(migrated_path: Path) -> None:
    calls: list[str] = []

    def unauthorized(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(401, json={"error": {"code": "invalid_api_key"}})

    async with gateway_for(migrated_path, settings_for(), unauthorized) as (gateway, ledger, _, _):
        first = await gateway.infer(invocation(), "unauthorized")
        replay = await gateway.infer(invocation(), "unauthorized")
        assert first.state is RequestState.FAILED
        assert replay.replayed and replay.request_id == first.request_id
        assert (await ledger.balance("p")).reserved == 0
    assert calls == ["/v1/chat/completions"]


@pytest.mark.anyio
async def test_received_answer_with_missing_usage_retains_financial_hold(
    migrated_path: Path,
) -> None:
    async with gateway_for(migrated_path, settings_for(), lambda _: answer(None, None)) as (
        gateway,
        ledger,
        _,
        _,
    ):
        result = await gateway.infer(invocation(), "missing-usage")
        assert result.state is RequestState.AMBIGUOUS
        assert result.result is not None and result.result.content == "bounded answer"
        assert (await ledger.balance("p")).reserved == Decimal("0.03")


@pytest.mark.anyio
async def test_provider_bound_violation_is_charged_and_disables_alias(migrated_path: Path) -> None:
    calls: list[str] = []

    def over_limit(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer(5, 21)

    async with gateway_for(migrated_path, settings_for(), over_limit) as (gateway, ledger, _, _):
        result = await gateway.infer(invocation(), "over-limit")
        assert result.state is RequestState.COMPLETED
        assert result.error_code == "billable_bound_violation"
        with pytest.raises(GatewayError, match="billable_bound_unverified"):
            await gateway.infer(invocation(), "next-request")
        assert (await ledger.balance("p")).spent == Decimal("0.026")
    assert calls == ["/v1/chat/completions"]


@pytest.mark.anyio
async def test_concurrent_spool_admission_rejects_without_leaking_a_hold(
    migrated_path: Path,
) -> None:
    entered, release = Event(), Event()
    results: list[GatewayResult] = []
    settings = settings_for().model_copy(update={"spool_limit_bytes": 100000})

    async def delayed(_: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return answer()

    async with gateway_for(migrated_path, settings, delayed) as (gateway, ledger, database, _):

        async def first() -> None:
            results.append(await gateway.infer(invocation(), "first"))

        with fail_after(5):
            async with create_task_group() as group:
                group.start_soon(first)
                await entered.wait()
                try:
                    with pytest.raises(SpoolError, match="spool_capacity_exhausted"):
                        await gateway.infer(invocation(), "second")
                finally:
                    release.set()
        assert results[0].state is RequestState.COMPLETED
        balance = await ledger.balance("p")
        assert balance.reserved == 0
        assert balance.spent == Decimal("0.008")
        async with database.session() as session:
            assert await session.scalar(select(func.count()).select_from(requests)) == 1
            assert await session.scalar(select(func.count()).select_from(request_reservations)) == 1


@pytest.mark.anyio
async def test_restart_settles_a_durable_response_without_calling_provider_again(
    migrated_path: Path,
) -> None:
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    settings = settings_for()
    async with gateway_for(migrated_path, settings, respond) as (gateway, _, database, _):
        async with database.write() as session:
            await session.execute(
                text(
                    "CREATE TRIGGER settlement_fault BEFORE INSERT ON usage_transactions "
                    "BEGIN SELECT RAISE(ABORT, 'simulated settlement crash'); END"
                )
            )
        with pytest.raises(IntegrityError, match="simulated settlement crash"):
            await gateway.infer(invocation(), "settlement-crash")
        async with database.write() as session:
            request_id: str = (await session.execute(select(requests.c.id))).scalar_one()
            await session.execute(text("DROP TRIGGER settlement_fault"))
        assert (await gateway.get(request_id)).state is RequestState.DISPATCHED
    async with gateway_for(migrated_path, settings, respond) as (gateway, ledger, _, _):
        replay = await gateway.infer(invocation(), "settlement-crash")
        assert replay.state is RequestState.COMPLETED and replay.replayed
        assert (await ledger.balance("p")).reserved == 0
        assert (await ledger.balance("p")).spent == Decimal("0.008")
    assert calls == ["/v1/chat/completions"]


@pytest.mark.anyio
async def test_restart_before_dispatch_releases_reserved_funds(migrated_path: Path) -> None:
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    settings = settings_for()
    async with gateway_for(migrated_path, settings, respond) as (gateway, _, database, _):
        async with database.write() as session:
            await session.execute(
                text(
                    "CREATE TRIGGER dispatch_fault BEFORE INSERT ON request_events "
                    "WHEN NEW.kind IN ('dispatched', 'failed') "
                    "BEGIN SELECT RAISE(ABORT, 'simulated pre-dispatch crash'); END"
                )
            )
        with pytest.raises(GatewayError, match="local_storage_unavailable"):
            await gateway.infer(invocation(), "before-dispatch")
        async with database.write() as session:
            await session.execute(text("DROP TRIGGER dispatch_fault"))
    async with gateway_for(migrated_path, settings, respond) as (gateway, ledger, _, _):
        replay = await gateway.infer(invocation(), "before-dispatch")
        assert replay.state is RequestState.FAILED
        assert replay.error_code == "interrupted_before_dispatch"
        assert (await ledger.balance("p")).reserved == 0
    assert calls == []


@pytest.mark.anyio
async def test_restart_after_interrupted_dispatch_retains_hold_and_never_replays(
    migrated_path: Path,
) -> None:
    entered = Event()
    calls: list[str] = []

    async def interrupted(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        entered.set()
        await sleep_forever()
        raise AssertionError("cancelled provider unexpectedly resumed")

    settings = settings_for()
    async with gateway_for(migrated_path, settings, interrupted) as (gateway, _, _, _):
        with fail_after(5):
            async with create_task_group() as group:
                group.start_soon(gateway.infer, invocation(), "after-dispatch")
                await entered.wait()
                group.cancel_scope.cancel()
    async with gateway_for(migrated_path, settings, interrupted) as (gateway, ledger, database, _):
        replay = await gateway.infer(invocation(), "after-dispatch")
        assert replay.state is RequestState.AMBIGUOUS
        assert replay.error_code == "interrupted_after_dispatch"
        assert (await ledger.balance("p")).reserved == Decimal("0.03")
        async with database.session() as session:
            assert (
                await session.scalar(
                    select(reservation_events.c.kind)
                    .order_by(reservation_events.c.sequence.desc())
                    .limit(1)
                )
                == "ambiguous"
            )
    assert calls == ["/v1/chat/completions"]


@pytest.mark.anyio
async def test_explicit_provider_quota_rejection_blocks_further_local_policy_dispatch(
    migrated_path: Path,
) -> None:
    calls: list[str] = []

    def exhausted(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(429, json={"error": {"code": "insufficient_quota"}})

    async with gateway_for(migrated_path, settings_for(), exhausted) as (gateway, ledger, _, _):
        first = await gateway.infer(invocation(), "provider-rejection")
        assert first.state is RequestState.FAILED
        assert (await ledger.balance("p")).reserved == 0
        with pytest.raises(GatewayError, match="provider_capacity_exhausted"):
            await gateway.infer(invocation(), "next-request")
    assert calls == ["/v1/chat/completions"]


@pytest.mark.anyio
@pytest.mark.parametrize("resolve", [False, True])
async def test_response_uses_verified_content_before_terminal_archive_reclamation(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch, resolve: bool
) -> None:
    secret = "synthetic-e005-inference-key"

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "configured-model",
                "choices": [{"message": {"content": "safe answer"}}],
                "usage": None if resolve else {"prompt_tokens": 5, "completion_tokens": 3},
                secret: "metadata under a credential-bearing property name",
            },
        )

    async with gateway_for(migrated_path, settings_for(), respond) as (gateway, _, database, spool):
        original_read = spool.read

        async def read_when_pinned(reference: ArchiveReference) -> bytes:
            async with database.session() as session:
                latest = await session.scalar(
                    select(request_events.c.kind).order_by(request_events.c.id.desc()).limit(1)
                )
            if latest in {"completed", "resolved"}:
                raise SpoolError("archive_unavailable")
            return await original_read(reference)

        monkeypatch.setattr(spool, "read", read_when_pinned)
        result = await gateway.infer(invocation(), "reclaim-race")
        if resolve:
            result = await gateway.resolve(
                result.request_id, None, "confirmed no charge", "operator"
            )
        assert result.state is (RequestState.RESOLVED if resolve else RequestState.COMPLETED)
        assert result.result is not None and result.result.content == "safe answer"
        assert secret not in result.model_dump_json()
