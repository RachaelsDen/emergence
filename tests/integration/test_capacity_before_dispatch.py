"""Capacity snapshots reserve admitted work but exclude known unsent failures."""

from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from _gateway_fixture import NOW, answer, gateway_for, invocation, settings_for
from anyio import Event, create_task_group, fail_after
from pydantic import JsonValue
from sqlalchemy import select
from test_migrations import migrated_path as migrated_path
from test_provider_quota_policy import gateway_key as gateway_key

from emergence.inference.gateway import GatewayError, GatewayResult, RequestState
from emergence.resources.reporting_store import ReportingStore
from emergence.resources.schemas import QuotaObservation
from emergence.storage.archive import RecordKey
from emergence.storage.capacity import live_claims
from emergence.storage.models import request_events
from emergence.storage.repositories import ArchiveReference


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("unit", "remaining"),
    [("requests", Decimal(1)), ("tokens", Decimal(30)), ("USD", Decimal("0.03"))],
)
async def test_failed_input_publication_does_not_consume_observed_capacity(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch, unit: str, remaining: Decimal
) -> None:
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    async with gateway_for(migrated_path, settings_for(), respond) as (
        gateway,
        ledger,
        database,
        spool,
    ):
        await ReportingStore(database, spool).append(
            (
                QuotaObservation(
                    provider="p",
                    resource="capacity",
                    unit=unit,
                    remaining=remaining,
                    observed_at=NOW - timedelta(seconds=1),
                    source="fixture",
                    status="observed",
                ),
            ),
            60,
        )

        async def unavailable(
            key: RecordKey, payload: JsonValue, *, claim_request_id: str | None = None
        ) -> ArchiveReference:
            raise OSError("synthetic input storage failure")

        with monkeypatch.context() as patch:
            patch.setattr(spool, "publish", unavailable)
            with pytest.raises(GatewayError, match="input_archive_failed") as captured:
                await gateway.infer(invocation(), "unsent")
        request_id = captured.value.request_id
        assert request_id is not None
        assert (await gateway.get(request_id)).state is RequestState.FAILED
        assert (await ledger.balance("p", NOW)).reserved == 0
        assert calls == []

        result = await gateway.infer(invocation(), "sent")
        assert result.state is RequestState.COMPLETED
        assert calls == ["/v1/chat/completions"]
        balance = await ledger.balance("p", NOW)
        assert (balance.spent, balance.reserved) == (Decimal("0.008"), Decimal(0))


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("unit", "remaining"),
    [("requests", Decimal(1)), ("tokens", Decimal(30)), ("USD", Decimal("0.03"))],
)
async def test_pending_input_publication_still_reserves_observed_capacity(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch, unit: str, remaining: Decimal
) -> None:
    entered, release = Event(), Event()
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    async with gateway_for(migrated_path, settings_for(), respond) as (
        gateway,
        _,
        database,
        spool,
    ):
        await ReportingStore(database, spool).append(
            (
                QuotaObservation(
                    provider="p",
                    resource="capacity",
                    unit=unit,
                    remaining=remaining,
                    observed_at=NOW - timedelta(seconds=1),
                    source="fixture",
                    status="observed",
                ),
            ),
            60,
        )
        publish = spool.publish

        async def paused(
            key: RecordKey, payload: JsonValue, *, claim_request_id: str | None = None
        ) -> ArchiveReference:
            if not entered.is_set():
                entered.set()
                await release.wait()
            return await publish(key, payload, claim_request_id=claim_request_id)

        monkeypatch.setattr(spool, "publish", paused)

        async def first() -> None:
            assert (await gateway.infer(invocation(), "pending")).state is RequestState.COMPLETED

        denied: GatewayError | None = None
        async with create_task_group() as tasks:
            tasks.start_soon(first)
            with fail_after(2):
                await entered.wait()
            try:
                await gateway.infer(invocation(), "competing")
            except GatewayError as error:
                denied = error
            finally:
                release.set()
        assert denied is not None and denied.code == "provider_capacity_insufficient"
        assert calls == ["/v1/chat/completions"]


@pytest.mark.anyio
async def test_retry_without_output_preserves_genuine_inflight_dispatch(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-replay-key")
    entered, release = Event(), Event()
    calls: list[str] = []
    results: list[GatewayResult] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        entered.set()
        await release.wait()
        return answer()

    async with gateway_for(migrated_path, settings_for(), respond) as (gateway, _, db, _):

        async def first() -> None:
            results.append(await gateway.infer(invocation(), "inflight"))

        with fail_after(10):
            async with create_task_group() as group:
                group.start_soon(first)
                await entered.wait()
                try:
                    replay = await gateway.infer(invocation(), "inflight")
                    assert replay.state is RequestState.DISPATCHED and replay.result is None
                    async with db.session() as session:
                        assert len(await live_claims(session)) == 1
                        assert tuple(await session.scalars(select(request_events.c.kind))) == (
                            "admitted",
                            "dispatched",
                        )
                finally:
                    release.set()
        assert results[0].state is RequestState.COMPLETED
    assert calls == ["/v1/chat/completions"]
