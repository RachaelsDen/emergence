"""Quota uncertainty policies operate on real bounded dispatch, not capability flags."""

from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from _gateway_fixture import NOW, Clock, answer, gateway_for, invocation, settings_for
from anyio import Event, create_task_group, fail_after
from sqlalchemy import func, select
from test_migrations import migrated_path as migrated_path

from emergence.config import ProviderKind, QuotaPolicy
from emergence.inference.gateway import GatewayError, RequestState
from emergence.resources.reporting_store import ReportingStore
from emergence.resources.schemas import QuotaObservation, ReportingIssue
from emergence.storage.models import requests


@pytest.fixture(autouse=True)
def gateway_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-e005-inference-key")


@pytest.mark.anyio
@pytest.mark.parametrize("kind", list(ProviderKind))
async def test_local_policy_allows_bounded_dispatch_without_absolute_quota(
    migrated_path: Path, kind: ProviderKind
) -> None:
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    async with gateway_for(migrated_path, settings_for(kind), respond) as (gateway, _, _, _):
        result = await gateway.infer(invocation(), "bounded")
        assert result.state is RequestState.COMPLETED
        assert result.result is not None and result.result.content == "bounded answer"
    assert calls == ["/v1/chat/completions"]


@pytest.mark.anyio
@pytest.mark.parametrize("kind", list(ProviderKind))
async def test_strict_policy_rejects_missing_quota_before_provider_dispatch(
    migrated_path: Path, kind: ProviderKind
) -> None:
    calls: list[str] = []
    settings = settings_for(kind)
    settings = settings.model_copy(
        update={
            "providers": {
                "p": settings.providers["p"].model_copy(
                    update={"quota_policy": QuotaPolicy.REQUIRE_FRESH}
                )
            }
        }
    )

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    async with gateway_for(migrated_path, settings, respond) as (gateway, _, database, _):
        with pytest.raises(GatewayError, match="provider_capacity_unknown"):
            await gateway.infer(invocation(), "strict")
        async with database.session() as session:
            assert await session.scalar(select(func.count()).select_from(requests)) == 0
    assert calls == []


@pytest.mark.anyio
async def test_known_credit_accounts_for_a_call_still_inside_the_provider(
    migrated_path: Path,
) -> None:
    entered, release = Event(), Event()
    calls: list[str] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        return answer(None, None)

    async with gateway_for(migrated_path, settings_for(ProviderKind.OPENAI), respond) as (
        gateway,
        ledger,
        database,
        spool,
    ):
        await ReportingStore(database, spool).append(
            (
                QuotaObservation(
                    provider="p",
                    resource="credits",
                    unit="USD",
                    remaining=Decimal("0.03"),
                    observed_at=NOW - timedelta(seconds=1),
                    source="fixture",
                    status="observed",
                ),
            ),
            60,
        )
        denied: GatewayError | None = None
        async with create_task_group() as tasks:
            tasks.start_soon(gateway.infer, invocation(), "first-credit")
            with fail_after(2):
                await entered.wait()
            try:
                await gateway.infer(invocation(), "competing-credit")
            except GatewayError as error:
                denied = error
            finally:
                release.set()
        assert denied is not None and denied.code == "provider_capacity_insufficient"
        assert calls == ["/v1/chat/completions"]
        balance = await ledger.balance("p", NOW)
        assert (balance.spent, balance.reserved) == (Decimal(0), Decimal("0.03"))


@pytest.mark.anyio
async def test_strict_policy_uses_latest_failure_and_recovers_only_on_new_capacity(
    migrated_path: Path,
) -> None:
    clock = Clock(NOW)
    settings = settings_for()
    settings = settings.model_copy(
        update={
            "providers": {
                "p": settings.providers["p"].model_copy(
                    update={"quota_policy": QuotaPolicy.REQUIRE_FRESH}
                )
            }
        }
    )
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    async with gateway_for(migrated_path, settings, respond, clock) as (
        gateway,
        ledger,
        database,
        spool,
    ):
        store = ReportingStore(database, spool)
        await store.append(
            (
                QuotaObservation(
                    provider="p",
                    resource="quota",
                    unit="requests",
                    remaining=Decimal(5),
                    observed_at=NOW - timedelta(seconds=5),
                    source="fixture",
                    status="observed",
                ),
            ),
            60,
        )
        await store.append(
            (
                QuotaObservation(
                    provider="p",
                    resource="quota",
                    observed_at=NOW,
                    source="fixture",
                    status="unavailable",
                    issue=ReportingIssue.TIMEOUT,
                ),
            ),
            60,
        )
        denied: GatewayError | None = None
        try:
            await gateway.infer(invocation(), "outage")
        except GatewayError as error:
            denied = error
        assert denied is not None and denied.code == "provider_capacity_unknown"
        assert calls == []
        assert (await ledger.balance("p", NOW)).reserved == 0
        clock.at += timedelta(seconds=1)
        await store.append(
            (
                QuotaObservation(
                    provider="p",
                    resource="quota",
                    unit="requests",
                    remaining=Decimal(5),
                    observed_at=clock.at,
                    source="fixture",
                    status="observed",
                ),
            ),
            60,
        )
        assert (await gateway.infer(invocation(), "recovered")).state is RequestState.COMPLETED
        assert calls == ["/v1/chat/completions"]


@pytest.mark.anyio
@pytest.mark.parametrize("kind", list(ProviderKind))
@pytest.mark.parametrize("remaining", [Decimal(0), Decimal(-1)])
async def test_known_exhaustion_blocks_even_when_capacity_observation_is_stale(
    migrated_path: Path, kind: ProviderKind, remaining: Decimal
) -> None:
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    async with gateway_for(migrated_path, settings_for(kind), respond) as (
        gateway,
        _,
        database,
        spool,
    ):
        await ReportingStore(database, spool).append(
            (
                QuotaObservation(
                    provider="p",
                    resource="inference",
                    unit="requests",
                    remaining=remaining,
                    observed_at=NOW - timedelta(days=1),
                    source="inference",
                    status="observed",
                    confidence="synthetic known rejection",
                ),
            ),
            10,
        )
        with pytest.raises(GatewayError, match="provider_capacity_exhausted"):
            await gateway.infer(invocation(), "exhausted")
        async with database.session() as session:
            assert await session.scalar(select(func.count()).select_from(requests)) == 0
    assert calls == []
