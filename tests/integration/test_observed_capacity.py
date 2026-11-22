"""Observed capacity covers unfinished work and stays fresh across dimensions."""

from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from _gateway_fixture import NOW, Clock, answer, gateway_for, invocation, settings_for
from anyio import Event, create_task_group, fail_after
from test_migrations import migrated_path as migrated_path
from test_provider_quota_policy import gateway_key as gateway_key

from emergence.config import ProviderKind, QuotaPolicy
from emergence.inference.gateway import GatewayError, RequestState
from emergence.resources.reporting_store import ReportingStore
from emergence.resources.schemas import QuotaObservation, ReportingIssue


@pytest.mark.anyio
@pytest.mark.parametrize("settled", [False, True])
async def test_known_credit_covers_requests_started_before_the_observation(
    migrated_path: Path, settled: bool
) -> None:
    entered, release, finished = Event(), Event(), Event()
    clock = Clock(NOW)
    calls: list[str] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        return answer() if settled else answer(None, None)

    async with gateway_for(migrated_path, settings_for(ProviderKind.OPENAI), respond, clock) as (
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

        async def first() -> None:
            await gateway.infer(invocation(), "first-credit")
            finished.set()

        denied: GatewayError | None = None
        async with create_task_group() as tasks:
            tasks.start_soon(first)
            with fail_after(2):
                await entered.wait()
            clock.at += timedelta(seconds=1)
            await store.append(
                (
                    QuotaObservation(
                        provider="p",
                        resource="credits",
                        unit="USD",
                        remaining=Decimal("0.03"),
                        observed_at=clock.at,
                        source="fixture",
                        status="observed",
                    ),
                ),
                60,
            )
            clock.at += timedelta(seconds=1)
            if settled:
                release.set()
                with fail_after(2):
                    await finished.wait()
            try:
                await gateway.infer(invocation(), "competing-credit")
            except GatewayError as error:
                denied = error
            finally:
                release.set()
        assert denied is not None and denied.code == "provider_capacity_insufficient"
        assert calls == ["/v1/chat/completions"]
        balance = await ledger.balance("p", NOW)
        assert (balance.spent, balance.reserved) == (
            (Decimal("0.008"), Decimal(0)) if settled else (Decimal(0), Decimal("0.03"))
        )
        if settled:
            clock.at += timedelta(seconds=1)
            await store.append(
                (
                    QuotaObservation(
                        provider="p",
                        resource="credits",
                        unit="USD",
                        remaining=Decimal("0.03"),
                        observed_at=clock.at,
                        source="fixture",
                        status="observed",
                    ),
                ),
                60,
            )
            assert (
                await gateway.infer(invocation(), "covered-credit")
            ).state is RequestState.COMPLETED
            assert len(calls) == 2


@pytest.mark.anyio
@pytest.mark.parametrize(
    "fresh_requests,known_tokens", [(False, True), (True, True), (True, False)]
)
async def test_strict_policy_uses_latest_failure_and_recovers_only_on_new_capacity(
    migrated_path: Path, fresh_requests: bool, known_tokens: bool
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
                    resource="reported-cost",
                    unit="USD",
                    used=Decimal(1),
                    observed_at=NOW,
                    source="reporting",
                    status="verified",
                ),
            ),
            60,
        )
        await store.append(
            (
                QuotaObservation(
                    provider="p",
                    resource="quota",
                    unit="tokens",
                    remaining=Decimal(100),
                    observed_at=NOW - timedelta(seconds=5),
                    source="fixture",
                    status="observed",
                ),
            )
            if known_tokens
            else (),
            60,
        )
        if fresh_requests:
            await store.append(
                (
                    QuotaObservation(
                        provider="p",
                        resource="requests",
                        unit="requests",
                        remaining=Decimal(5),
                        observed_at=NOW,
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
                    unit="unknown" if known_tokens else "tokens",
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
                    unit="tokens",
                    remaining=Decimal(100),
                    observed_at=clock.at,
                    source="fixture",
                    status="observed",
                ),
            ),
            60,
        )
        assert (await gateway.infer(invocation(), "recovered")).state is RequestState.COMPLETED
        assert calls == ["/v1/chat/completions"]
