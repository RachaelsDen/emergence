"""Quota uncertainty policies operate on real bounded dispatch, not capability flags."""

from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from _gateway_fixture import NOW, answer, gateway_for, invocation, settings_for
from sqlalchemy import func, select
from test_migrations import migrated_path as migrated_path

from emergence.config import ProviderKind, QuotaPolicy
from emergence.inference.gateway import GatewayError, RequestState
from emergence.resources.reporting_store import ReportingStore
from emergence.resources.schemas import QuotaObservation
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
