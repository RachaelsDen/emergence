"""Reporting-only credentials cannot reach inference or create a financial hold."""

from pathlib import Path

import httpx
import pytest
from _gateway_fixture import answer, gateway_for, invocation, settings_for
from sqlalchemy import func, select
from test_migrations import migrated_path as migrated_path

from emergence.config import ProviderConfig, ProviderKind, QuotaPolicy
from emergence.providers.base import ProviderFailure
from emergence.storage.models import request_reservations, requests


@pytest.mark.anyio
@pytest.mark.parametrize("enforce_budgets", [True, False], ids=["enforced", "shadow"])
@pytest.mark.parametrize("kind", [ProviderKind.OPENAI, ProviderKind.ZAI])
async def test_admin_value_collision_rejects_before_hold_or_transport(
    migrated_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enforce_budgets: bool,
    kind: ProviderKind,
) -> None:
    shared = "synthetic-b2-only"
    monkeypatch.setenv("TEST_E005_INFERENCE", shared)
    monkeypatch.setenv("TEST_B2_ADMIN", shared)
    settings = settings_for(kind)
    if kind is ProviderKind.OPENAI:
        providers = {
            "p": settings.providers["p"].model_copy(
                update={"reporting_credential_env": "TEST_B2_ADMIN"}
            )
        }
    else:
        providers = {
            **settings.providers,
            "reporting-provider": ProviderConfig(
                enabled=False,
                kind=ProviderKind.OPENAI,
                base_url="https://reporting-fixture.invalid/v1",
                quota_policy=QuotaPolicy.LOCAL_LIMITS_ONLY,
                reporting_credential_env="TEST_B2_ADMIN",
            ),
        }
    settings = settings.model_copy(
        update={"providers": providers, "enforce_budgets": enforce_budgets}
    )
    calls: list[bool] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("Authorization") == f"Bearer {shared}")
        return answer()

    async with gateway_for(migrated_path, settings, respond) as (gateway, _, database, _):
        with pytest.raises(ProviderFailure, match="shared_inference_credential") as failure:
            await gateway.infer(invocation(), "admin-collision")
        assert not failure.value.ambiguous
        assert shared not in str(failure.value)
        assert calls == []
        async with database.session() as session:
            assert await session.scalar(select(func.count()).select_from(requests)) == 0
            assert await session.scalar(select(func.count()).select_from(request_reservations)) == 0


@pytest.mark.anyio
@pytest.mark.parametrize("operator", [False, True], ids=["reader", "operator"])
@pytest.mark.parametrize("enforce_budgets", [True, False], ids=["enforced", "shadow"])
async def test_api_role_value_cannot_reach_provider_or_create_a_hold(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch, operator: bool, enforce_budgets: bool
) -> None:
    shared = "synthetic-api-role-only-0123456789abcdef"
    settings = settings_for().model_copy(update={"enforce_budgets": enforce_budgets})
    monkeypatch.setenv("TEST_E005_INFERENCE", shared)
    monkeypatch.setenv(
        settings.operator_token_env if operator else settings.reader_token_env, shared
    )
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    async with gateway_for(migrated_path, settings, respond) as (gateway, _, database, _):
        with pytest.raises(ProviderFailure, match="shared_inference_credential") as failure:
            await gateway.infer(invocation(), "role-collision")
        assert shared not in str(failure.value)
        assert calls == []
        async with database.session() as session:
            assert await session.scalar(select(func.count()).select_from(requests)) == 0
            assert await session.scalar(select(func.count()).select_from(request_reservations)) == 0
