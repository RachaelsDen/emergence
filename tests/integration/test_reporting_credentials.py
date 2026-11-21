from pathlib import Path
from typing import Literal

import httpx
import pytest
from test_quota_polling import (
    config,
    cost,
    monitor_for,
    page,
    usage,
)
from test_quota_polling import (
    migrated_path as migrated_path,
)
from test_quota_polling import (
    synthetic_keys as synthetic_keys,
)

from emergence.config import ProviderKind, Settings
from emergence.resources.schemas import ReportingIssue


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("kind", "reporting_env"),
    [
        (ProviderKind.OPENAI, "TEST_REPORTING_EXPLICIT"),
        (ProviderKind.ZAI, "TEST_REPORTING_EXPLICIT"),
        (ProviderKind.MINIMAX, "TEST_REPORTING_EXPLICIT"),
        (ProviderKind.ZAI, None),
        (ProviderKind.MINIMAX, None),
    ],
)
@pytest.mark.parametrize("role", ["operator", "reader"])
@pytest.mark.parametrize("credential", ["collision", "role_name", "distinct"])
async def test_reporting_rejects_role_values_but_allows_distinct_credentials(
    migrated_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    kind: ProviderKind,
    reporting_env: str | None,
    role: Literal["operator", "reader"],
    credential: Literal["collision", "role_name", "distinct"],
) -> None:
    role_env = f"TEST_REPORTING_{role.upper()}"
    role_secret = f"synthetic-{role}-role-secret-value-000000"
    monkeypatch.setenv("TEST_REPORTING_OPERATOR", "synthetic-operator-role-secret-value-000000")
    monkeypatch.setenv("TEST_REPORTING_READER", "synthetic-reader-role-secret-value-000000")
    reporting_secret = {
        "collision": role_secret,
        "role_name": role_env,
        "distinct": "synthetic-separate-reporting-secret",
    }[credential]
    monkeypatch.setenv(reporting_env or "TEST_E004_INFERENCE", reporting_secret)
    settings = Settings(
        operator_token_env="TEST_REPORTING_OPERATOR",
        reader_token_env="TEST_REPORTING_READER",
        providers={"p": config(kind, reporting_credential_env=reporting_env)},
    )
    dispatched: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        dispatched.append(request)
        assert request.headers["Authorization"] == f"Bearer {reporting_secret}"
        if kind is ProviderKind.OPENAI:
            return page(
                request, rows=usage() if request.url.path.endswith("completions") else cost("1")
            )
        return httpx.Response(
            200, json={"data": {"limits": [{"type": "TOKENS_LIMIT", "percentage": 37}]}}
        )

    async with monitor_for(migrated_path, settings, handler) as (monitor, _):
        observations = await monitor.poll_once("p")
        persisted = await monitor.latest()

    expected_count = 2 if kind is ProviderKind.OPENAI else 1
    assert len(observations) == len(persisted) == expected_count
    if credential == "collision":
        assert dispatched == []
        for observation in (*observations, *persisted):
            assert observation.status == "unavailable"
            assert observation.issue is ReportingIssue.SHARED_CREDENTIAL
            assert observation.raw_payload_ref is None
            assert observation.used is observation.limit is observation.remaining is None
            assert observation.usage_percent is None
            assert observation.usage == observation.costs == ()
        archived = b"".join(
            path.read_bytes() for path in (migrated_path.parent / "report-spool").iterdir()
        )
        exposed = caplog.text + "".join(item.model_dump_json() for item in persisted)
        assert role_secret not in exposed
        assert role_secret.encode() not in archived
    else:
        assert len(dispatched) == expected_count
        assert all(item.issue is None for item in observations)
        assert all(item.status != "unavailable" for item in persisted)


@pytest.mark.anyio
@pytest.mark.parametrize("kind", [ProviderKind.ZAI, ProviderKind.MINIMAX])
@pytest.mark.parametrize("reporting_env", ["TEST_SUBSCRIPTION_REPORTING", None])
@pytest.mark.parametrize("admin_enabled", [True, False])
@pytest.mark.parametrize("collision", [True, False])
async def test_subscription_reporting_never_dispatches_openai_admin_values(
    migrated_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    kind: ProviderKind,
    reporting_env: str | None,
    admin_enabled: bool,
    collision: bool,
) -> None:
    admin_secret = "synthetic-protected-openai-admin-value"
    reporting_secret = admin_secret if collision else "synthetic-distinct-subscription-value"
    settings = Settings(
        providers={
            "other_admin": config(ProviderKind.OPENAI, project="proj_other"),
            "protected_admin": config(
                ProviderKind.OPENAI,
                enabled=admin_enabled,
                reporting_credential_env="TEST_PROTECTED_ADMIN",
                credential_env="TEST_PROTECTED_INFERENCE",
            ),
            "subscription": config(kind, reporting_credential_env=reporting_env),
        }
    )
    monkeypatch.setenv("TEST_PROTECTED_ADMIN", admin_secret)
    monkeypatch.setenv("TEST_PROTECTED_INFERENCE", "synthetic-distinct-openai-inference")
    monkeypatch.setenv(reporting_env or "TEST_E004_INFERENCE", reporting_secret)
    dispatched: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        dispatched.append(request)
        assert request.headers["Authorization"] == f"Bearer {reporting_secret}"
        return httpx.Response(
            200, json={"data": {"limits": [{"type": "TOKENS_LIMIT", "percentage": 37}]}}
        )

    async with monitor_for(migrated_path, settings, handler) as (monitor, _):
        observations = await monitor.poll_once("subscription")
        persisted = await monitor.latest()

    assert len(observations) == len(persisted) == 1
    if collision:
        assert dispatched == []
        for observation in (*observations, *persisted):
            assert observation.status == "unavailable"
            assert observation.issue is ReportingIssue.SHARED_CREDENTIAL
            assert observation.raw_payload_ref is None
            assert observation.used is observation.limit is observation.remaining is None
            assert observation.usage_percent is None
            assert observation.usage == observation.costs == ()
        archived = b"".join(
            path.read_bytes() for path in (migrated_path.parent / "report-spool").iterdir()
        )
        exposed = caplog.text + "".join(item.model_dump_json() for item in persisted)
        assert admin_secret not in exposed
        assert admin_secret.encode() not in archived
    else:
        assert len(dispatched) == 1
        assert all(item.issue is None for item in (*observations, *persisted))
        assert all(item.status == "experimental" for item in (*observations, *persisted))
