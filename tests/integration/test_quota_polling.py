from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Final

import httpx
import pytest
from anyio import Event, create_task_group, fail_after, sleep, sleep_forever
from sqlalchemy import select, text
from test_migrations import migrated_path as migrated_path

from emergence.config import ModelConfig, ProviderConfig, ProviderKind, QuotaPolicy, Settings
from emergence.resources import QuotaMonitor
from emergence.resources.schemas import ReportingIssue
from emergence.storage.archive import Spool
from emergence.storage.database import Database
from emergence.storage.models import quota_observations, usage_transactions
from emergence.storage.repositories import ensure_catalog
from emergence.telemetry.logging import Redactor

_NOW: Final = datetime(2026, 11, 20, 12, tzinfo=UTC)
_ADMIN: Final = "sk-synthetic-admin-report"
_INFERENCE: Final = "sk-synthetic-inference-only"
_FIXTURES: Final = Path(__file__).resolve().parents[1] / "fixtures" / "reporting"
type Handler = (
    Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
)


def config(kind: ProviderKind, **updates: str | int | float | datetime | None) -> ProviderConfig:
    baseline = ProviderConfig(
        kind=kind,
        base_url="https://inference.invalid/v1",
        quota_url="https://fixture.invalid/quota",
        quota_policy=QuotaPolicy.LOCAL_LIMITS_ONLY,
        models={"alias": ModelConfig(id="upstream")},
        credential_env="TEST_E004_INFERENCE",
        reporting_credential_env="TEST_E004_ADMIN",
        reporting_base_url="https://fixture.invalid/v1",
        organization="org_fixture",
        project="proj_fixture",
        stale_after_seconds=10,
        poll_interval_seconds=1,
        timeout_seconds=1,
    )
    return ProviderConfig.model_validate({**baseline.model_dump(), **updates})


@pytest.fixture(autouse=True)
def synthetic_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_E004_ADMIN", _ADMIN)
    monkeypatch.setenv("TEST_E004_INFERENCE", _INFERENCE)


@asynccontextmanager
async def monitor_for(
    path: Path, settings: Settings, handler: Handler, clock: Callable[[], datetime] = lambda: _NOW
) -> AsyncGenerator[tuple[QuotaMonitor, Database]]:
    async with (
        Database(path) as db,
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            auth=("unused", _INFERENCE),
        ) as client,
    ):
        await ensure_catalog(db, settings)
        spool = Spool(path.parent / "report-spool", db, Redactor(settings.secrets()))
        yield QuotaMonitor(settings, db, spool, client, clock), db


def page(
    request: httpx.Request,
    *,
    rows: str,
    cursor: str | None = None,
    offset: int = 0,
    more: bool | None = None,
) -> httpx.Response:
    end = int(request.url.params["end_time"]) - offset * 86400
    next_page = f'"{cursor}"' if cursor is not None else "null"
    has_more = "true" if (more if more is not None else cursor is not None) else "false"
    return httpx.Response(
        200,
        content=(
            '{"object":"page","data":[{"object":"bucket",'
            f'"start_time":{end - 86400},"end_time":{end},"results":[{rows}]'
            "}],"
            f'"has_more":{has_more},"next_page":{next_page}}}'
        ),
    )


def usage() -> str:
    return (
        '{"object":"organization.usage.completions.result","project_id":"proj_fixture",'
        '"model":"fixture-model","input_tokens":20,"output_tokens":7,"num_model_requests":3,'
        '"input_cached_tokens":null,"unknown_future_dimension":6}'
    )


def cost(amount: str, project: str = "proj_fixture") -> str:
    return (
        f'{{"object":"organization.costs.result","project_id":"{project}",'
        f'"line_item":"tokens","amount":{{"value":{amount},"currency":"usd"}}}}'
    )


@pytest.mark.anyio
async def test_reporting_depth_limit_records_unavailable_and_allows_later_poll(
    migrated_path: Path,
) -> None:
    settings = Settings(providers={"p": config(ProviderKind.MINIMAX)})
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        depth = 64 if calls == 1 else 63  # The root object is also a container.
        body = b'{"unknown":' + b"[" * depth + b"null" + b"]" * depth + b"}"
        return httpx.Response(200, content=body)

    async with monitor_for(migrated_path, settings, handler) as (monitor, db):
        (rejected,) = await monitor.poll_once("p")
        assert (rejected.status, rejected.issue) == ("unavailable", ReportingIssue.INVALID_JSON)
        (accepted,) = await monitor.poll_once("p")
        assert accepted.status == "experimental"
        async with db.session() as session:
            rows = (
                await session.execute(
                    select(quota_observations.c.status).order_by(quota_observations.c.id)
                )
            ).scalars()
            assert tuple(rows) == ("unavailable", "experimental")


@pytest.mark.anyio
async def test_complete_openai_reports_persist_exact_idempotent_deltas(migrated_path: Path) -> None:
    settings = Settings(providers={"o": config(ProviderKind.OPENAI)})
    val = "7.00000000000000000000000000000000000001"
    unchanged = val
    raised = "8.00000000000000000000000000000000000002"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "fixture.invalid"
        assert request.headers["Authorization"] == f"Bearer {_ADMIN}"
        assert request.headers["OpenAI-Organization"] == "org_fixture"
        assert request.url.params.get_list("project_ids") == ["proj_fixture"]
        second = request.url.params.get("page") == "two"
        if request.url.path == "/v1/organization/usage/completions":
            assert request.url.params.get_list("group_by") == ["project_id", "model"]
            return page(request, rows=usage(), cursor=None if second else "two", offset=int(second))
        assert request.url.path == "/v1/organization/costs"
        assert request.url.params.get_list("group_by") == ["project_id", "line_item"]
        return page(
            request,
            rows=cost("-0.00000000000000000000000000000000000003" if second else val),
            cursor=None if second else "two",
            offset=int(second),
        )

    async with monitor_for(migrated_path, settings, handler) as (first, db):
        for val in (unchanged, unchanged, raised, unchanged):
            result = {item.resource: item for item in await first.poll_once("o")}
            assert len(result["usage"].usage) == len(result["costs"].costs) == 2
            assert result["costs"].costs[0].amount.as_tuple().digits[-1] == int(val[-1])
        async with db.session() as session:
            ledger = (
                await session.execute(
                    text(
                        "SELECT measured_cost, request_id, model_id, budget_period_id "
                        "FROM usage_transactions ORDER BY rowid"
                    )
                )
            ).all()
            assert ledger == [
                (unchanged, None, None, None),
                ("-0.00000000000000000000000000000000000003", None, None, None),
                ("1.00000000000000000000000000000000000001", None, None, None),
                ("-1.00000000000000000000000000000000000001", None, None, None),
            ]
            count = await session.scalar(select(text("count(*)")).select_from(quota_observations))
            assert count == 8

    later = _NOW + timedelta(seconds=10)
    async with monitor_for(migrated_path, settings, handler, lambda: later) as (reopened, _):
        recovered = {item.resource: item for item in await reopened.latest()}
    assert recovered.keys() == {"usage", "costs"}
    assert all(o.is_stale and o.raw_payload_ref for o in recovered.values())
    assert recovered["usage"].usage[0].model == "fixture-model"
    assert str(recovered["costs"].costs[0].amount) == unchanged


@pytest.mark.anyio
async def test_subscription_quota_can_use_its_coding_plan_credential(migrated_path: Path) -> None:
    settings = Settings(providers={"z": config(ProviderKind.ZAI, reporting_credential_env=None)})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Authorization") != f"Bearer {_INFERENCE}":
            return httpx.Response(401)
        return httpx.Response(
            200, json={"data": {"limits": [{"type": "TOKENS_LIMIT", "percentage": 37}]}}
        )

    async with monitor_for(migrated_path, settings, handler) as (monitor, _):
        observations = await monitor.poll_once("z")
    assert len(observations) == 1
    assert observations[0].status == "experimental"
    assert observations[0].usage_percent == Decimal(37)
    assert observations[0].remaining is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("fault", "issue"),
    [
        ("missing", ReportingIssue.INVALID_PAGINATION),
        ("cycle", ReportingIssue.INVALID_PAGINATION),
        ("second_401", ReportingIssue.UNAUTHORIZED),
        ("project", ReportingIssue.SCOPE_MISMATCH),
        ("html", ReportingIssue.INVALID_JSON),
    ],
)
async def test_broken_admin_cost_snapshot_never_posts_partial_money(
    migrated_path: Path, fault: str, issue: ReportingIssue
) -> None:
    settings = Settings(providers={"o": config(ProviderKind.OPENAI)})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("completions"):
            return page(request, rows=usage())
        if request.url.params.get("page"):
            if fault == "second_401":
                return httpx.Response(401)
            return page(request, rows=cost("1"), cursor="two", offset=1)
        if fault == "missing":
            return page(request, rows=cost("7"), more=True)
        if fault == "project":
            return page(request, rows=cost("7", "proj_wrong"))
        if fault == "html":
            return httpx.Response(200, text="<html>")
        return page(request, rows=cost("7"), cursor="two")

    async with monitor_for(migrated_path, settings, handler) as (monitor, db):
        result = {item.resource: item for item in await monitor.poll_once("o")}
        assert result["usage"].usage[0].num_model_requests == 3
        assert result["costs"].issue is issue and result["costs"].costs == ()
        async with db.session() as session:
            assert (await session.execute(select(usage_transactions))).first() is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("overrides", "issue"),
    [
        ({"project": None}, ReportingIssue.MISSING_SCOPE),
        ({"reporting_credential_env": "TEST_E004_INFERENCE"}, ReportingIssue.SHARED_CREDENTIAL),
        ({"reporting_credential_env": "TEST_E004_MISSING"}, ReportingIssue.MISSING_CREDENTIAL),
        ({"reporting_key_expires_at": _NOW}, ReportingIssue.EXPIRED_CREDENTIAL),
    ],
)
async def test_admin_configuration_failures_never_dispatch(
    migrated_path: Path, overrides: dict[str, str | datetime | None], issue: ReportingIssue
) -> None:
    settings = Settings(providers={"o": config(ProviderKind.OPENAI, **overrides)})

    def forbidden(_: httpx.Request) -> httpx.Response:
        pytest.fail("an unscoped or invalid Admin credential reached the network")

    async with monitor_for(migrated_path, settings, forbidden) as (monitor, _):
        assert [item.issue for item in await monitor.poll_once("o")] == [issue, issue]


@pytest.mark.anyio
async def test_concurrent_admin_polls_reconcile_once_and_warn_before_expiry(
    migrated_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = Settings(
        providers={
            "o": config(ProviderKind.OPENAI, reporting_key_expires_at=_NOW + timedelta(days=1)),
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        rows = usage() if request.url.path.endswith("completions") else cost("7")
        return page(request, rows=rows)

    async with (
        monitor_for(migrated_path, settings, handler) as (first, db),
        monitor_for(migrated_path, settings, handler) as (second, _),
    ):
        async with create_task_group() as group:
            group.start_soon(first.poll_once, "o")
            group.start_soon(second.poll_once, "o")
        async with db.session() as session:
            rows = (await session.execute(select(usage_transactions.c.measured_cost))).all()
        assert rows == [(Decimal(7),)]
        assert {obs.issue for obs in await first.latest()} == {ReportingIssue.EXPIRING_CREDENTIAL}
    assert "expiring_reporting_credential" in caplog.text
    assert _ADMIN not in caplog.text and _INFERENCE not in caplog.text


@pytest.mark.anyio
async def test_partial_outage_preserves_local_and_inference_observations(
    migrated_path: Path,
) -> None:
    settings = Settings(
        providers={
            "z": config(ProviderKind.ZAI),
            "m": config(ProviderKind.MINIMAX),
            "local": config(ProviderKind.LLAMACPP),
            "absent": config(ProviderKind.MINIMAX, quota_url=None),
        },
    )
    broken = False

    def handler(_: httpx.Request) -> httpx.Response:
        if broken:
            return httpx.Response(429)
        payload = (_FIXTURES / "zai-synthetic.json").read_bytes()
        return httpx.Response(
            200,
            content=payload.replace(
                b'"code": 200',
                b'"api_key": "' + _ADMIN.encode() + b'", "code": 200',
            ),
        )

    async with monitor_for(migrated_path, settings, handler) as (monitor, db):
        async with db.write() as session:
            await session.execute(
                text(
                    "INSERT INTO requests "
                    "(id,idempotency_scope,idempotency_key,input_fingerprint,provider_id,"
                    " model_id,created_at) VALUES ('r','scope','key',:fp,'local','alias',:now)"
                ),
                {"fp": "a" * 64, "now": _NOW.isoformat()},
            )
            await session.execute(
                quota_observations.insert().values(
                    provider_id="z",
                    resource="quota",
                    unit="requests",
                    remaining=Decimal(0),
                    observed_at=_NOW.isoformat(),
                    fresh_until=(_NOW + timedelta(seconds=10)).isoformat(),
                    source="inference",
                    status="observed",
                )
            )
        assert (await monitor.poll_once("absent"))[0].issue is ReportingIssue.UNSUPPORTED
        assert (await monitor.poll_once("m"))[0].remaining is None
        await monitor.poll_once("local")
        await monitor.poll_once("z")
        broken = True
        await monitor.poll_once("z")
        items = {
            (item.provider, item.source, item.resource): item for item in await monitor.latest()
        }
    assert items["z", "reporting:zai", "quota"].issue is ReportingIssue.RATE_LIMITED
    assert items["z", "inference", "quota"].remaining == 0
    assert str(items["z", "reporting:zai", "TOKENS_LIMIT"].usage_percent) == "21.75"
    local = items["local", "local", "accepted_requests"]
    assert local.used == 1 and local.limit is local.remaining is local.raw_payload_ref is None
    archived = b"".join(f.read_bytes() for f in (migrated_path.parent / "report-spool").iterdir())
    assert b"[REDACTED]" in archived and _ADMIN.encode() not in archived


@pytest.mark.anyio
async def test_http_timeout_is_an_unavailable_observation(migrated_path: Path) -> None:
    settings = Settings(providers={"z": config(ProviderKind.ZAI, timeout_seconds=0.01)})

    async def handler(_: httpx.Request) -> httpx.Response:
        await sleep_forever()
        raise AssertionError("timeout did not cancel the request")

    async with monitor_for(migrated_path, settings, handler) as (monitor, _):
        result = await monitor.poll_once("z")
    assert result[0].issue is ReportingIssue.TIMEOUT


@pytest.mark.anyio
async def test_run_caps_concurrency_and_cancels_inflight_poll(migrated_path: Path) -> None:
    settings = Settings(
        providers={name: config(ProviderKind.MINIMAX) for name in ("a", "b", "c")},
        poll_concurrency=1,
    )
    entered, exited = Event(), Event()
    inflight = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal inflight
        inflight += 1
        assert inflight == 1
        entered.set()
        try:
            await sleep_forever()
            raise AssertionError("shutdown did not cancel the request")
        finally:
            inflight -= 1
            exited.set()

    async with monitor_for(migrated_path, settings, handler) as (monitor, _):
        with fail_after(2):
            async with create_task_group() as tasks:
                tasks.start_soon(monitor.run)
                await entered.wait()
                # Give both other provider loops time to attempt to acquire capacity.
                await sleep(0.2)
                tasks.cancel_scope.cancel()
        assert exited.is_set() and inflight == 0
        assert await monitor.latest() == ()
