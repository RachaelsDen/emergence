"""Operator recovery through the composed app, real SQLite, and synthetic wire."""

from collections import deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, tzinfo
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Final

import httpx
import pytest
from _gateway_fixture import NOW, Clock, answer, invocation, settings_for
from anyio import Event, create_task_group, fail_after
from httpx import AsyncClient
from pydantic import JsonValue
from sqlalchemy import select
from test_api import JSON_OBJECT, OPERATOR, READER
from test_migrations import migrated_path as migrated_path

from emergence.api.app import create_app
from emergence.api.routes import Budgets
from emergence.config import ProviderKind, QuotaPolicy, Settings
from emergence.inference.gateway import GatewayResult as Receipt
from emergence.storage.archive import RecordKey, Spool
from emergence.storage.database import Database
from emergence.storage.models import audit_events, quota_observations, reservation_events
from emergence.storage.repositories import ArchiveReference

PATH: Final = "/api/v1/providers/p/exhaustion/resolution"
RECOVERIES: Final = select(audit_events).where(
    audit_events.c.action == "provider_exhaustion_resolved"
)


@dataclass(frozen=True, slots=True)
class Harness:
    settings: Settings
    clock: Clock
    replies: deque[httpx.Response]
    calls: list[str]

    @asynccontextmanager
    async def open(self) -> AsyncGenerator[AsyncClient]:
        app = create_app(self.settings)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://core",
                headers={"Authorization": f"Bearer {OPERATOR}"},
            ) as client,
        ):
            assert (await client.get("/health/ready")).status_code == 200
            yield client


@pytest.fixture
def harness(migrated_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    clock = Clock(NOW)

    class FrozenTime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return clock().astimezone(tz)

    monkeypatch.setattr("emergence.resources.schemas.datetime", FrozenTime)
    monkeypatch.setattr("emergence.resources.ledger.datetime", FrozenTime)
    for name, value in (
        ("TEST_API_OPERATOR", OPERATOR),
        ("TEST_API_READER", READER),
        ("TEST_E005_INFERENCE", "synthetic-recovery-upstream"),
    ):
        monkeypatch.setenv(name, value)
    base = settings_for(ProviderKind.OPENAI)
    settings = base.model_copy(
        update={
            "providers": {
                **base.providers,
                "disabled": base.providers["p"].model_copy(update={"enabled": False}),
                "local": settings_for(ProviderKind.LLAMACPP).providers["p"],
            },
            "database_path": migrated_path,
            "spool_path": migrated_path.parent / "recovery-spool",
            "operator_token_env": "TEST_API_OPERATOR",
            "reader_token_env": "TEST_API_READER",
        }
    )
    result = Harness(settings, clock, deque(), [])

    def respond(request: httpx.Request) -> httpx.Response:
        result.calls.append(request.url.path)
        return result.replies.popleft() if result.replies else answer()

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(AsyncClient, transport=httpx.MockTransport(respond))
    )
    return result


async def infer(client: AsyncClient, key: str) -> httpx.Response:
    return await client.post(
        "/api/v1/inference",
        json=invocation().model_dump(mode="json"),
        headers={"Idempotency-Key": key},
    )


def denial() -> httpx.Response:
    return httpx.Response(429, json={"error": {"code": "insufficient_quota"}})


@pytest.mark.anyio
async def test_explicit_recovery_survives_restart_without_releasing_unrelated_holds(
    harness: Harness,
) -> None:
    harness.replies.extend((answer(None, None), denial()))
    async with harness.open() as client, Database(harness.settings.database_path) as database:
        held = Receipt.model_validate_json((await infer(client, "hold")).content)
        rejected = Receipt.model_validate_json((await infer(client, "denial")).content)
        assert rejected.state == "failed"
        before = (await client.get(f"/api/v1/history/{rejected.request_id}")).content
        adjustment = await client.post(
            "/api/v1/budgets/adjustments",
            json={"provider": "p", "amount": "-99.97", "kind": "reduction", "reason": "test"},
        )
        assert JSON_OBJECT.validate_json(adjustment.content)["available"] == "0.00"
        harness.clock.at = datetime(2026, 12, 1, tzinfo=UTC)
        renewed = Budgets.model_validate_json((await client.get("/api/v1/budgets")).content)
        assert renewed.budgets[0].available == 100
        blocked = await infer(client, "after-renewal")
        assert blocked.status_code == 429
        assert JSON_OBJECT.validate_json(blocked.content)["error"] == "provider_capacity_exhausted"
        assert len(harness.calls) == 2
        async with database.session() as session:
            original = (await session.execute(select(quota_observations))).mappings().all()
            holds = (await session.execute(select(reservation_events))).all()
        body = {"request_id": rejected.request_id, "reason": f"Confirmed replenishment {OPERATOR}"}
        for token, code in (("", 401), (READER, 403)):
            headers = {"Authorization": f"Bearer {token}"}
            response = await client.post(PATH, json=body, headers=headers)
            assert response.status_code == code
        for provider, code in (("missing", 404), ("disabled", 404), ("local", 400)):
            path = PATH.replace("/p/", f"/{provider}/")
            assert (await client.post(path, json=body)).status_code == code
        for reason in ("", " \t\n", "x" * 2001):
            assert (await client.post(PATH, json={**body, "reason": reason})).status_code == 422
        wrong = {**body, "request_id": held.request_id}
        assert (await client.post(PATH, json=wrong)).status_code == 409
        resolved = await client.post(PATH, json=body)
        assert resolved.status_code == 200
        assert OPERATOR not in resolved.text
        assert (await client.post(PATH, json=body)).status_code == 409
        async with database.session() as session:
            rows = (await session.execute(select(quota_observations))).mappings().all()
            audit = (await session.execute(RECOVERIES)).one()
            assert (await session.execute(select(reservation_events))).all() == holds
        assert all(row in rows for row in original)
        recovery = next(row for row in rows if row["source"] == "operator")
        assert (recovery["remaining"], recovery["unit"], recovery["confidence"]) == (
            1,
            "availability",
            "operator_confirmed_recovery",
        )
        assert audit.actor_id == "operator" and audit.request_id == rejected.request_id
        archive = (harness.settings.spool_path / audit.detail_archive_record_id).read_text()
        assert OPERATOR not in archive and "[REDACTED]" in archive
        assert rejected.request_id in archive
        assert (await client.get(f"/api/v1/history/{rejected.request_id}")).content == before
        assert (
            Receipt.model_validate_json((await infer(client, "recovered")).content).state
            == "completed"
        )
    provider = harness.settings.providers["p"]
    fresh = provider.model_copy(update={"quota_policy": QuotaPolicy.REQUIRE_FRESH})
    settings = harness.settings.model_copy(update={"providers": {"p": fresh}})
    async with replace(harness, settings=settings).open() as client:
        assert (
            Receipt.model_validate_json((await infer(client, "restart")).content).state
            == "completed"
        )
        status = await client.get(f"/api/v1/inference/{held.request_id}")
        assert Receipt.model_validate_json(status.content).state == "ambiguous"
        harness.clock.at += timedelta(seconds=settings.providers["p"].stale_after_seconds)
        stale = await infer(client, "stale-recovery")
        assert JSON_OBJECT.validate_json(stale.content)["error"] == "provider_capacity_unknown"
    assert len(harness.calls) == 4


@pytest.mark.anyio
@pytest.mark.parametrize("unit", ["USD", "requests"])
async def test_recovery_preserves_other_exhausted_resources(harness: Harness, unit: str) -> None:
    harness.replies.append(denial())
    async with harness.open() as client, Database(harness.settings.database_path) as database:
        rejected = Receipt.model_validate_json((await infer(client, "denial")).content)
        async with database.write() as session:
            await session.execute(
                quota_observations.insert().values(
                    provider_id="p",
                    resource="independent",
                    unit=unit,
                    remaining=Decimal(0),
                    observed_at=NOW.isoformat(),
                    source="reporting:test",
                    status="observed",
                )
            )
        body = {"request_id": rejected.request_id, "reason": "Confirmed provider replenishment"}
        assert (await client.post(PATH, json=body)).status_code == 200
        blocked = await infer(client, "independent-gate")
        assert JSON_OBJECT.validate_json(blocked.content)["error"] == "provider_capacity_exhausted"
        assert len(harness.calls) == 1


@pytest.mark.anyio
async def test_writer_rechecks_new_denial_and_serializes_competing_confirmations(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second, release, old_done, new_done = (Event() for _ in range(5))
    results: dict[str, Receipt] = {}
    statuses: list[int] = []

    async def wire(_request: httpx.Request) -> httpx.Response:
        harness.calls.append("inference")
        if len(harness.calls) == 1:
            first.set()
            await second.wait()
        else:
            second.set()
            await release.wait()
        return denial()

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(AsyncClient, transport=httpx.MockTransport(wire))
    )
    publish = Spool.publish

    async def interleave(
        spool: Spool, key: RecordKey, payload: JsonValue, *, claim_request_id: str | None = None
    ) -> ArchiveReference:
        reference = await publish(spool, key, payload, claim_request_id=claim_request_id)
        if key.kind == "provider-recovery":
            release.set()
            await new_done.wait()
        return reference

    async with harness.open() as client:

        async def dispatch(key: str, done: Event) -> None:
            results[key] = Receipt.model_validate_json((await infer(client, key)).content)
            done.set()

        async def confirm(request_id: str) -> None:
            body = {"request_id": request_id, "reason": "Confirmed"}
            response = await client.post(PATH, json=body)
            statuses.append(response.status_code)

        with fail_after(10):
            async with create_task_group() as tasks:
                tasks.start_soon(dispatch, "old", old_done)
                await first.wait()
                tasks.start_soon(dispatch, "new", new_done)
                await old_done.wait()
                monkeypatch.setattr(Spool, "publish", interleave)
                await confirm(results["old"].request_id)
        assert statuses == [409]
        assert (await infer(client, "still-blocked")).status_code == 429
        statuses.clear()
        async with create_task_group() as tasks:
            tasks.start_soon(confirm, results["new"].request_id)
            tasks.start_soon(confirm, results["new"].request_id)
        assert sorted(statuses) == [200, 409]
        async with (
            Database(harness.settings.database_path) as database,
            database.session() as session,
        ):
            audits = (await session.execute(RECOVERIES)).all()
            assert [row.request_id for row in audits] == [results["new"].request_id]
