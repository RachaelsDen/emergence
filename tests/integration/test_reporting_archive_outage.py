from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from test_archive_backpressure import snapshot_when
from test_archive_delivery import mounted, run_until
from test_migrations import migrated_path as migrated_path
from test_quota_polling import config

from emergence.api.app import create_app
from emergence.api.routes import Resources
from emergence.config import ProviderKind, Settings
from emergence.resources.reporting_store import ReportingStore
from emergence.resources.schemas import CostBucket, QuotaObservation, ReportScope, Window
from emergence.storage.archive import Spool
from emergence.storage.database import Database
from emergence.storage.delivery import ArchiveDelivery
from emergence.storage.repositories import ensure_catalog
from emergence.telemetry.logging import Redactor


@pytest.mark.anyio
async def test_reclaimed_observations_keep_resources_visible_until_verified_read_returns(
    migrated_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount = tmp_path / "nas"
    mount.mkdir()
    token = "synthetic-reclaimed-report-operator"
    monkeypatch.setenv("TEST_ARCHIVE_OPERATOR", token)
    monkeypatch.delenv("TEST_E004_ADMIN", raising=False)
    settings = Settings(
        database_path=migrated_path,
        spool_path=tmp_path / "spool",
        archive_path=mount,
        operator_token_env="TEST_ARCHIVE_OPERATOR",
        metrics_enabled=False,
        disk_headroom_bytes=1,
        providers={
            "z": config(ProviderKind.ZAI, quota_url=None),
            "o": config(ProviderKind.OPENAI),
        },
    )
    day = datetime(2026, 11, 18, tzinfo=UTC)
    window = Window(start=day, end=day + timedelta(days=1))
    async with Database(migrated_path) as db:
        await ensure_catalog(db, settings)
        spool = Spool(settings.spool_path, db, Redactor(), archive_path=mount)
        store = ReportingStore(db, spool)
        (z,) = await store.append(
            (
                QuotaObservation(
                    provider="z",
                    resource="TOKENS_LIMIT",
                    unit="percent",
                    observed_at=day,
                    source="reporting:zai",
                    status="experimental",
                    usage_percent=Decimal("21.75"),
                ),
            ),
            60,
            ({"data": {"limits": [{"type": "TOKENS_LIMIT", "percentage": Decimal("21.75")}]}},),
        )
        (o,) = await store.append(
            (
                QuotaObservation(
                    provider="o",
                    resource="costs",
                    unit="USD",
                    window=window,
                    observed_at=day + timedelta(days=1),
                    source="reporting:openai",
                    scope=ReportScope(organization="org_fixture", project="proj_fixture"),
                    costs=(CostBucket(window=window, line_item="A", amount=Decimal(7)),),
                    status="verified",
                ),
            ),
            60,
            ({"fixture": True},),
        )
        with monkeypatch.context() as present:
            mounted(present, mount)
            await run_until(ArchiveDelivery(spool, db), "healthy")
            assert (await spool.stats()).used_bytes == 0

    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://core",
            headers={"Authorization": f"Bearer {token}"},
        ) as client,
    ):
        assert (await client.get("/health/ready")).status_code == 200
        assert (await client.get("/metrics")).status_code == 404
        snapshot = Resources.model_validate(await snapshot_when(client, "unavailable"))
        assert snapshot.archive.last_error == "archive_mount_absent"
        assert snapshot.storage.archived_records == 2 and snapshot.storage.pending_records >= 1
        assert 0 < snapshot.storage.used_bytes < snapshot.storage.limit_bytes
        items = {obs.raw_payload_ref: obs for obs in snapshot.observations}
        assert items[z.raw_payload_ref].usage_percent == Decimal("21.75")
        assert items[z.raw_payload_ref].remaining is None
        unavailable = items[o.raw_payload_ref]
        assert unavailable.window == window and unavailable.scope is None
        for obs in (unavailable, items[z.raw_payload_ref]):
            assert obs.status == "unavailable" and obs.issue == "archive_detail_unavailable"

        mounted(monkeypatch, mount)
        recovered = Resources.model_validate(await snapshot_when(client, "healthy"))
        items = {obs.raw_payload_ref: obs for obs in recovered.observations}
        assert items[o.raw_payload_ref].scope == o.scope
        assert items[o.raw_payload_ref].status == "verified"
        assert items[o.raw_payload_ref].costs == o.costs
        assert items[z.raw_payload_ref].issue is None

        assert o.raw_payload_ref is not None
        record = mount / o.raw_payload_ref
        record.chmod(0o600)
        record.write_bytes(b"damaged")
        record.chmod(0o400)
        response = await client.get("/api/v1/resources")
        assert (response.status_code, response.json()) == (
            507,
            {"error": "archive_checksum_mismatch"},
        )
