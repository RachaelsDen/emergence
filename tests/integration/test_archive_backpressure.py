"""Archive outage/recovery remains visible through the composed authenticated API."""

from pathlib import Path

import httpx
import pytest
from anyio import fail_after, lowlevel
from pydantic import JsonValue, TypeAdapter
from test_archive_delivery import mounted
from test_migrations import migrated_path as migrated_path

from emergence.api.app import create_app
from emergence.config import Settings


async def snapshot_when(client: httpx.AsyncClient, state: str) -> dict[str, JsonValue]:
    with fail_after(5):
        while True:
            response = await client.get("/api/v1/resources")
            assert response.status_code == 200
            body = TypeAdapter(dict[str, JsonValue]).validate_json(response.content)
            archive = body["archive"]
            assert isinstance(archive, dict)
            if archive["state"] == state:
                return body
            await lowlevel.checkpoint()


@pytest.mark.anyio
async def test_archive_outage_recovery_and_offline_restart_keep_metadata_usable_without_metrics(
    migrated_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount = tmp_path / "nas"
    mount.mkdir()
    token = "archive-api-synthetic-aaaaaaaaaaaaaaaa"
    monkeypatch.setenv("TEST_ARCHIVE_OPERATOR", token)
    settings = Settings(
        database_path=migrated_path,
        spool_path=tmp_path / "api-spool",
        archive_path=mount,
        operator_token_env="TEST_ARCHIVE_OPERATOR",
        metrics_enabled=False,
        disk_headroom_bytes=1,
    )
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://core",
            headers={"Authorization": f"Bearer {token}"},
        ) as client,
    ):
        before = await snapshot_when(client, "unavailable")
        archive = before["archive"]
        assert isinstance(archive, dict) and archive["last_error"] == "archive_mount_absent"
        storage = before["storage"]
        assert isinstance(storage, dict) and storage["pending_records"] == 1
        assert (await client.get("/metrics")).status_code == 404
        with monkeypatch.context() as mount_fact:
            mounted(mount_fact, mount)
            after = await snapshot_when(client, "healthy")
            storage = after["storage"]
            assert isinstance(storage, dict)
            assert (
                storage["pending_records"],
                storage["archived_records"],
                storage["used_bytes"],
            ) == (0, 1, 0)

    restarted = create_app(settings)
    async with (
        restarted.router.lifespan_context(restarted),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restarted),
            base_url="http://core",
            headers={"Authorization": f"Bearer {token}"},
        ) as client,
    ):
        assert (await client.get("/health/ready")).status_code == 200
        restored = await snapshot_when(client, "unavailable")
        storage = restored["storage"]
        assert isinstance(storage, dict)
        assert (storage["pending_records"], storage["archived_records"]) == (0, 1)
        assert (await client.get("/api/v1/budgets")).status_code == 200
