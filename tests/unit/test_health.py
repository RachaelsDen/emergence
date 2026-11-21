"""Health is local and follows the application lifecycle."""

from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config

from emergence.api.app import create_app
from emergence.config import ProviderConfig, ProviderKind, QuotaPolicy, Settings


@pytest.fixture
def migrated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    database_path = tmp_path / "core.sqlite3"
    monkeypatch.setenv("EMERGENCE_DATABASE_PATH", str(database_path))
    command.upgrade(Config("alembic.ini"), "head")
    return Settings(database_path=database_path, spool_path=tmp_path / "spool")


@pytest.mark.anyio
async def test_liveness_and_readiness_follow_lifecycle(migrated_settings: Settings) -> None:
    application = create_app(migrated_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        assert (await client.get("/health/live")).json() == {"status": "alive"}
        assert (await client.get("/health/ready")).status_code == 503
        async with application.router.lifespan_context(application):
            assert (await client.get("/health/ready")).json() == {"status": "ready"}
        assert (await client.get("/health/ready")).status_code == 503


@pytest.mark.anyio
async def test_missing_database_does_not_fail_liveness_or_create_database(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "missing.sqlite3", spool_path=tmp_path / "spool")
    application = create_app(settings)
    async with (
        application.router.lifespan_context(application),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client,
    ):
        assert (await client.get("/health/live")).status_code == 200
        assert (await client.get("/health/ready")).status_code == 503
    assert not settings.database_path.exists()


@pytest.mark.anyio
async def test_database_loss_is_reflected_in_readiness(migrated_settings: Settings) -> None:
    application = create_app(migrated_settings)
    async with application.router.lifespan_context(application):
        migrated_settings.database_path.unlink()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            assert (await client.get("/health/live")).status_code == 200
            assert (await client.get("/health/ready")).status_code == 503


@pytest.mark.anyio
async def test_unsafe_spool_is_a_readiness_failure(migrated_settings: Settings) -> None:
    migrated_settings.spool_path.mkdir(mode=0o755)
    migrated_settings.spool_path.chmod(0o755)
    application = create_app(migrated_settings)
    async with (
        application.router.lifespan_context(application),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client,
    ):
        assert (await client.get("/health/live")).status_code == 200
        assert (await client.get("/health/ready")).status_code == 503


@pytest.mark.anyio
@pytest.mark.parametrize("kind", list(ProviderKind))
async def test_provider_without_verified_quota_can_start_under_local_policy(
    migrated_settings: Settings, kind: ProviderKind
) -> None:
    settings = migrated_settings.model_copy(
        update={
            "providers": {
                "configured": ProviderConfig(
                    kind=kind,
                    base_url="https://fixture.invalid/v1",
                    quota_policy=QuotaPolicy.LOCAL_LIMITS_ONLY,
                )
            }
        }
    )
    application = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        async with application.router.lifespan_context(application):
            assert (await client.get("/health/ready")).status_code == 200
        assert (await client.get("/health/ready")).status_code == 503
