"""Configured aliases cannot rewrite the identity of historical requests."""

from pathlib import Path

import pytest
from sqlalchemy import select
from test_migrations import migrated_path as migrated_path

from emergence.config import ConfigurationError, ProviderKind, Settings
from emergence.storage.database import Database
from emergence.storage.models import models, providers
from emergence.storage.repositories import ensure_catalog


def catalog(upstream: str) -> Settings:
    return Settings.model_validate(
        {
            "providers": {
                "configured": {
                    "kind": "zai",
                    "base_url": "https://fixture.invalid/v1",
                    "quota_policy": "local_limits_only",
                    "models": {"alias": {"id": upstream}},
                }
            }
        }
    )


@pytest.mark.anyio
async def test_catalog_registration_is_idempotent_and_preserves_upstream_identity(
    migrated_path: Path,
) -> None:
    async with Database(migrated_path) as database:
        await ensure_catalog(database, catalog("original-model"))
        await ensure_catalog(database, catalog("original-model"))
        with pytest.raises(ConfigurationError, match="model_identity_changed_use_new_alias"):
            await ensure_catalog(database, catalog("replacement-model"))
        async with database.session() as session:
            identities = (await session.execute(select(models.c.upstream_model_id))).scalars().all()
    assert identities == ["original-model"]


@pytest.mark.anyio
async def test_renaming_openai_account_cannot_create_an_independent_safety_cap(
    migrated_path: Path,
) -> None:
    account = (
        catalog("model")
        .providers["configured"]
        .model_copy(
            update={
                "kind": ProviderKind.OPENAI,
                "organization": "same-organization",
                "project": "same-project",
            }
        )
    )
    async with Database(migrated_path) as database:
        await ensure_catalog(database, Settings(providers={"original": account}))
        with pytest.raises(
            ConfigurationError, match="openai_account_already_bound_use_existing_provider"
        ):
            await ensure_catalog(database, Settings(providers={"renamed": account}))
        async with database.session() as session:
            names = (await session.execute(select(providers.c.id))).scalars().all()
    assert names == ["original"]
