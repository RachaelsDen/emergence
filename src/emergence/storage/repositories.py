"""Archive authority and its rebuildable outbox within a caller-owned transaction."""

from dataclasses import dataclass

from sqlalchemy import insert as append
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.config import ConfigurationError, ProviderKind, Settings

from .database import Database
from .gateway_state import provider_accounts
from .models import archive_outbox, archive_references, models, providers


@dataclass(frozen=True, slots=True)
class ArchiveReference:
    record_id: str
    checksum: str
    size_bytes: int


class ArchiveReferenceConflict(RuntimeError):
    """An immutable archive reference disagrees with the supplied file."""


async def list_archive_references(session: AsyncSession) -> tuple[ArchiveReference, ...]:
    result = await session.execute(
        select(archive_references).order_by(archive_references.c.record_id)
    )
    return tuple(ArchiveReference(row.record_id, row.checksum, row.size_bytes) for row in result)


async def register_archive_reference(session: AsyncSession, reference: ArchiveReference) -> None:
    """Append once and ensure an outbox row; require Database.write()."""
    existing = (
        await session.execute(
            select(archive_references).where(archive_references.c.record_id == reference.record_id)
        )
    ).one_or_none()
    if existing is None:
        await session.execute(
            insert(archive_references).values(
                record_id=reference.record_id,
                checksum=reference.checksum,
                size_bytes=reference.size_bytes,
            )
        )
    elif (existing.checksum, existing.size_bytes) != (reference.checksum, reference.size_bytes):
        raise ArchiveReferenceConflict
    await session.execute(
        insert(archive_outbox).values(record_id=reference.record_id).on_conflict_do_nothing()
    )


async def ensure_catalog(database: Database, settings: Settings) -> None:
    """Register configured identities without rewriting historical model/provider identity."""
    async with database.write() as session:
        for name, provider in settings.providers.items():
            if not provider.enabled:
                continue
            existing_kind = await session.scalar(
                select(providers.c.kind).where(providers.c.id == name)
            )
            if existing_kind is None:
                await session.execute(append(providers).values(id=name, kind=provider.kind.value))
            elif existing_kind != provider.kind.value:
                raise ConfigurationError("provider_identity_changed_use_new_name")
            organization, project = (
                (provider.organization, provider.project)
                if provider.kind is ProviderKind.OPENAI
                else (None, None)
            )
            if organization is not None and project is not None:
                bound = await session.scalar(
                    select(provider_accounts.c.provider_id).where(
                        provider_accounts.c.organization == organization,
                        provider_accounts.c.project == project,
                        provider_accounts.c.provider_id != name,
                    )
                )
                if bound is not None:
                    raise ConfigurationError("openai_account_already_bound_use_existing_provider")
            account = (
                await session.execute(
                    select(provider_accounts).where(provider_accounts.c.provider_id == name)
                )
            ).one_or_none()
            if account is None:
                await session.execute(
                    append(provider_accounts).values(
                        provider_id=name,
                        organization=organization,
                        project=project,
                    )
                )
            elif (account.organization, account.project) != (organization, project):
                raise ConfigurationError("provider_account_changed_use_new_name")
            for alias, model in provider.models.items():
                existing_model = await session.scalar(
                    select(models.c.upstream_model_id).where(
                        models.c.provider_id == name, models.c.id == alias
                    )
                )
                if existing_model is None:
                    await session.execute(
                        append(models).values(
                            provider_id=name, id=alias, upstream_model_id=model.id
                        )
                    )
                elif existing_model != model.id:
                    raise ConfigurationError("model_identity_changed_use_new_alias")
