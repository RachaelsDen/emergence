"""Explicit operator attestation superseding one inference exhaustion observation."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import AwareDatetime, Field, StringConstraints
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.config import ProviderKind, SafeModel, Settings
from emergence.resources.pricing import stamp
from emergence.storage.archive import RecordKey
from emergence.storage.models import audit_events, quota_observations
from emergence.telemetry.logging import Redactor

from .dependencies import ERROR_DESCRIPTIONS, Authorization, ErrorResponse, Identity, Services


class ExhaustionResolution(SafeModel):
    request_id: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    observation_id: Annotated[int, Field(strict=True, gt=0)] | None = None


class ExhaustionRecovery(SafeModel):
    provider: str
    request_id: str
    resolved_observation_id: int
    observation_id: int
    audit_event_id: int
    observed_at: AwareDatetime
    fresh_until: AwareDatetime


async def _current_target(
    session: AsyncSession, provider: str, body: ExhaustionResolution
) -> tuple[int, datetime]:
    """Use admission's resource ordering, including its timestamp tie breaker."""
    quota = quota_observations.c
    row = (
        await session.execute(
            select(quota_observations)
            .where(quota.provider_id == provider, quota.resource == "inference")
            .order_by(quota.observed_at.desc(), quota.id.desc())
            .limit(1)
        )
    ).one_or_none()
    if (
        row is None
        or row.unit != "availability"
        or row.status != "observed"
        or row.resets_at is not None
    ):
        raise HTTPException(409, "provider_exhaustion_target_changed")
    denial = (
        body.observation_id is None
        and row.source == "inference"
        and row.remaining == Decimal(0)
        and row.confidence == "provider_rejection"
        and row.raw_archive_record_id == RecordKey(body.request_id, "output").record_id
    )
    confirmation = (
        body.observation_id == row.id
        and row.source == "operator"
        and row.remaining == Decimal(1)
        and row.confidence == "operator_confirmed_recovery"
    )
    if confirmation:
        confirmation = (
            await session.scalar(
                select(audit_events.c.id).where(
                    audit_events.c.provider_id == provider,
                    audit_events.c.request_id == body.request_id,
                    audit_events.c.action == "provider_exhaustion_resolved",
                    audit_events.c.detail_archive_record_id == row.raw_archive_record_id,
                )
            )
        ) is not None
    if not (denial or confirmation):
        raise HTTPException(409, "provider_exhaustion_target_changed")
    return row.id, datetime.fromisoformat(row.observed_at)


def install_provider_recovery(
    app: FastAPI, settings: Settings, services: Callable[[], Services]
) -> None:
    auth = Authorization(settings)
    router = APIRouter(
        prefix="/api/v1/providers",
        responses={
            code: {"model": ErrorResponse, "description": description}
            for code, description in ERROR_DESCRIPTIONS.items()
        },
    )

    @router.post("/{provider}/exhaustion/resolution")
    async def resolve_exhaustion(
        provider: str,
        body: ExhaustionResolution,
        identity: Annotated[Identity, Depends(auth.operator)],
    ) -> ExhaustionRecovery:
        if (config := settings.providers.get(provider)) is None or not config.enabled:
            raise HTTPException(404, "provider_not_found")
        if config.kind is ProviderKind.LLAMACPP:
            raise HTTPException(400, "provider_recovery_not_supported")
        gateway = services().gateway
        async with gateway.database.session() as session:
            target = await _current_target(session, provider, body)
        now = gateway.clock().astimezone(UTC)
        # A regressed clock must not append a recovery that admission would rank older.
        if now < target[1]:
            raise HTTPException(409, "provider_exhaustion_target_changed")
        fresh_until = now + timedelta(seconds=config.stale_after_seconds)
        reference = await gateway.spool.publish(
            RecordKey(uuid4().hex, "provider-recovery"),
            {
                "provider": provider,
                "request_id": body.request_id,
                "resolved_observation_id": target[0],
                "original_output_record_id": RecordKey(body.request_id, "output").record_id,
                "actor": identity.actor,
                "occurred_at": stamp(now),
                "reason": Redactor(settings.secrets()).text(body.reason),
                "assertion": "operator_confirmed_recovery",
            },
        )
        async with gateway.database.write() as session:
            if await _current_target(session, provider, body) != target:
                raise HTTPException(409, "provider_exhaustion_target_changed")
            observation_id = (
                await session.execute(
                    quota_observations.insert()
                    .values(
                        provider_id=provider,
                        resource="inference",
                        unit="availability",
                        remaining=Decimal(1),
                        observed_at=stamp(now),
                        fresh_until=stamp(fresh_until),
                        source="operator",
                        confidence="operator_confirmed_recovery",
                        status="observed",
                        raw_archive_record_id=reference.record_id,
                    )
                    .returning(quota_observations.c.id)
                )
            ).scalar_one()
            audit_id = (
                await session.execute(
                    audit_events.insert()
                    .values(
                        occurred_at=stamp(now),
                        actor_id=identity.actor,
                        action="provider_exhaustion_resolved",
                        request_id=body.request_id,
                        provider_id=provider,
                        detail_archive_record_id=reference.record_id,
                    )
                    .returning(audit_events.c.id)
                )
            ).scalar_one()
        return ExhaustionRecovery(
            provider=provider,
            request_id=body.request_id,
            resolved_observation_id=target[0],
            observation_id=observation_id,
            audit_event_id=audit_id,
            observed_at=now,
            fresh_until=fresh_until,
        )

    app.include_router(router)
