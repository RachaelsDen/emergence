"""Immutable financial audit records share their caller's transaction."""

import json
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from emergence.storage.models import audit_events
from emergence.telemetry.logging import Redactor

from .pricing import BudgetError, stamp


async def record_audit(
    session: AsyncSession,
    redactor: Redactor,
    provider: str,
    request: str | None,
    now: datetime,
    actor: str,
    action: str,
    reason: str,
) -> int:
    if not actor.strip() or not reason.strip():
        raise BudgetError("audit_detail_required")
    result = await session.execute(
        audit_events.insert()
        .values(
            provider_id=provider,
            request_id=request,
            occurred_at=stamp(now),
            actor_id=redactor.text(actor.strip()),
            action=json.dumps([action, redactor.text(reason.strip())]),
        )
        .returning(audit_events.c.id)
    )
    return result.scalar_one()
