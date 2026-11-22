"""Durable request states and immutable archive envelopes."""

from datetime import datetime
from enum import StrEnum

from pydantic import AwareDatetime, JsonValue
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.config import SafeModel
from emergence.inference.errors import GatewayError
from emergence.inference.schemas import InferenceResult
from emergence.resources.pricing import stamp
from emergence.storage.models import request_events


class RequestState(StrEnum):
    ADMITTED = "admitted"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    RESOLVED = "resolved"


class FailureRecord(SafeModel):
    code: str
    ambiguous: bool
    status_code: int | None = None
    raw_response: JsonValue = None


class ArchivedOutcome(SafeModel):
    request_id: str
    observed_at: AwareDatetime
    result: InferenceResult | None = None
    failure: FailureRecord | None = None


class GatewayResult(SafeModel):
    request_id: str
    state: RequestState
    result: InferenceResult | None = None
    error_code: str | None = None
    replayed: bool = False


async def append_event(
    session: AsyncSession,
    request_id: str,
    state: RequestState,
    now: datetime,
    *,
    reference: str | None = None,
    code: str | None = None,
) -> None:
    previous = await session.scalar(
        select(func.max(request_events.c.sequence)).where(request_events.c.request_id == request_id)
    )
    await session.execute(
        request_events.insert().values(
            request_id=request_id,
            sequence=0 if previous is None else previous + 1,
            kind=state.value,
            occurred_at=stamp(now),
            archive_record_id=reference,
            outcome_code=code,
        )
    )


async def read_state(session: AsyncSession, request_id: str) -> tuple[RequestState, str | None]:
    latest = (
        await session.execute(
            select(request_events.c.kind, request_events.c.outcome_code)
            .where(request_events.c.request_id == request_id)
            .order_by(request_events.c.sequence.desc())
            .limit(1)
        )
    ).one_or_none()
    if latest is None:
        raise GatewayError("request_not_found", 404)
    try:
        return RequestState(latest.kind), latest.outcome_code
    except ValueError:
        raise GatewayError("request_state_invalid") from None
