"""Reconstruct storage claims from immutable allocations and request events."""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .gateway_state import request_capacity
from .models import archive_references, request_events, requests


@dataclass(frozen=True, slots=True)
class CapacityClaim:
    request_id: str
    provider: str
    state: str
    remaining_bytes: int
    input_record_id: str
    output_record_id: str
    input_tokens_bound: int
    output_tokens_bound: int
    created_at: str


async def live_claims(session: AsyncSession) -> tuple[CapacityClaim, ...]:
    latest = select(
        request_events.c.request_id,
        request_events.c.kind,
        func.row_number()
        .over(
            partition_by=request_events.c.request_id,
            order_by=request_events.c.sequence.desc(),
        )
        .label("rank"),
    ).subquery()
    rows = (
        await session.execute(
            select(request_capacity, requests.c.provider_id, latest.c.kind)
            .join(requests, requests.c.id == request_capacity.c.request_id)
            .join(latest, latest.c.request_id == request_capacity.c.request_id)
            .where(
                latest.c.rank == 1,
                latest.c.kind.in_(("admitted", "dispatched", "ambiguous")),
            )
        )
    ).all()
    claims: list[CapacityClaim] = []
    for row in rows:
        published = await session.scalars(
            select(archive_references.c.size_bytes).where(
                archive_references.c.record_id.in_((row.input_record_id, row.output_record_id))
            )
        )
        # Once the HTTP operation has terminated ambiguously no more output can
        # arrive through this process. Monetary holds and concurrency remain;
        # unused storage can be released independently of financial uncertainty.
        remaining = row.allocated_bytes - sum(published) if row.kind != "ambiguous" else 0
        claims.append(
            CapacityClaim(
                request_id=row.request_id,
                provider=row.provider_id,
                state=row.kind,
                remaining_bytes=remaining,
                input_record_id=row.input_record_id,
                output_record_id=row.output_record_id,
                input_tokens_bound=row.input_tokens_bound,
                output_tokens_bound=row.output_tokens_bound,
                created_at=row.created_at,
            )
        )
    return tuple(claims)
