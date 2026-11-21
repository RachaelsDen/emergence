"""Known-capacity and freshness admission under the caller's SQLite writer."""

from datetime import datetime
from decimal import Decimal, localcontext

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.config import QuotaPolicy, Settings
from emergence.inference.errors import GatewayError
from emergence.inference.schemas import InferenceRequest
from emergence.storage.capacity import live_claims
from emergence.storage.gateway_state import request_capacity
from emergence.storage.models import quota_observations, request_reservations, requests


async def check_quota(
    settings: Settings,
    session: AsyncSession,
    invocation: InferenceRequest,
    now: datetime,
    amount: Decimal | None,
    *,
    exclude: str | None = None,
) -> None:
    config = settings.providers[invocation.provider]
    model = config.models[invocation.model]
    ordered = select(
        quota_observations,
        func.row_number()
        .over(
            partition_by=quota_observations.c.resource,
            order_by=(
                quota_observations.c.observed_at.desc(),
                quota_observations.c.id.desc(),
            ),
        )
        .label("rank"),
    ).where(quota_observations.c.provider_id == invocation.provider)
    latest = ordered.subquery()
    latest_ids = {
        row.resource: row.id
        for row in await session.execute(
            select(latest.c.resource, latest.c.id).where(latest.c.rank == 1)
        )
    }
    ranked = ordered.where(
        quota_observations.c.status.in_(("verified", "observed")),
        quota_observations.c.remaining.is_not(None),
    ).subquery()
    rows = (await session.execute(select(ranked).where(ranked.c.rank == 1))).all()
    claims = await live_claims(session)
    active = {claim.request_id for claim in claims}
    commitments = (
        await session.execute(
            select(request_capacity, request_reservations.c.amount.label("reserved_usd"))
            .join(requests, requests.c.id == request_capacity.c.request_id)
            .outerjoin(
                request_reservations,
                request_reservations.c.request_id == request_capacity.c.request_id,
            )
            .where(requests.c.provider_id == invocation.provider)
        )
    ).all()
    has_fresh_capacity = False
    for row in rows:
        if row.resets_at and row.unit != "USD" and datetime.fromisoformat(row.resets_at) <= now:
            continue
        if row.remaining <= 0:
            raise GatewayError("provider_capacity_exhausted", 429)
        fresh = (
            latest_ids[row.resource] == row.id
            and row.fresh_until is not None
            and datetime.fromisoformat(row.observed_at)
            <= now
            < datetime.fromisoformat(row.fresh_until)
        )
        observed = datetime.fromisoformat(row.observed_at)
        relevant = [
            item
            for item in commitments
            if item.request_id != exclude
            and (datetime.fromisoformat(item.created_at) >= observed or item.request_id in active)
        ]
        need: Decimal | None = None
        match row.unit:
            case "requests" | "request" | "calls" | "call":
                need = Decimal(len(relevant) + 1)
            case "tokens" | "token":
                need = Decimal(
                    model.max_input_tokens
                    + invocation.max_output_tokens
                    + sum(item.input_tokens_bound + item.output_tokens_bound for item in relevant)
                )
            case "USD":
                if amount is not None:
                    with localcontext(prec=2048):
                        need = amount + sum(
                            (
                                item.reserved_usd
                                for item in relevant
                                if item.reserved_usd is not None
                            ),
                            Decimal(0),
                        )
            case "availability":
                need = Decimal(1)
            case _:
                continue
        if need is not None:
            if need > row.remaining:
                raise GatewayError("provider_capacity_insufficient", 429)
            has_fresh_capacity = has_fresh_capacity or fresh
    if config.quota_policy is QuotaPolicy.REQUIRE_FRESH and not has_fresh_capacity:
        raise GatewayError("provider_capacity_unknown")
