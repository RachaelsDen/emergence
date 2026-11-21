"""Calendar period projection and conservative, unattributed cost comparison."""

import json
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, localcontext
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.config import MONTHLY_BUDGET_USD, ProviderKind, SafeModel, Settings
from emergence.storage.models import (
    budget_adjustments,
    budget_period_events,
    budget_periods,
    request_reservations,
    reservation_events,
    usage_transactions,
)

from .pricing import BudgetError, stamp


class BudgetBalance(SafeModel):
    provider: str
    enforcement_mode: Literal["enforced", "shadow", "local"]
    period_id: str | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None
    opening_budget: Decimal | None = None
    adjustments: Decimal | None = None
    spent: Decimal | None = None
    reserved: Decimal | None = None
    available: Decimal | None = None
    estimated_total: Decimal | None = None
    virtual_total: Decimal | None = None
    measured_total: Decimal | None = None
    measured_basis: Literal["unattributed_overlapping_utc_buckets"] | None = None
    safety_exposure: Decimal | None = None
    safety_limit: Decimal | None = None
    safety_limit_confirmed: bool | None = None


class AdjustmentReceipt(BudgetBalance):
    adjustment_id: str
    replayed: bool


def month(at: datetime, zone: str) -> tuple[datetime, datetime]:
    if at.utcoffset() is None:
        raise BudgetError("timezone_required")
    local = at.astimezone(ZoneInfo(zone))
    start = datetime(local.year, local.month, 1, tzinfo=local.tzinfo)
    following = (start + timedelta(days=32)).replace(day=1)
    return start.astimezone(UTC), following.astimezone(UTC)


async def period(
    session: AsyncSession, settings: Settings, provider: str, at: datetime
) -> tuple[str, datetime, datetime, Decimal]:
    start, end = month(at, settings.accounting_timezone)
    periods = budget_periods.c
    existing = (
        await session.execute(
            select(budget_periods).where(
                periods.provider_id == provider,
                periods.currency == "USD",
                periods.starts_at < stamp(end),
                periods.ends_at > stamp(start),
            )
        )
    ).all()
    if existing:
        row = existing[0]
        if len(existing) != 1 or (row.starts_at, row.ends_at) != (stamp(start), stamp(end)):
            raise BudgetError("budget_period_overlap")
        return row.id, start, end, row.opening_budget
    key = uuid4().hex
    opening = MONTHLY_BUDGET_USD
    await session.execute(
        budget_periods.insert().values(
            id=key,
            provider_id=provider,
            currency="USD",
            starts_at=stamp(start),
            ends_at=stamp(end),
            timezone=settings.accounting_timezone,
            opening_budget=opening,
        )
    )
    await session.execute(
        budget_period_events.insert().values(
            budget_period_id=key,
            kind="opened",
            occurred_at=stamp(at),
        )
    )
    closed = select(budget_period_events.c.budget_period_id).where(
        budget_period_events.c.kind == "closed"
    )
    prior = await session.execute(
        select(periods.id, periods.ends_at).where(
            periods.provider_id == provider,
            periods.ends_at <= stamp(start),
            periods.id.not_in(closed),
        )
    )
    for identity, boundary in prior:
        await session.execute(
            budget_period_events.insert().values(
                budget_period_id=identity,
                kind="closed",
                occurred_at=boundary,
            )
        )
    return key, start, end, opening


async def projection(
    session: AsyncSession,
    settings: Settings,
    provider: str,
    at: datetime,
) -> BudgetBalance:
    config = settings.providers.get(provider)
    if config is None:
        raise BudgetError("unknown_provider")
    if config.kind == ProviderKind.LLAMACPP:
        return BudgetBalance(provider=provider, enforcement_mode="local")
    key, start, end, opening = await period(session, settings, provider, at)
    openai = config.kind == ProviderKind.OPENAI
    holds, events, usage = request_reservations.c, reservation_events.c, usage_transactions.c
    terminal = select(events.request_id).where(events.kind.in_(("settled", "released")))
    active = select(holds.budget_period_id, holds.amount).where(
        holds.provider_id == provider,
        holds.request_id.not_in(terminal),
    )
    costs = (
        select(usage_transactions, holds.created_at.label("admission"))
        .outerjoin(
            request_reservations,
            holds.request_id == usage.request_id,
        )
        .where(usage.provider_id == provider, usage.currency == "USD")
    )
    if not openai:
        active = active.where(holds.budget_period_id == key)
        costs = costs.where(usage.budget_period_id == key)
    adjustments = await session.scalars(
        select(budget_adjustments.c.amount_delta).where(
            budget_adjustments.c.budget_period_id == key
        )
    )
    with localcontext(prec=2048):
        adjusted = sum((Decimal(v) for v in adjustments), Decimal(0))
        reserved = all_holds = estimated = virtual = independent = period_independent = Decimal(0)
        for period_id, amount in await session.execute(active):
            all_holds += amount
            if period_id == key:
                reserved += amount
        measured: defaultdict[date, Decimal] = defaultdict(Decimal)
        estimates: defaultdict[date, Decimal] = defaultdict(Decimal)
        period_estimates: defaultdict[date, Decimal] = defaultdict(Decimal)
        for row in await session.execute(costs):
            if row.request_id is None:
                if not openai or row.measured_cost is None:
                    continue
                scope = json.loads(row.scope)
                if (scope["source"], scope["organization"], scope["project"]) != (
                    "reporting:openai",
                    config.organization,
                    config.project,
                ):
                    continue
                bucket = datetime.fromisoformat(row.window_start).astimezone(UTC)
                if bucket != datetime.combine(bucket.date(), datetime.min.time(), UTC) or (
                    datetime.fromisoformat(row.window_end) != bucket + timedelta(days=1)
                ):
                    raise BudgetError("unexpected_cost_window")
                # All line items and their signed revisions belong to this one scoped day.
                measured[bucket.date()] += row.measured_cost
            else:
                estimate = row.estimated_cost or Decimal(0)
                if row.budget_period_id == key:
                    estimated += estimate
                    virtual += row.virtual_charge or Decimal(0)
                if openai:
                    first = datetime.fromisoformat(row.admission).astimezone(UTC).date()
                    last = datetime.fromisoformat(row.occurred_at).astimezone(UTC).date()
                    if first == last:
                        estimates[first] += estimate
                        if row.budget_period_id == key:
                            period_estimates[first] += estimate
                    else:
                        # The billable instant is unknown: do not claim any day's coverage.
                        independent += estimate
                        if row.budget_period_id == key:
                            period_independent += estimate
        exposure = measured_total = None
        spent = virtual
        if openai:
            exposure = (
                all_holds
                + independent
                + sum(
                    max(measured.get(day, Decimal(0)), estimates.get(day, Decimal(0)))
                    for day in estimates.keys() | measured.keys()
                )
            )
            # Keep absent days unknown. max() above must not create evidence of a report.
            period_reports = {
                day: amount
                for day, amount in measured.items()
                if start.date() <= day <= (end - timedelta(microseconds=1)).date()
            }
            measured_total = sum(period_reports.values(), Decimal(0)) if period_reports else None
            spent = period_independent + sum(
                max(period_reports.get(day, Decimal(0)), period_estimates[day])
                for day in period_reports.keys() | period_estimates.keys()
            )
        return BudgetBalance(
            provider=provider,
            period_id=key,
            period_start=start,
            period_end=end,
            opening_budget=opening,
            adjustments=adjusted,
            spent=spent,
            reserved=reserved,
            available=opening + adjusted - spent - reserved,
            estimated_total=estimated,
            virtual_total=None if openai else virtual,
            measured_total=measured_total,
            measured_basis="unattributed_overlapping_utc_buckets"
            if measured_total is not None
            else None,
            safety_exposure=exposure,
            safety_limit=config.safety_limit if openai else None,
            safety_limit_confirmed=config.safety_limit_confirmed if openai else None,
            enforcement_mode="enforced" if settings.enforce_budgets else "shadow",
        )
