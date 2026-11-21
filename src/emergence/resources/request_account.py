"""Original reservation and pricing context for one durable request."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.config import ProviderKind
from emergence.inference.schemas import TokenUsage
from emergence.storage.models import (
    providers,
    request_reservations,
    requests,
    reservation_events,
    usage_transactions,
)

from .pricing import BudgetError, stamp


@dataclass(frozen=True, slots=True)
class RequestAccount:
    request: str
    provider: str
    model: str
    kind: ProviderKind
    period: str | None
    price: str | None
    event: str | None
    sequence: int

    @classmethod
    async def load(cls, session: AsyncSession, request_id: str) -> "RequestAccount":
        r, h, e = requests.c, request_reservations.c, reservation_events.c
        row = (
            await session.execute(
                select(
                    r.provider_id,
                    r.model_id,
                    providers.c.kind,
                    h.budget_period_id,
                    h.price_version_id,
                    e.kind.label("event"),
                    e.sequence,
                )
                .select_from(requests)
                .join(providers, providers.c.id == r.provider_id)
                .outerjoin(request_reservations, h.request_id == r.id)
                .outerjoin(reservation_events, e.request_id == r.id)
                .where(r.id == request_id)
                .order_by(e.sequence.desc())
                .limit(1)
            )
        ).one_or_none()
        if row is None:
            raise BudgetError("unknown_request")
        if row.kind != ProviderKind.LLAMACPP and row.event is None:
            raise BudgetError("reservation_missing")
        return cls(
            request_id,
            row.provider_id,
            row.model_id,
            ProviderKind(row.kind),
            row.budget_period_id,
            row.price_version_id,
            row.event,
            (row.sequence or 0) + 1,
        )

    async def append(
        self, session: AsyncSession, kind: str, now: datetime, tx: str | None = None
    ) -> None:
        await session.execute(
            reservation_events.insert().values(
                request_id=self.request,
                sequence=self.sequence,
                kind=kind,
                settled_request_id=self.request if tx else None,
                usage_transaction_id=tx,
                occurred_at=stamp(now),
            )
        )

    async def charge(
        self, session: AsyncSession, amount: Decimal | None, usage: TokenUsage, at: datetime
    ) -> None:
        uid = uuid4().hex
        await session.execute(
            usage_transactions.insert().values(
                id=uid,
                reconciliation_id=f"request:{self.request}",
                request_id=self.request,
                provider_id=self.provider,
                model_id=self.model,
                budget_period_id=self.period,
                currency="USD",
                estimated_cost=amount,
                virtual_charge=amount
                if self.kind in (ProviderKind.ZAI, ProviderKind.MINIMAX)
                else None,
                occurred_at=stamp(at),
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                output_tokens=usage.output_tokens,
            )
        )
        if self.period is not None:
            await self.append(session, "settled", at, uid)
