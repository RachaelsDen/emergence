"""Atomic request financial lifecycle. Callers own admission and settlement writes."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from typing import Literal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.config import ProviderKind, Settings
from emergence.inference.schemas import InferenceRequest, TokenUsage
from emergence.storage.database import Database
from emergence.storage.models import (
    audit_events,
    budget_adjustments,
    providers,
    request_reservations,
    requests,
    reservation_events,
    usage_transactions,
)
from emergence.telemetry.logging import Redactor

from .budgets import BudgetBalance as BudgetBalance
from .budgets import projection
from .pricing import BudgetError as BudgetError
from .pricing import check_money, effective, install_prices, rates, stamp, version_key


@dataclass(frozen=True, slots=True)
class _Account:
    request: str
    provider: str
    model: str
    kind: ProviderKind
    period: str | None
    price: str | None
    event: str | None
    sequence: int

    @classmethod
    async def load(cls, session: AsyncSession, request_id: str) -> "_Account":
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


class BudgetLedger:
    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database
        self._redactor = Redactor(settings.secrets())
        self._versions = frozenset(version_key(p) for p in settings.prices)

    async def initialize(self) -> None:
        async with self.database.write() as session:
            await install_prices(session, self.settings, self._redactor)

    async def reserve(
        self,
        session: AsyncSession,
        request_id: str,
        request: InferenceRequest,
        now: datetime,
    ) -> Decimal | None:
        provider = self.settings.providers.get(request.provider)
        if provider is None or not provider.enabled or request.model not in provider.models:
            raise BudgetError("unknown_model")
        if provider.kind == ProviderKind.LLAMACPP:
            return None
        if provider.kind == ProviderKind.OPENAI:
            if not provider.safety_limit_confirmed:
                raise BudgetError("safety_limit_unconfirmed")
            if not (provider.organization or "").strip() or not (provider.project or "").strip():
                raise BudgetError("safety_scope_missing")
        version = await effective(session, request.provider, request.model, now)
        if version not in self._versions:
            raise BudgetError("price_unavailable")
        model = provider.models[request.model]
        amount = (await rates(session, version)).cost(
            model.max_input_tokens, request.max_output_tokens
        )
        check_money(amount)
        balance = await projection(session, self.settings, request.provider, now)
        if balance.available is None or balance.period_id is None:
            raise BudgetError("budget_missing")
        with localcontext(prec=2048):
            if (
                balance.safety_exposure is not None
                and balance.safety_exposure + amount > provider.safety_limit
            ):
                raise BudgetError("safety_limit_exceeded")
        if amount > balance.available:
            if self.settings.enforce_budgets:
                raise BudgetError("budget_insufficient")
            await self._audit(
                session,
                request.provider,
                request_id,
                now,
                "system",
                "shadow_denial",
                "budget_insufficient",
            )
        await session.execute(
            request_reservations.insert().values(
                request_id=request_id,
                provider_id=request.provider,
                model_id=request.model,
                currency="USD",
                budget_period_id=balance.period_id,
                price_version_id=version,
                amount=amount,
                created_at=stamp(now),
            )
        )
        await session.execute(
            reservation_events.insert().values(
                request_id=request_id,
                sequence=0,
                kind="reserved",
                occurred_at=stamp(now),
            )
        )
        return amount

    async def settle(
        self, session: AsyncSession, request_id: str, usage: TokenUsage, now: datetime
    ) -> bool:
        account = await _Account.load(session, request_id)
        if account.event == "settled":
            return True
        if account.event == "released":
            raise BudgetError("reservation_released")
        amount = None
        if account.price is None:
            if await session.scalar(
                select(usage_transactions.c.id).where(
                    usage_transactions.c.reconciliation_id == f"request:{request_id}"
                )
            ):
                return True
        elif usage.input_tokens is None or usage.output_tokens is None:
            if account.event != "ambiguous":
                await account.append(session, "ambiguous", now)
            return False
        else:
            amount = (await rates(session, account.price)).cost(
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_input_tokens,
            )
        await account.charge(session, amount, usage, now)
        return True

    async def _audit(
        self,
        session: AsyncSession,
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
                actor_id=self._redactor.text(actor.strip()),
                action=json.dumps([action, self._redactor.text(reason.strip())]),
            )
            .returning(audit_events.c.id)
        )
        return result.scalar_one()

    async def release(
        self,
        session: AsyncSession,
        request_id: str,
        now: datetime,
        reason: str,
        actor: str = "system",
    ) -> None:
        account = await _Account.load(session, request_id)
        if account.event == "released":
            return
        if account.event == "settled":
            raise BudgetError("reservation_settled")
        await self._audit(session, account.provider, request_id, now, actor, "no_charge", reason)
        if account.period is not None:
            await account.append(session, "released", now)

    async def resolve(
        self,
        session: AsyncSession,
        request_id: str,
        amount: Decimal | None,
        now: datetime,
        reason: str,
        actor: str,
    ) -> None:
        if amount is None:
            await self.release(session, request_id, now, reason, actor)
            return
        check_money(amount)
        account = await _Account.load(session, request_id)
        if amount < 0 or account.period is None or account.event in {"settled", "released"}:
            raise BudgetError("invalid_resolution")
        await self._audit(
            session, account.provider, request_id, now, actor, "operator_estimate", reason
        )
        await account.charge(session, amount, TokenUsage(), now)

    async def balance(self, provider: str, at: datetime | None = None) -> BudgetBalance:
        async with self.database.write() as session:
            return await projection(session, self.settings, provider, at or datetime.now(UTC))

    async def adjust(
        self,
        provider: str,
        amount: Decimal,
        kind: Literal["grant", "reduction", "correction"],
        reason: str,
        actor: str,
        at: datetime | None = None,
    ) -> BudgetBalance:
        check_money(amount)
        if (
            not amount
            or kind not in {"grant", "reduction", "correction"}
            or (kind == "grant" and amount < 0 or kind == "reduction" and amount > 0)
        ):
            raise BudgetError("invalid_adjustment")
        now = at or datetime.now(UTC)
        async with self.database.write() as session:
            balance = await projection(session, self.settings, provider, now)
            if balance.period_id is None:
                raise BudgetError("no_dollar_budget")
            aid = await self._audit(session, provider, None, now, actor, kind, reason)
            await session.execute(
                budget_adjustments.insert().values(
                    id=uuid4().hex,
                    budget_period_id=balance.period_id,
                    amount_delta=amount,
                    kind=kind,
                    reason=self._redactor.text(reason.strip()),
                    audit_event_id=aid,
                    occurred_at=stamp(now),
                )
            )
            return await projection(session, self.settings, provider, now)
