"""Atomic request financial lifecycle. Callers own admission and settlement writes."""

import json
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from hashlib import sha256
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.config import ProviderKind, Settings
from emergence.inference.schemas import InferenceRequest, TokenUsage
from emergence.storage.database import Database
from emergence.storage.models import (
    audit_events,
    budget_adjustments,
    budget_periods,
    request_reservations,
    reservation_events,
    usage_transactions,
)
from emergence.telemetry.logging import Redactor

from .audit import record_audit
from .budgets import AdjustmentReceipt, find_period, month, period, projection
from .budgets import BudgetBalance as BudgetBalance
from .pricing import BudgetError as BudgetError
from .pricing import check_money, effective, install_prices, rates, stamp, version_key
from .request_account import RequestAccount


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
        current = await period(session, self.settings, request.provider, now)
        balance = await projection(session, self.settings, request.provider, current)
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
            await record_audit(
                session,
                self._redactor,
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
        account = await RequestAccount.load(session, request_id)
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

    async def release(
        self,
        session: AsyncSession,
        request_id: str,
        now: datetime,
        reason: str,
        actor: str = "system",
    ) -> None:
        account = await RequestAccount.load(session, request_id)
        if account.event == "released":
            return
        if account.event == "settled":
            raise BudgetError("reservation_settled")
        await record_audit(
            session, self._redactor, account.provider, request_id, now, actor, "no_charge", reason
        )
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
        account = await RequestAccount.load(session, request_id)
        if amount < 0 or account.period is None or account.event in {"settled", "released"}:
            raise BudgetError("invalid_resolution")
        await record_audit(
            session,
            self._redactor,
            account.provider,
            request_id,
            now,
            actor,
            "operator_estimate",
            reason,
        )
        await account.charge(session, amount, TokenUsage(), now)

    async def balance(self, provider: str, at: datetime | None = None) -> BudgetBalance:
        config = self.settings.providers.get(provider)
        if config is None:
            raise BudgetError("unknown_provider")
        if config.kind == ProviderKind.LLAMACPP:
            return BudgetBalance(provider=provider, enforcement_mode="local")
        now = at or datetime.now(UTC)
        start, end = month(now, self.settings.accounting_timezone)
        async with self.database.session() as session:
            if (current := await find_period(session, provider, start, end)) is not None:
                return await projection(session, self.settings, provider, current)
        # Recheck under the writer in case another request opened the period.
        async with self.database.write() as session:
            current = await period(session, self.settings, provider, now)
        async with self.database.session() as session:
            return await projection(session, self.settings, provider, current)

    async def adjust(
        self,
        provider: str,
        amount: Decimal,
        kind: Literal["grant", "reduction", "correction"],
        reason: str,
        actor: str,
        at: datetime | None = None,
        *,
        idempotency_key: str,
    ) -> AdjustmentReceipt:
        check_money(amount)
        if (
            not amount
            or kind not in {"grant", "reduction", "correction"}
            or (kind == "grant" and amount < 0 or kind == "reduction" and amount > 0)
        ):
            raise BudgetError("invalid_adjustment")
        if not 1 <= len(idempotency_key) <= 128 or any(
            not "!" <= char <= "~" for char in idempotency_key
        ):
            raise BudgetError("invalid_idempotency_key")
        identity = sha256(json.dumps([actor.strip(), idempotency_key]).encode()).hexdigest()
        fingerprint = sha256(
            json.dumps([provider, amount.as_integer_ratio(), kind, reason.strip()]).encode()
        ).hexdigest()
        action = f"{kind}:{fingerprint}"
        now = at or datetime.now(UTC)
        async with self.database.write() as session:
            existing = (
                await session.execute(
                    select(budget_adjustments, audit_events.c.action, budget_periods.c.timezone)
                    .join(audit_events, audit_events.c.id == budget_adjustments.c.audit_event_id)
                    .join(
                        budget_periods, budget_periods.c.id == budget_adjustments.c.budget_period_id
                    )
                    .where(budget_adjustments.c.id == identity)
                )
            ).one_or_none()
            if existing is not None:
                if existing.action != json.dumps([action, existing.reason]):
                    raise BudgetError("adjustment_idempotency_conflict")
                bound = self.settings.model_copy(update={"accounting_timezone": existing.timezone})
                current = await period(
                    session, bound, provider, datetime.fromisoformat(existing.occurred_at)
                )
                balance = await projection(session, bound, provider, current)
                return AdjustmentReceipt(
                    **balance.model_dump(), adjustment_id=identity, replayed=True
                )
            config = self.settings.providers.get(provider)
            if config is None:
                raise BudgetError("unknown_provider")
            if config.kind == ProviderKind.LLAMACPP:
                raise BudgetError("no_dollar_budget")
            current = await period(session, self.settings, provider, now)
            aid = await record_audit(
                session, self._redactor, provider, None, now, actor, action, reason
            )
            await session.execute(
                budget_adjustments.insert().values(
                    id=identity,
                    budget_period_id=current.id,
                    amount_delta=amount,
                    kind=kind,
                    reason=self._redactor.text(reason.strip()),
                    audit_event_id=aid,
                    occurred_at=stamp(now),
                )
            )
            balance = await projection(session, self.settings, provider, current)
            return AdjustmentReceipt(**balance.model_dump(), adjustment_id=identity, replayed=False)
