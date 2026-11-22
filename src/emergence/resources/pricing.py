"""Immutable, operator-sourced USD prices. No provider defaults or discovery."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, localcontext

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.config import PriceConfig, ProviderKind, Settings
from emergence.storage.models import audit_events, price_dimensions, price_versions
from emergence.telemetry.logging import Redactor


class BudgetError(Exception):
    """Only stable codes, never operator text or database parameters."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def stamp(at: datetime) -> str:
    if at.utcoffset() is None:
        raise BudgetError("timezone_required")
    return at.astimezone(UTC).isoformat(timespec="microseconds")


def check_money(amount: Decimal) -> None:
    """Match reporting's supported envelope; allow carry space for all SQLite rows."""
    exponent = amount.as_tuple().exponent
    if not (
        amount.is_finite()
        and isinstance(exponent, int)
        and -1000 <= exponent <= 1000
        and amount.adjusted() <= 1000
    ):
        raise BudgetError("invalid_amount")


@dataclass(frozen=True, slots=True)
class Rates:
    input: Decimal
    output: Decimal
    cached_input: Decimal | None

    def cost(self, inputs: int, outputs: int, cached: int | None = None) -> Decimal:
        """Cache and reasoning are subsets of input and output, not extras."""
        with localcontext(prec=2048):
            hits = cached or 0
            cache_rate = self.input if self.cached_input is None else self.cached_input
            amount = (inputs - hits) * self.input + outputs * self.output + hits * cache_rate
            return amount / 1_000_000


async def effective(session: AsyncSession, provider: str, model: str, now: datetime) -> str:
    pv = price_versions.c
    version: str | None = await session.scalar(
        select(pv.id)
        .where(
            pv.provider_id == provider,
            pv.model_id == model,
            pv.currency == "USD",
            pv.effective_at <= stamp(now),
        )
        .order_by(pv.effective_at.desc())
        .limit(1)
    )
    if version is None:
        raise BudgetError("price_unavailable")
    return version


def version_key(price: PriceConfig) -> str:
    return json.dumps([price.provider, price.model, price.version], separators=(",", ":"))


async def rates(session: AsyncSession, version: str) -> Rates:
    dims = price_dimensions.c
    rows = await session.execute(
        select(dims.dimension, dims.unit, dims.unit_price).where(dims.price_version_id == version)
    )
    values: dict[str, Decimal] = {}
    for name, unit, value in rows:
        if unit != "million_tokens" or name not in {"input", "output", "cached_input"}:
            raise BudgetError("invalid_price_dimensions")
        values[name] = value
    if "input" not in values or "output" not in values:
        raise BudgetError("invalid_price_dimensions")
    return Rates(values["input"], values["output"], values.get("cached_input"))


async def install_prices(session: AsyncSession, settings: Settings, redactor: Redactor) -> None:
    pv = price_versions.c
    for price in settings.prices:
        provider = settings.providers.get(price.provider)
        if (
            not provider
            or price.model not in provider.models
            or provider.kind == ProviderKind.LLAMACPP
        ):
            raise BudgetError("invalid_price_model")
        if not provider.enabled:
            continue
        key, start = version_key(price), stamp(price.effective_at)
        if redactor.text(key) != key or not price.source.strip() or not price.version.strip():
            raise BudgetError("invalid_price_metadata")
        provenance = json.dumps(["price_registered", key, redactor.text(price.source)])
        row = (
            await session.execute(
                select(pv.id, pv.effective_at).where(
                    or_(
                        pv.id == key,
                        (pv.provider_id == price.provider)
                        & (pv.model_id == price.model)
                        & (pv.effective_at == start)
                        & (pv.currency == "USD"),
                    )
                )
            )
        ).first()
        expected = Rates(
            price.input_per_million, price.output_per_million, price.cached_input_per_million
        )
        if row is not None:
            if (
                (row.id, row.effective_at) != (key, start)
                or await rates(session, key) != expected
                or not await session.scalar(
                    select(audit_events.c.id).where(
                        audit_events.c.provider_id == price.provider,
                        audit_events.c.action == provenance,
                    )
                )
            ):
                raise BudgetError("price_version_conflict")
            continue
        await session.execute(
            price_versions.insert().values(
                id=key,
                provider_id=price.provider,
                model_id=price.model,
                effective_at=start,
                currency="USD",
            )
        )
        for name, amount in (
            ("input", expected.input),
            ("output", expected.output),
            ("cached_input", expected.cached_input),
        ):
            if amount is not None:
                await session.execute(
                    price_dimensions.insert().values(
                        price_version_id=key,
                        dimension=name,
                        unit="million_tokens",
                        unit_price=amount,
                    )
                )
        await session.execute(
            audit_events.insert().values(
                provider_id=price.provider,
                actor_id="configuration",
                action=provenance,
                occurred_at=stamp(datetime.now(UTC)),
            )
        )
