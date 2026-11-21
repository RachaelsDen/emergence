from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from anyio import Event, create_task_group
from sqlalchemy.ext.asyncio import AsyncSession
from test_migrations import migrated_path as migrated_path

from emergence.config import (
    ModelConfig,
    PriceConfig,
    ProviderConfig,
    ProviderKind,
    QuotaPolicy,
    Settings,
)
from emergence.inference.schemas import InferenceRequest, Message, TokenUsage
from emergence.resources.ledger import BudgetError, BudgetLedger
from emergence.storage.database import Database
from emergence.storage.models import requests
from emergence.storage.repositories import ensure_catalog

NOW = datetime(2026, 11, 29, 12, tzinfo=UTC)


def configuration(
    kind: ProviderKind = ProviderKind.ZAI,
    *,
    confirmed: bool = True,
    cap: int = 100,
    shadow: bool = False,
    zone: str = "UTC",
    project: str | None = "project",
) -> Settings:
    cfg = ProviderConfig(
        kind=kind,
        base_url="https://fixture.invalid",
        models={"alias": ModelConfig(id="upstream", max_input_tokens=1)},
        quota_policy=QuotaPolicy.LOCAL_LIMITS_ONLY,
        safety_limit_confirmed=confirmed,
        safety_limit=Decimal(cap),
        organization="org",
        project=project,
    )
    # Synthetic arithmetic only: one dollar per input/output token.
    price = PriceConfig(
        version="fixture-v1",
        source="fixture tariff",
        effective_at=NOW.replace(day=1),
        provider="p",
        model="alias",
        input_per_million=Decimal(1_000_000),
        output_per_million=Decimal(1_000_000),
        cached_input_per_million=Decimal(0),
    )
    return Settings(
        providers={"p": cfg},
        prices=() if kind == ProviderKind.LLAMACPP else (price,),
        enforce_budgets=not shadow,
        accounting_timezone=zone,
    )


async def prepared(db: Database, settings: Settings) -> BudgetLedger:
    await ensure_catalog(db, settings)
    ledger = BudgetLedger(settings, db)
    await ledger.initialize()
    return ledger


async def admit(
    session: AsyncSession, ledger: BudgetLedger, key: str, tokens: int, now: datetime = NOW
) -> Decimal | None:
    request = InferenceRequest(
        provider="p",
        model="alias",
        max_output_tokens=tokens,
        messages=(Message(role="user", content="fixture"),),
    )
    await session.execute(
        requests.insert().values(
            id=key,
            idempotency_scope="test",
            idempotency_key=key,
            input_fingerprint="f" * 64,
            provider_id=request.provider,
            model_id=request.model,
            created_at=now.isoformat(),
        )
    )
    return await ledger.reserve(session, key, request, now)


@pytest.mark.anyio
async def test_independent_writers_compete_for_one_remaining_hold(migrated_path: Path) -> None:
    settings = configuration()
    async with Database(migrated_path) as db:
        ledger = await prepared(db, settings)
        async with db.write() as session:
            await admit(session, ledger, "spent", 89)
            await ledger.settle(session, "spent", TokenUsage(input_tokens=1, output_tokens=89), NOW)
        outcomes: list[str] = []

        async def contender(key: str) -> None:
            async with Database(migrated_path) as peer:
                try:
                    async with peer.write() as session:
                        await admit(session, BudgetLedger(settings, peer), key, 9)
                    outcomes.append("admitted")
                except BudgetError as error:
                    outcomes.append(error.code)

        async with create_task_group() as group:
            for key in ("one", "two", "three", "four"):
                group.start_soon(contender, key)
        assert sorted(outcomes) == ["admitted", *(["budget_insufficient"] * 3)]
        balance = await ledger.balance("p", NOW)
        assert (balance.spent, balance.reserved, balance.available) == (90, 10, 0)


class SimulatedCrash(Exception):
    pass


@pytest.mark.anyio
@pytest.mark.parametrize("crash", [True, False])
async def test_replacing_hold_never_exposes_intermediate_free_balance(
    migrated_path: Path, crash: bool
) -> None:
    settings = configuration()
    async with Database(migrated_path) as db, Database(migrated_path) as peer:
        ledger, other = await prepared(db, settings), BudgetLedger(settings, peer)
        async with db.write() as session:
            await admit(session, ledger, "first", 89)
            await ledger.settle(session, "first", TokenUsage(input_tokens=1, output_tokens=89), NOW)
            await admit(session, ledger, "held", 9)
        replacing, racing = Event(), Event()

        async def replace() -> None:
            try:
                async with db.write() as session:
                    await ledger.settle(
                        session, "held", TokenUsage(input_tokens=1, output_tokens=9), NOW
                    )
                    replacing.set()
                    await racing.wait()
                    if crash:
                        raise SimulatedCrash
            except SimulatedCrash:
                pass

        async def race() -> None:
            await replacing.wait()
            racing.set()
            with pytest.raises(BudgetError, match="budget_insufficient"):
                async with peer.write() as session:
                    await admit(session, other, "too-much", 1)

        async with create_task_group() as group:
            group.start_soon(replace)
            group.start_soon(race)
        actual = await other.balance("p", NOW)
        assert (actual.spent, actual.reserved, actual.available) == (
            (90, 10, 0) if crash else (100, 0, 0)
        )
