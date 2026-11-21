from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import select
from test_budget_concurrency import NOW, admit, configuration, prepared
from test_migrations import migrated_path as migrated_path

from emergence.config import ProviderKind
from emergence.inference.schemas import TokenUsage
from emergence.resources.ledger import BudgetError, BudgetLedger
from emergence.resources.reporting_store import ReportingStore
from emergence.resources.schemas import CostBucket, QuotaObservation, ReportScope, Window
from emergence.storage.archive import Spool
from emergence.storage.database import Database
from emergence.storage.models import audit_events, budget_adjustments, usage_transactions
from emergence.telemetry.logging import Redactor


@pytest.mark.anyio
async def test_late_completion_uses_original_period_and_price(migrated_path: Path) -> None:
    settings = configuration(ProviderKind.MINIMAX)
    old = settings.prices[0]
    later = NOW.replace(month=12, day=1)
    after = old.model_copy(
        update={"version": "v2", "effective_at": later, "output_per_million": Decimal("125000")}
    )
    settings = settings.model_copy(update={"prices": (old, after)})
    async with Database(migrated_path) as db:
        ledger = await prepared(db, settings)
        await ledger.initialize()
        async with db.write() as session:
            await admit(session, ledger, "old", 99)
            assert not await ledger.settle(session, "old", TokenUsage(input_tokens=1), NOW)
        prior = await ledger.balance("p", NOW)
        assert (prior.reserved, prior.available) == (100, 0)
        with pytest.raises(BudgetError, match="budget_insufficient"):
            async with db.write() as session:
                await admit(session, ledger, "denied", 1)
    async with Database(migrated_path) as reopened:
        ledger = await prepared(reopened, settings)
        renewed = await ledger.balance("p", later)
        assert prior.period_id != renewed.period_id
        assert (renewed.opening_budget, renewed.reserved, renewed.available) == (100, 0, 100)
        async with reopened.write() as session:
            assert await admit(session, ledger, "new", 8, later) == 2
            actual = TokenUsage(
                input_tokens=1, cached_input_tokens=1, output_tokens=101, reasoning_tokens=8
            )
            assert await ledger.settle(session, "old", actual, later)
            assert await ledger.settle(session, "old", actual, later)
        late = await ledger.balance("p", NOW)
        assert (late.estimated_total, late.virtual_total, late.measured_total, late.available) == (
            101,
            101,
            None,
            -1,
        )
        assert (await ledger.balance("p", later)).available == 98


@pytest.mark.anyio
async def test_immutable_prices_do_not_allow_redefinition_or_silent_fallback(
    migrated_path: Path,
) -> None:
    settings = configuration(shadow=True)
    async with Database(migrated_path) as db:
        await prepared(db, settings)
        old = settings.prices[0]
        for change in (
            {"source": "other"},
            {"output_per_million": Decimal(0)},
            {"version": "duplicate-instant"},
        ):
            altered = settings.model_copy(update={"prices": (old.model_copy(update=change),)})
            with pytest.raises(BudgetError, match="price_version_conflict"):
                await BudgetLedger(altered, db).initialize()
        for absent, instant in (
            (settings, NOW.replace(month=10)),
            (settings.model_copy(update={"prices": ()}), NOW),
        ):
            with pytest.raises(BudgetError, match="price_unavailable"):
                async with db.write() as session:
                    await admit(session, BudgetLedger(absent, db), "missing", 1, instant)


@pytest.mark.anyio
async def test_audited_reductions_and_resolution_preserve_negative_balances(
    migrated_path: Path,
) -> None:
    async with Database(migrated_path) as db:
        ledger = await prepared(db, configuration())
        async with db.write() as session:
            await admit(session, ledger, "held", 2)
            await admit(session, ledger, "uncharged", 1)
        for reason, actor in (("", "operator"), ("checked", "  ")):
            with pytest.raises(BudgetError, match="audit_detail_required"):
                await ledger.adjust("p", Decimal(1), "grant", reason, actor, NOW)
        balance = await ledger.adjust(
            "p", Decimal(-100), "reduction", "password=private", "Bearer private", NOW
        )
        assert (balance.available, balance.adjustments, balance.reserved) == (-5, -100, 5)
        with pytest.raises(BudgetError, match="budget_insufficient"):
            async with db.write() as session:
                await admit(session, ledger, "blocked", 1)
        async with db.write() as session:
            await ledger.resolve(session, "held", Decimal("7.25"), NOW, "case#123", "operator")
            await ledger.resolve(
                session, "uncharged", None, NOW, "provider voided invoice", "operator"
            )
        balance = await ledger.adjust(
            "p", Decimal("2.01"), "correction", "entry #1", "operator", NOW
        )
        assert (balance.spent, balance.available, balance.reserved, balance.measured_total) == (
            Decimal("7.25"),
            Decimal("-5.24"),
            0,
            None,
        )
        async with db.session() as session:
            assert list(
                await session.scalars(
                    select(budget_adjustments.c.reason).order_by(
                        budget_adjustments.c.audit_event_id
                    )
                )
            ) == ["[REDACTED]", "entry #1"]
            audit = (
                await session.execute(select(audit_events.c.actor_id, audit_events.c.action))
            ).all()
            assert not any("private" in str(row) for row in audit)
            assert any(
                "operator_estimate" in action and actor == "operator" for actor, action in audit
            )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("confirmed", "project", "code"),
    [(False, "project", "safety_limit_unconfirmed"), (True, None, "safety_scope_missing")],
)
async def test_proposed_limit_and_missing_scope_fail_closed_even_in_shadow(
    migrated_path: Path, confirmed: bool, project: str | None, code: str
) -> None:
    async with Database(migrated_path) as db:
        ledger = await prepared(
            db,
            configuration(
                ProviderKind.OPENAI, confirmed=confirmed, project=project, shadow=True, cap=95
            ),
        )
        with pytest.raises(BudgetError, match=code):
            async with db.write() as session:
                await admit(session, ledger, "denied", 1)


@pytest.mark.anyio
async def test_safety_persists_after_exhausted_virtual_month_renews(migrated_path: Path) -> None:
    settings = configuration(ProviderKind.OPENAI)
    async with Database(migrated_path) as db:
        ledger = await prepared(db, settings)
        async with db.write() as session:
            await admit(session, ledger, "used", 99)
            await ledger.settle(session, "used", TokenUsage(input_tokens=1, output_tokens=99), NOW)
        prior = await ledger.balance("p", NOW)
        assert (prior.available, prior.safety_exposure, prior.measured_total) == (0, 100, None)
    async with Database(migrated_path) as db:
        ledger = BudgetLedger(settings.model_copy(update={"enforce_budgets": False}), db)
        next_month = NOW.replace(month=12, day=1)
        renewed = await ledger.balance("p", next_month)
        assert (renewed.available, renewed.safety_exposure) == (100, 100)
        with pytest.raises(BudgetError, match="safety_limit_exceeded"):
            async with db.write() as session:
                await admit(session, ledger, "denied", 1, next_month)


async def report(
    store: ReportingStore, at: datetime, amount: int, project: str = "project"
) -> None:
    start = at.replace(hour=0, minute=0, second=0, microsecond=0)
    window = Window(start=start, end=start + timedelta(days=1))
    obs = QuotaObservation(
        provider="p",
        resource="costs",
        observed_at=start + timedelta(days=1),
        source="reporting:openai",
        status="verified",
        scope=ReportScope(organization="org", project=project),
        costs=(
            CostBucket(window=window, line_item="input", amount=Decimal(amount) - 1),
            CostBucket(window=window, line_item="output", amount=Decimal(1)),
        ),
    )
    await store.append((obs,), 60, ({"fixture": True},))


@pytest.mark.anyio
async def test_bucket_revisions_only_cover_same_day_estimates_and_never_holds(
    migrated_path: Path,
) -> None:
    async with Database(migrated_path) as db:
        ledger = await prepared(db, configuration(ProviderKind.OPENAI, zone="America/New_York"))
        store = ReportingStore(db, Spool(migrated_path.parent / "reports", db, Redactor()))
        async with db.write() as session:
            await admit(session, ledger, "uncertain", 1)
            assert not await ledger.settle(session, "uncertain", TokenUsage(), NOW)
            for key, day, tokens in (
                ("overlap", NOW, 5),
                ("uncovered", NOW - timedelta(days=1), 2),
            ):
                await admit(session, ledger, key, tokens, day)
                await ledger.settle(
                    session, key, TokenUsage(input_tokens=1, output_tokens=tokens), day
                )
        await report(store, NOW, 90, "different-project")
        for amount in (7, 7, 8, 7):
            await report(store, NOW, amount)
            actual = await ledger.balance("p", NOW)
            assert (actual.measured_total, actual.estimated_total, actual.virtual_total) == (
                amount,
                9,
                None,
            )
            assert (actual.spent, actual.reserved, actual.safety_exposure) == (
                amount + 3,
                2,
                amount + 5,
            )
        # This independent UTC day touches two New York accounting months; never prorate it.
        boundary = NOW.replace(month=12, day=1)
        await report(store, boundary, 11)
        old, new = await ledger.balance("p", NOW), await ledger.balance("p", boundary)
        assert (old.measured_total, new.measured_total, new.measured_basis) == (
            18,
            11,
            "unattributed_overlapping_utc_buckets",
        )
        assert old.safety_exposure == new.safety_exposure == 23


@pytest.mark.anyio
async def test_local_unknown_usage_is_terminal_without_dollars(migrated_path: Path) -> None:
    async with Database(migrated_path) as db:
        ledger = await prepared(db, configuration(ProviderKind.LLAMACPP))
        async with db.write() as session:
            assert await admit(session, ledger, "local", 10) is None
            assert await ledger.settle(session, "local", TokenUsage(), NOW)
            assert await ledger.settle(session, "local", TokenUsage(), NOW)
        balance = await ledger.balance("p", NOW)
        assert (balance.available, balance.reserved, balance.spent, balance.measured_total) == (
            None,
            None,
            None,
            None,
        )
        async with db.session() as session:
            assert (
                await session.execute(
                    select(
                        usage_transactions.c.input_tokens,
                        usage_transactions.c.estimated_cost,
                        usage_transactions.c.virtual_charge,
                    )
                )
            ).all() == [(None, None, None)]
