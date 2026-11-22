"""Adjustment identities bind their original period and roll back atomically."""

from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from _gateway_fixture import NOW, answer, gateway_for, settings_for
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from test_migrations import migrated_path as migrated_path

from emergence.storage.models import audit_events, budget_adjustments


@pytest.mark.anyio
async def test_adjustment_retry_after_restart_and_renewal_keeps_original_period(
    migrated_path: Path,
) -> None:
    async with gateway_for(migrated_path, settings_for(), lambda _: answer()) as (_, ledger, _, _):
        first = await ledger.adjust(
            "p",
            Decimal("1.00"),
            "grant",
            "confirmed",
            "operator",
            NOW,
            idempotency_key="original-period",
        )
    following = NOW + timedelta(days=32)
    async with gateway_for(migrated_path, settings_for(), lambda _: answer()) as (
        _,
        ledger,
        database,
        _,
    ):
        retried = await ledger.adjust(
            "p",
            Decimal(1),
            "grant",
            "confirmed",
            "operator",
            following,
            idempotency_key="original-period",
        )
        assert retried.replayed and retried.adjustment_id == first.adjustment_id
        assert retried.period_id == first.period_id and retried.available == 101
        assert (await ledger.balance("p", following)).available == 100
        async with database.session() as session:
            assert await session.scalar(select(func.count()).select_from(budget_adjustments)) == 1


@pytest.mark.anyio
async def test_failed_adjustment_rolls_back_identity_and_audit_before_retry(
    migrated_path: Path,
) -> None:
    async with gateway_for(migrated_path, settings_for(), lambda _: answer()) as (
        _,
        ledger,
        database,
        _,
    ):
        async with database.write() as session:
            prior_audits = (await session.execute(select(audit_events))).all()
            await session.execute(
                text(
                    "CREATE TRIGGER fail_adjustment BEFORE INSERT ON budget_adjustments "
                    "BEGIN SELECT RAISE(ABORT, 'synthetic write failure'); END"
                )
            )
        with pytest.raises(IntegrityError, match="synthetic write failure"):
            await ledger.adjust(
                "p",
                Decimal(2),
                "grant",
                "confirmed",
                "operator",
                NOW,
                idempotency_key="retry-after-rollback",
            )
        async with database.write() as session:
            await session.execute(text("DROP TRIGGER fail_adjustment"))
            assert (await session.execute(select(audit_events))).all() == prior_audits
        receipt = await ledger.adjust(
            "p",
            Decimal(2),
            "grant",
            "confirmed",
            "operator",
            NOW,
            idempotency_key="retry-after-rollback",
        )
        assert not receipt.replayed and receipt.adjustments == 2 and receipt.available == 102
        async with database.session() as session:
            assert await session.scalar(select(func.count()).select_from(budget_adjustments)) == 1
            assert (
                await session.scalar(select(func.count()).select_from(audit_events))
                == len(prior_audits) + 1
            )
