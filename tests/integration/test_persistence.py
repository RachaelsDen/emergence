from decimal import Decimal
from pathlib import Path

import pytest
from anyio import fail_after
from sqlalchemy import String, column, insert, select, text, type_coerce
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from test_migrations import migrated_path as migrated_path

from emergence.storage._schema_types import ExactDecimal
from emergence.storage.database import Database
from emergence.storage.models import budget_periods


async def seed_request(session: AsyncSession) -> None:
    await session.execute(text("INSERT INTO providers VALUES ('p', 'openai')"))
    await session.execute(text("INSERT INTO models VALUES ('p', 'alias', 'resolved')"))
    await session.execute(
        text(
            "INSERT INTO requests "
            "(id, idempotency_scope, idempotency_key, input_fingerprint, provider_id, "
            "model_id, created_at) VALUES ('r', 'caller', 'key', :fingerprint, 'p', 'alias', 'now')"
        ),
        {"fingerprint": "f" * 64},
    )


@pytest.mark.anyio
async def test_decimal_precision_survives_a_reopened_database(migrated_path: Path) -> None:
    amount = Decimal("9007199254740993.0000000000000000007")
    async with Database(migrated_path) as database, database.write() as session:
        await seed_request(session)
        await session.execute(
            insert(budget_periods).values(
                id="period",
                provider_id="p",
                currency="USD",
                starts_at="2026-11-01T00:00:00Z",
                ends_at="2026-12-01T00:00:00Z",
                timezone="UTC",
                opening_budget=amount,
            )
        )

    async with Database(migrated_path) as reopened, reopened.session() as session:
        stmt = select(
            column("opening_budget", ExactDecimal),
            type_coerce(text("typeof(opening_budget)"), String),
        ).select_from(budget_periods)
        actual, sqlite_type = (await session.execute(stmt)).one()
        assert actual == amount
        assert sqlite_type == "text"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE requests SET created_at = 'different'",
        "DELETE FROM requests WHERE id = 'r'",
        "INSERT OR REPLACE INTO requests SELECT id, idempotency_scope, idempotency_key, "
        "input_fingerprint, provider_id, model_id, agent_id, experiment_id, 'different' "
        "FROM requests",
        "INSERT INTO requests SELECT 'r2', idempotency_scope, idempotency_key, "
        "input_fingerprint, provider_id, model_id, agent_id, experiment_id, 'new' FROM requests",
    ],
)
async def test_request_authority_rejects_mutation_and_duplicate_identity(
    migrated_path: Path, sql: str
) -> None:
    async with Database(migrated_path) as database:
        async with database.write() as session:
            await seed_request(session)

        with pytest.raises(IntegrityError, match="append-only authority"):
            async with database.write() as session:
                await session.execute(text(sql))

        async with database.session() as session:
            assert (await session.execute(text("SELECT id, created_at FROM requests"))).all() == [
                ("r", "now")
            ]


@pytest.mark.anyio
@pytest.mark.parametrize("duplicate", ["reconciliation", "settlement"])
async def test_reconciliation_and_settlement_cannot_be_recorded_twice(
    migrated_path: Path, duplicate: str
) -> None:
    async with Database(migrated_path) as database:
        async with database.write() as session:
            await seed_request(session)
            for statement in (
                "INSERT INTO price_versions VALUES ('v', 'p', 'alias', 'now', 'USD')",
                "INSERT INTO budget_periods VALUES "
                "('period', 'p', 'USD', 'start', 'stop', 'UTC', '100.00')",
                "INSERT INTO request_reservations VALUES "
                "('r', 'p', 'alias', 'USD', 'period', 'v', '1.00', 'now')",
                "INSERT INTO usage_transactions "
                "(id, reconciliation_id, request_id, provider_id, model_id, currency, occurred_at) "
                "VALUES ('u', 'reconcile', 'r', 'p', 'alias', 'USD', 'now'), "
                "('u2', 'separate', 'r', 'p', 'alias', 'USD', 'now')",
                "INSERT INTO reservation_events "
                "(request_id, sequence, kind, settled_request_id, "
                "usage_transaction_id, occurred_at) "
                "VALUES ('r', 1, 'settled', 'r', 'u', 'now')",
            ):
                await session.execute(text(statement))

        attempts = {
            "reconciliation": "INSERT INTO usage_transactions "
            "(id, reconciliation_id, provider_id, currency, occurred_at) "
            "VALUES ('different', 'reconcile', 'p', 'USD', 'later')",
            "settlement": "INSERT INTO reservation_events "
            "(request_id, sequence, kind, settled_request_id, usage_transaction_id, occurred_at) "
            "VALUES ('r', 2, 'settled', 'r', 'u2', 'later')",
        }
        with pytest.raises(IntegrityError, match="append-only authority"):
            async with database.write() as session:
                await session.execute(text(attempts[duplicate]))


@pytest.mark.anyio
async def test_failed_write_rolls_back_and_read_sessions_cannot_write(migrated_path: Path) -> None:
    async with Database(migrated_path) as database:
        statement = text("INSERT INTO providers VALUES ('p', 'openai')")
        with pytest.raises(IntegrityError, match="FOREIGN KEY"):
            async with database.write() as session:
                await session.execute(statement)
                await session.execute(text("INSERT INTO models VALUES ('other', 'alias', 'x')"))
        with pytest.raises(OperationalError, match="readonly"):
            async with database.session() as session:
                await session.execute(statement)
        async with database.session() as session:
            assert (await session.execute(text("SELECT id FROM providers"))).all() == []


@pytest.mark.anyio
async def test_independent_writer_hits_finite_lock_timeout_before_body(migrated_path: Path) -> None:
    async with Database(migrated_path) as database, Database(migrated_path) as competitor:
        async with database.write() as session:
            await seed_request(session)
            with fail_after(4), pytest.raises(OperationalError, match="database is locked"):
                async with competitor.write():
                    pytest.fail("second writer acquired the lock before the first committed")

        async with competitor.write() as session:
            request = await session.scalar(text("SELECT id FROM requests"))
            assert request == "r"
