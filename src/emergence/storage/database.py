"""Non-provisioning async SQLite connections and explicit transaction scopes."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from types import TracebackType
from typing import Final, Self

from anyio import CancelScope
from anyio.lowlevel import checkpoint_if_cancelled
from sqlalchemy import URL, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from .gateway_state import AUTHORITY_TABLES as GATEWAY_AUTHORITY
from .gateway_state import SCHEMA_HEAD
from .gateway_state import metadata as gateway_metadata
from .models import AUTHORITY_TABLES, metadata, trigger_definitions

BUSY_TIMEOUT_MS: Final = 1000


class Database:
    """Local SQLite service handle; only the operator may create or migrate it."""

    def __init__(self, path: Path) -> None:
        self._engine = create_async_engine(
            URL.create(
                "sqlite+aiosqlite",
                database=path.absolute().as_uri(),
                query={"mode": "rw", "uri": "true"},
            ),
            poolclass=NullPool,
            hide_parameters=True,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        with CancelScope(shield=True):
            await self._engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession]:
        """Open an independent read-only snapshot. Do not commit it manually."""
        async with self._transaction(writable=False) as session:
            yield session

    @asynccontextmanager
    async def write(self) -> AsyncGenerator[AsyncSession]:
        """Acquire the SQLite write lock before reads; commit only on success.

        Keep work local and short. Lock contention raises SQLAlchemy OperationalError
        after a finite busy timeout; this layer does not retry transactions.
        """
        async with self._transaction(writable=True) as session:
            yield session

    @asynccontextmanager
    async def _transaction(self, *, writable: bool) -> AsyncGenerator[AsyncSession]:
        session = AsyncSession(self._engine, expire_on_commit=False)
        try:
            # Finish acquiring the connection under a shield so cancellation cannot
            # strand a just-opened driver connection before session cleanup owns it.
            with CancelScope(shield=True):
                connection = await session.connection()
                await connection.exec_driver_sql(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
                await connection.exec_driver_sql("PRAGMA foreign_keys = ON")
                await connection.exec_driver_sql("PRAGMA recursive_triggers = ON")
                await connection.exec_driver_sql("PRAGMA synchronous = FULL")
                if not writable:
                    await connection.exec_driver_sql("PRAGMA query_only = ON")
                await connection.exec_driver_sql("BEGIN IMMEDIATE" if writable else "BEGIN")
            await checkpoint_if_cancelled()
            yield session
            if writable:
                await checkpoint_if_cancelled()
                with CancelScope(shield=True):
                    await session.commit()
        finally:
            with CancelScope(shield=True):
                await session.close()

    async def readiness(self) -> bool:
        """Check the Alembic revision, table columns and authority triggers."""
        try:
            async with self.session() as session:
                versions = await session.scalars(text("SELECT version_num FROM alembic_version"))
                if tuple(versions) != (SCHEMA_HEAD,):
                    return False
                for table in (*metadata.sorted_tables, *gateway_metadata.sorted_tables):
                    await session.execute(select(table).limit(0))
                result = await session.execute(
                    text("SELECT name, sql FROM sqlite_schema WHERE type = 'trigger'")
                )
                definitions = {row[0]: row[1] for row in result}
                return all(
                    definitions.get(name) == sql
                    for table in (*AUTHORITY_TABLES, *GATEWAY_AUTHORITY)
                    for name, sql in trigger_definitions(table).items()
                )
        except SQLAlchemyError:
            return False
