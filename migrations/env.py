"""Operator-run migrations for the configured local SQLite database."""

import os
from pathlib import Path

from alembic import context
from sqlalchemy import URL, create_engine
from sqlalchemy.pool import NullPool

from emergence.config import load_settings


def run_migrations() -> None:
    if context.is_offline_mode():
        context.configure(dialect_name="sqlite", literal_binds=True, transactional_ddl=True)
        with context.begin_transaction():
            context.run_migrations()
        return

    override = os.environ.get("EMERGENCE_DATABASE_PATH")
    path = Path(override) if override else load_settings().database_path
    engine = create_engine(
        URL.create("sqlite", database=str(path.absolute())),
        poolclass=NullPool,
        hide_parameters=True,
    )
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("PRAGMA foreign_keys = ON")
            connection.exec_driver_sql("PRAGMA recursive_triggers = ON")
            connection.exec_driver_sql("PRAGMA busy_timeout = 1000")
            connection.exec_driver_sql("PRAGMA synchronous = FULL")
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            context.configure(connection=connection, transactional_ddl=True)
            context.run_migrations()
    finally:
        engine.dispose()


run_migrations()
