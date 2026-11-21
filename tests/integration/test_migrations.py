import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.exc import OperationalError

from emergence.storage.database import Database

PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def migrated_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "new core #1.sqlite3"
    monkeypatch.setenv("EMERGENCE_DATABASE_PATH", str(path))
    command.upgrade(Config(str(PROJECT / "alembic.ini")), "head")
    return path


@pytest.mark.anyio
async def test_readiness_does_not_create_a_missing_database(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite3"
    async with Database(path) as database:
        assert not await database.readiness()
    assert not path.exists()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "damage", ["missing_version", "stale_version", "missing_trigger", "missing_gateway_trigger"]
)
async def test_readiness_rejects_unusable_schema(migrated_path: Path, damage: str) -> None:
    commands = {
        "missing_version": "DROP TABLE alembic_version",
        "stale_version": "UPDATE alembic_version SET version_num = 'predecessor'",
        "missing_trigger": "DROP TRIGGER reservation_events_no_update",
        "missing_gateway_trigger": "DROP TRIGGER request_capacity_no_update",
    }
    async with Database(migrated_path) as database:
        assert await database.readiness()
        async with database.write() as session:
            connection = await session.connection()
            await connection.exec_driver_sql(commands[damage])
        assert not await database.readiness()


def test_upgrade_rolls_back_ddl_when_existing_schema_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "conflict.sqlite3"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE requests (preexisting TEXT)")
        connection.execute("INSERT INTO requests VALUES ('keep me')")
    monkeypatch.setenv("EMERGENCE_DATABASE_PATH", str(path))

    with pytest.raises(OperationalError, match="already exists"):
        command.upgrade(Config(str(PROJECT / "alembic.ini")), "head")

    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT name FROM sqlite_schema").fetchall() == [("requests",)]
        assert connection.execute("SELECT preexisting FROM requests").fetchall() == [("keep me",)]


def test_gateway_upgrade_preserves_initial_data_and_protects_account_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "upgrade.sqlite3"
    monkeypatch.setenv("EMERGENCE_DATABASE_PATH", str(path))
    config = Config(str(PROJECT / "alembic.ini"))
    command.upgrade(config, "0001_operational_state")
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("INSERT INTO providers VALUES ('existing', 'openai')")
    command.upgrade(config, "head")
    with closing(sqlite3.connect(path)) as connection, connection:
        assert connection.execute("SELECT id, kind FROM providers").fetchall() == [
            ("existing", "openai")
        ]
        connection.execute(
            "INSERT INTO provider_accounts VALUES ('existing', 'original-org', 'original-project')"
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only authority"):
            connection.execute("UPDATE provider_accounts SET organization = 'different-org'")
        assert connection.execute(
            "SELECT organization, project FROM provider_accounts"
        ).fetchall() == [("original-org", "original-project")]
