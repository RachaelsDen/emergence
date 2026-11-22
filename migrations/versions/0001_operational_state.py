"""Create the initial operational state and append-only protections."""

from alembic import op
from sqlalchemy.schema import CreateIndex, CreateTable

from emergence.storage.models import AUTHORITY_TABLES, metadata, trigger_definitions

revision = "0001_operational_state"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in metadata.sorted_tables:
        op.execute(CreateTable(table))
        for index in table.indexes:
            op.execute(CreateIndex(index))
    for table in AUTHORITY_TABLES:
        for statement in trigger_definitions(table).values():
            op.execute(statement)


def downgrade() -> None:
    for table in reversed(metadata.sorted_tables):
        op.drop_table(table.name)
