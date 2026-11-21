"""Add durable capacity allocations and provider-account identity bindings."""

from alembic import op
from sqlalchemy.schema import CreateTable

from emergence.storage.gateway_state import AUTHORITY_TABLES, metadata
from emergence.storage.models import trigger_definitions

revision = "0002_gateway_state"
down_revision = "0001_operational_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in metadata.sorted_tables:
        op.execute(CreateTable(table))
    for table in AUTHORITY_TABLES:
        for statement in trigger_definitions(table).values():
            op.execute(statement)


def downgrade() -> None:
    for table in reversed(metadata.sorted_tables):
        op.drop_table(table.name)
