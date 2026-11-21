"""SQLAlchemy Core tables and append-only DDL for the initial revision.

Revision 0001 consumes these definitions as a snapshot. Preserve them when
introducing migrations that modify existing operational tables.
"""

from typing import Final

from sqlalchemy import Table, UniqueConstraint

from ._accounting import (
    budget_adjustments,
    budget_period_events,
    budget_periods,
    price_dimensions,
    price_versions,
    request_reservations,
    reservation_events,
    usage_transactions,
)
from ._operational import (
    archive_outbox as archive_outbox,
)
from ._operational import (
    archive_references as archive_references,
)
from ._operational import (
    audit_events,
    models,
    providers,
    quota_observations,
    request_events,
    requests,
)
from ._operational import (
    metadata as metadata,
)

SCHEMA_HEAD: Final = "0001_operational_state"
AUTHORITY_TABLES: Final = (
    providers,
    models,
    requests,
    request_events,
    archive_references,
    quota_observations,
    audit_events,
    price_versions,
    price_dimensions,
    budget_periods,
    budget_period_events,
    request_reservations,
    budget_adjustments,
    usage_transactions,
    reservation_events,
)


def trigger_definitions(table: Table) -> dict[str, str]:
    """Block replacement even for SQLite clients with recursive triggers off."""
    name = table.name
    update = f"{name}_no_update"
    delete = f"{name}_no_delete"
    replace = f"{name}_no_replace"
    identities = [tuple(table.primary_key.columns.keys())]
    for constraint in table.constraints:
        if isinstance(constraint, UniqueConstraint):
            identities.append(tuple(constraint.columns.keys()))
    conflict = " OR ".join(
        f"({' AND '.join(f'{column} = NEW.{column}' for column in identity)})"
        for identity in sorted(identities)
    )
    abort = "BEGIN SELECT RAISE(ABORT, 'append-only authority'); END"
    return {
        update: f"CREATE TRIGGER {update} BEFORE UPDATE ON {name} {abort}",
        delete: f"CREATE TRIGGER {delete} BEFORE DELETE ON {name} {abort}",
        replace: (
            f"CREATE TRIGGER {replace} BEFORE INSERT ON {name} "
            f"WHEN EXISTS (SELECT 1 FROM {name} WHERE {conflict}) {abort}"
        ),
    }
