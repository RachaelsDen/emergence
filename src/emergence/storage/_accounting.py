"""Frozen initial revision: immutable financial authority."""

from typing import Final

from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Table,
    Text,
    UniqueConstraint,
)

from ._operational import metadata
from ._schema_types import ExactDecimal, decimal_text

price_versions: Final = Table(
    "price_versions",
    metadata,
    Column("id", Text, primary_key=True, nullable=False),
    Column("provider_id", Text, nullable=False),
    Column("model_id", Text, nullable=False),
    Column("effective_at", Text, nullable=False),
    Column("currency", Text, nullable=False),
    ForeignKeyConstraint(["provider_id", "model_id"], ["models.provider_id", "models.id"]),
    UniqueConstraint("provider_id", "model_id", "effective_at", "currency"),
    UniqueConstraint("id", "provider_id", "model_id", "currency"),
    CheckConstraint("currency GLOB '[A-Z][A-Z][A-Z]'"),
)
price_dimensions: Final = Table(
    "price_dimensions",
    metadata,
    Column(
        "price_version_id", Text, ForeignKey("price_versions.id"), primary_key=True, nullable=False
    ),
    Column("dimension", Text, primary_key=True, nullable=False),
    Column("unit", Text, nullable=False),
    Column("unit_price", ExactDecimal, nullable=False),
    decimal_text("unit_price"),
    CheckConstraint("unit_price NOT GLOB '-*'"),
)
budget_periods: Final = Table(
    "budget_periods",
    metadata,
    Column("id", Text, primary_key=True, nullable=False),
    Column("provider_id", Text, ForeignKey("providers.id"), nullable=False),
    Column("currency", Text, nullable=False),
    Column("starts_at", Text, nullable=False),
    Column("ends_at", Text, nullable=False),
    Column("timezone", Text, nullable=False),
    Column("opening_budget", ExactDecimal, nullable=False),
    UniqueConstraint("provider_id", "currency", "starts_at"),
    UniqueConstraint("id", "provider_id", "currency"),
    CheckConstraint("currency GLOB '[A-Z][A-Z][A-Z]' AND starts_at < ends_at"),
    decimal_text("opening_budget"),
    CheckConstraint("opening_budget NOT GLOB '-*'"),
)
budget_period_events: Final = Table(
    "budget_period_events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("budget_period_id", Text, ForeignKey("budget_periods.id"), nullable=False),
    Column("kind", Text, nullable=False),
    Column("occurred_at", Text, nullable=False),
    UniqueConstraint("budget_period_id", "kind"),
    CheckConstraint("kind IN ('opened', 'closed')"),
)
request_reservations: Final = Table(
    "request_reservations",
    metadata,
    Column("request_id", Text, primary_key=True, nullable=False),
    Column("provider_id", Text, nullable=False),
    Column("model_id", Text, nullable=False),
    Column("currency", Text, nullable=False),
    Column("budget_period_id", Text, nullable=False),
    Column("price_version_id", Text, nullable=False),
    Column("amount", ExactDecimal, nullable=False),
    Column("created_at", Text, nullable=False),
    ForeignKeyConstraint(
        ["request_id", "provider_id", "model_id"],
        ["requests.id", "requests.provider_id", "requests.model_id"],
    ),
    ForeignKeyConstraint(
        ["budget_period_id", "provider_id", "currency"],
        ["budget_periods.id", "budget_periods.provider_id", "budget_periods.currency"],
    ),
    ForeignKeyConstraint(
        ["price_version_id", "provider_id", "model_id", "currency"],
        [
            "price_versions.id",
            "price_versions.provider_id",
            "price_versions.model_id",
            "price_versions.currency",
        ],
    ),
    decimal_text("amount"),
    CheckConstraint("amount NOT GLOB '-*'"),
)
budget_adjustments: Final = Table(
    "budget_adjustments",
    metadata,
    Column("id", Text, primary_key=True, nullable=False),
    Column("budget_period_id", Text, ForeignKey("budget_periods.id"), nullable=False),
    Column("request_id", Text, ForeignKey("requests.id")),
    Column("kind", Text, nullable=False),
    Column("amount_delta", ExactDecimal, nullable=False),
    Column("reason", Text, nullable=False),
    Column("audit_event_id", Integer, ForeignKey("audit_events.id"), nullable=False),
    Column("occurred_at", Text, nullable=False),
    decimal_text("amount_delta"),
    CheckConstraint("kind IN ('grant', 'reduction', 'correction')"),
)
usage_transactions: Final = Table(
    "usage_transactions",
    metadata,
    Column("id", Text, primary_key=True, nullable=False),
    Column("reconciliation_id", Text, nullable=False, unique=True),
    Column("request_id", Text),
    Column("provider_id", Text, ForeignKey("providers.id"), nullable=False),
    Column("model_id", Text),
    Column("budget_period_id", Text),
    Column("currency", Text, nullable=False),
    Column("measured_cost", ExactDecimal),
    Column("estimated_cost", ExactDecimal),
    Column("virtual_charge", ExactDecimal),
    Column("input_tokens", Integer),
    Column("cached_input_tokens", Integer),
    Column("output_tokens", Integer),
    Column("scope", Text),
    Column("window_start", Text),
    Column("window_end", Text),
    Column("detail_archive_record_id", Text, ForeignKey("archive_references.record_id")),
    Column("occurred_at", Text, nullable=False),
    UniqueConstraint("id", "request_id"),
    ForeignKeyConstraint(["provider_id", "model_id"], ["models.provider_id", "models.id"]),
    ForeignKeyConstraint(
        ["request_id", "provider_id", "model_id"],
        ["requests.id", "requests.provider_id", "requests.model_id"],
    ),
    ForeignKeyConstraint(
        ["budget_period_id", "provider_id", "currency"],
        ["budget_periods.id", "budget_periods.provider_id", "budget_periods.currency"],
    ),
    CheckConstraint("request_id IS NULL OR model_id IS NOT NULL"),
    CheckConstraint("currency GLOB '[A-Z][A-Z][A-Z]'"),
    *(
        CheckConstraint(f"{name} IS NULL OR ({decimal_text(name).sqltext})")
        for name in ("measured_cost", "estimated_cost", "virtual_charge")
    ),
    *(
        CheckConstraint(f"{name} IS NULL OR (typeof({name}) = 'integer' AND {name} >= 0)")
        for name in ("input_tokens", "cached_input_tokens", "output_tokens")
    ),
)
reservation_events: Final = Table(
    "reservation_events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("request_id", Text, ForeignKey("request_reservations.request_id"), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("kind", Text, nullable=False),
    Column("settled_request_id", Text, unique=True),
    Column("usage_transaction_id", Text, unique=True),
    Column("occurred_at", Text, nullable=False),
    UniqueConstraint("request_id", "sequence"),
    ForeignKeyConstraint(
        ["usage_transaction_id", "request_id"],
        ["usage_transactions.id", "usage_transactions.request_id"],
    ),
    CheckConstraint("typeof(sequence) = 'integer' AND sequence >= 0"),
    CheckConstraint("kind IN ('reserved', 'dispatched', 'ambiguous', 'released', 'settled')"),
    CheckConstraint(
        "CASE WHEN kind = 'settled' THEN settled_request_id IS NOT NULL "
        "AND settled_request_id = request_id AND usage_transaction_id IS NOT NULL "
        "ELSE settled_request_id IS NULL AND usage_transaction_id IS NULL END"
    ),
)
