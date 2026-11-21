"""Frozen initial revision: operational identities, lifecycle and archives."""

from typing import Final

from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
)

from ._schema_types import ExactDecimal, decimal_text, hex_digest

metadata: Final = MetaData()

providers: Final = Table(
    "providers",
    metadata,
    Column("id", Text, primary_key=True, nullable=False),
    Column("kind", Text, nullable=False),
)
models: Final = Table(
    "models",
    metadata,
    Column("provider_id", Text, ForeignKey("providers.id"), primary_key=True, nullable=False),
    Column("id", Text, primary_key=True, nullable=False),
    Column("upstream_model_id", Text, nullable=False),
)
requests: Final = Table(
    "requests",
    metadata,
    Column("id", Text, primary_key=True, nullable=False),
    Column("idempotency_scope", Text, nullable=False),
    Column("idempotency_key", Text, nullable=False),
    Column("input_fingerprint", Text, nullable=False),
    Column("provider_id", Text, nullable=False),
    Column("model_id", Text, nullable=False),
    Column("agent_id", Text),
    Column("experiment_id", Text),
    Column("created_at", Text, nullable=False),
    ForeignKeyConstraint(["provider_id", "model_id"], ["models.provider_id", "models.id"]),
    UniqueConstraint("idempotency_scope", "idempotency_key"),
    UniqueConstraint("id", "provider_id", "model_id"),
    hex_digest("input_fingerprint"),
)
archive_references: Final = Table(
    "archive_references",
    metadata,
    Column("record_id", Text, primary_key=True, nullable=False),
    Column("checksum", Text, nullable=False),
    Column("size_bytes", Integer, nullable=False),
    CheckConstraint("typeof(size_bytes) = 'integer' AND size_bytes > 0"),
    hex_digest("record_id"),
    hex_digest("checksum"),
)
archive_outbox: Final = Table(
    "archive_outbox",
    metadata,
    Column(
        "record_id",
        Text,
        ForeignKey("archive_references.record_id"),
        primary_key=True,
        nullable=False,
    ),
    Column("state", Text, nullable=False, server_default="pending"),
    CheckConstraint("state IN ('pending', 'archived')"),
)
request_events: Final = Table(
    "request_events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("request_id", Text, ForeignKey("requests.id"), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("kind", Text, nullable=False),
    Column("occurred_at", Text, nullable=False),
    Column("archive_record_id", Text, ForeignKey("archive_references.record_id")),
    Column("outcome_code", Text),
    UniqueConstraint("request_id", "sequence"),
    CheckConstraint("typeof(sequence) = 'integer' AND sequence >= 0"),
)
quota_observations: Final = Table(
    "quota_observations",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("provider_id", Text, ForeignKey("providers.id"), nullable=False),
    Column("resource", Text, nullable=False),
    Column("unit", Text, nullable=False),
    Column("used", ExactDecimal),
    Column("quota_limit", ExactDecimal),
    Column("remaining", ExactDecimal),
    Column("usage_percent", ExactDecimal),
    Column("window_start", Text),
    Column("window_end", Text),
    Column("resets_at", Text),
    Column("observed_at", Text, nullable=False),
    Column("fresh_until", Text),
    Column("source", Text, nullable=False),
    Column("confidence", Text),
    Column("status", Text, nullable=False),
    Column("raw_archive_record_id", Text, ForeignKey("archive_references.record_id")),
    CheckConstraint("status IN ('verified', 'observed', 'experimental', 'unavailable')"),
    *(
        CheckConstraint(f"{name} IS NULL OR ({decimal_text(name).sqltext})")
        for name in ("used", "quota_limit", "remaining", "usage_percent")
    ),
)
audit_events: Final = Table(
    "audit_events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("occurred_at", Text, nullable=False),
    Column("actor_id", Text, nullable=False),
    Column("action", Text, nullable=False),
    Column("request_id", Text, ForeignKey("requests.id")),
    Column("provider_id", Text, ForeignKey("providers.id")),
    Column("detail_archive_record_id", Text, ForeignKey("archive_references.record_id")),
)
