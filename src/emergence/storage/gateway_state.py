"""Revision 0002 additions; the initial schema snapshot remains unchanged."""

from typing import Final

from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
)

from .models import providers, requests

metadata: Final = MetaData()
SCHEMA_HEAD: Final = "0002_gateway_state"

request_capacity: Final = Table(
    "request_capacity",
    metadata,
    Column("request_id", Text, ForeignKey(requests.c.id), primary_key=True, nullable=False),
    Column("allocated_bytes", Integer, nullable=False),
    Column("input_record_id", Text, nullable=False, unique=True),
    Column("output_record_id", Text, nullable=False, unique=True),
    Column("input_tokens_bound", Integer, nullable=False),
    Column("output_tokens_bound", Integer, nullable=False),
    Column("created_at", Text, nullable=False),
    CheckConstraint("typeof(allocated_bytes) = 'integer' AND allocated_bytes > 0"),
    CheckConstraint("typeof(input_tokens_bound) = 'integer' AND input_tokens_bound >= 0"),
    CheckConstraint("typeof(output_tokens_bound) = 'integer' AND output_tokens_bound > 0"),
)

provider_accounts: Final = Table(
    "provider_accounts",
    metadata,
    Column("provider_id", Text, ForeignKey(providers.c.id), primary_key=True, nullable=False),
    Column("organization", Text),
    Column("project", Text),
    UniqueConstraint("organization", "project"),
)

AUTHORITY_TABLES: Final = (request_capacity, provider_accounts)
