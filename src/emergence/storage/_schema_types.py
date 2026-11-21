"""Exact decimal serialization used by the initial operational schema."""

from decimal import Decimal

from sqlalchemy import CheckConstraint, Text
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.types import TypeDecorator


class NonFiniteDecimalError(ValueError):
    """Non-finite accounting or quota values cannot be persisted."""


class ExactDecimal(TypeDecorator[Decimal]):
    """Bind Decimal as plain decimal TEXT, preserving precision and scale."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Decimal | None, dialect: Dialect) -> str | None:
        if value is None:
            return None
        if not value.is_finite():
            raise NonFiniteDecimalError
        return format(value, "f")

    def process_result_value(self, value: str | None, dialect: Dialect) -> Decimal | None:
        return Decimal(value) if value is not None else None


def decimal_text(name: str) -> CheckConstraint:
    """Exclude malformed direct SQL values without numeric conversion."""
    magnitude = f"(CASE WHEN substr({name}, 1, 1) = '-' THEN substr({name}, 2) ELSE {name} END)"
    return CheckConstraint(
        f"typeof({name}) = 'text' AND {magnitude} GLOB '[0-9]*' "
        f"AND {magnitude} NOT GLOB '*[^0-9.]*' AND {magnitude} NOT GLOB '*.*.*' "
        f"AND {magnitude} NOT GLOB '*.'",
        name=f"valid_decimal_{name}",
    )


def hex_digest(name: str) -> CheckConstraint:
    return CheckConstraint(
        f"length({name}) = 64 AND {name} NOT GLOB '*[^0-9a-f]*'", name=f"valid_digest_{name}"
    )
