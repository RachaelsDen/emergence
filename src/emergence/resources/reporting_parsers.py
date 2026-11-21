"""Only normalize fields whose meaning is supported by the available evidence."""

import json
import re
from decimal import Decimal, DecimalException
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_validator

from .schemas import ReportingError, ReportingIssue

type WireValue = str | int | Decimal | bool | None | list[WireValue] | dict[str, WireValue]
type WireObject = dict[str, WireValue]
type Counter = Annotated[StrictInt, Field(ge=0)]
_CATEGORY: Final = re.compile(r"[A-Z][A-Z0-9_]{0,63}")


def _invalid_constant(_: str) -> None:
    raise ReportingError(ReportingIssue.INVALID_JSON)


def _unique_fields(pairs: list[tuple[str, WireValue]]) -> WireObject:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ReportingError(ReportingIssue.INVALID_JSON)
    return fields


def decode_body(body: bytes) -> WireObject:
    try:
        value: WireValue = json.loads(
            body,
            parse_float=Decimal,
            parse_constant=_invalid_constant,
            object_pairs_hook=_unique_fields,
        )
    except (ValueError, DecimalException, RecursionError):
        raise ReportingError(ReportingIssue.INVALID_JSON) from None
    if not isinstance(value, dict):
        raise ReportingError(ReportingIssue.INVALID_SCHEMA)
    return value


class WireModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, hide_input_in_errors=True)


class _ZaiLimit(WireModel):
    type: str
    percentage: Decimal | None = None

    @field_validator("percentage", mode="before")
    @classmethod
    def safe_percentage(cls, value: WireValue) -> Decimal | None:
        match value:
            case bool():
                return None
            case int() | Decimal() if 0 <= value <= 100:
                exact = Decimal(value)
                exponent = exact.as_tuple().exponent
                return exact if isinstance(exponent, int) and abs(exponent) <= 1000 else None
            case _:
                return None


def parse_zai(body: WireObject) -> tuple[tuple[str, Decimal | None], ...]:
    """Retain category names; only TOKENS_LIMIT has documented project evidence of a percent."""
    data = body.get("data")
    limits = data.get("limits") if isinstance(data, dict) else None
    if not isinstance(limits, list):
        raise ReportingError(ReportingIssue.INVALID_SCHEMA)
    results: dict[str, Decimal | None] = {}
    ambiguous: set[str] = set()
    for entry in limits:
        try:
            row = _ZaiLimit.model_validate(entry)
        except ValidationError:
            continue
        if not _CATEGORY.fullmatch(row.type):
            continue
        if row.type in results:
            ambiguous.add(row.type)
        results[row.type] = row.percentage if row.type == "TOKENS_LIMIT" else None
    if not results:
        raise ReportingError(ReportingIssue.INVALID_SCHEMA)
    return tuple((name, None if name in ambiguous else value) for name, value in results.items())


def parse_minimax(body: WireObject) -> None:
    """Archive object-shaped experimental data; no observed schema supports a counter yet."""
    if not body:
        raise ReportingError(ReportingIssue.INVALID_SCHEMA)
