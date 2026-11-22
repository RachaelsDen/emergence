"""Parse verified archives, recognizing only historical normalized count overflow."""

from typing import Annotated, Final

from pydantic import AwareDatetime, Field, JsonValue, ValidationError

from emergence.config import SafeModel
from emergence.inference.errors import GatewayError
from emergence.inference.request_state import ArchivedOutcome, FailureRecord
from emergence.inference.schemas import TokenUsage

_MAX_COUNT: Final = 2**63 - 1
_USAGE_FIELDS: Final = frozenset(
    {"input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"}
)
_LegacyCount = Annotated[int, Field(strict=True, ge=0)]


class _LegacyUsage(TokenUsage):
    """Retain the original count types and inherited overlap validation."""

    input_tokens: _LegacyCount | None = None
    cached_input_tokens: _LegacyCount | None = None
    output_tokens: _LegacyCount | None = None
    reasoning_tokens: _LegacyCount | None = None


class _LegacyResult(SafeModel):
    """The historical result envelope, before normalized count bounds."""

    provider: str
    model: str
    upstream_model: str
    upstream_request_id: str | None
    content: str
    finish_reason: str | None
    usage: _LegacyUsage
    raw_usage: JsonValue
    raw_response: JsonValue


class _LegacyOutcome(SafeModel):
    request_id: str
    observed_at: AwareDatetime
    result: _LegacyResult | None = None
    failure: FailureRecord | None = None


def parse_archived_outcome(payload: bytes) -> ArchivedOutcome:
    """Keep corrupt archives fatal; turn historical overflow into uncertain usage."""
    try:
        return ArchivedOutcome.model_validate_json(payload)
    except ValidationError as error:
        for detail in error.errors():
            match detail:
                case {
                    "type": "less_than_equal",
                    "loc": ("result", "usage", field),
                    "input": int(value),
                } if field in _USAGE_FIELDS and type(value) is int and value > _MAX_COUNT:
                    continue
                case _:
                    raise GatewayError("request_archive_invalid") from None
    try:
        legacy = _LegacyOutcome.model_validate_json(payload)
    except ValidationError:
        raise GatewayError("request_archive_invalid") from None
    return ArchivedOutcome(
        request_id=legacy.request_id,
        observed_at=legacy.observed_at,
        failure=FailureRecord(
            code="provider_schema_error",
            ambiguous=True,
            raw_response=payload.decode("utf-8"),
        ),
    )
