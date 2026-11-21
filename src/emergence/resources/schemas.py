"""Normalized reporting snapshots; null fields do not imply capacity."""

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime

from emergence.config import SafeModel


def utc_now() -> datetime:
    return datetime.now(UTC)


class ReportingIssue(StrEnum):
    UNSUPPORTED = "unsupported"
    MISSING_CREDENTIAL = "missing_reporting_credential"
    SHARED_CREDENTIAL = "shared_reporting_credential"
    MISSING_SCOPE = "missing_reporting_scope"
    EXPIRED_CREDENTIAL = "expired_reporting_credential"
    EXPIRING_CREDENTIAL = "expiring_reporting_credential"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    TOO_LARGE = "response_too_large"
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"
    INVALID_PAGINATION = "invalid_pagination"
    SCOPE_MISMATCH = "scope_mismatch"
    ARCHIVE_DETAIL_UNAVAILABLE = "archive_detail_unavailable"


class ReportingError(Exception):
    """A sanitized error code; never carries response, URL or credentials."""

    def __init__(self, issue: ReportingIssue) -> None:
        self.issue = issue
        super().__init__(issue.value)


class Window(SafeModel):
    start: AwareDatetime | None = None
    end: AwareDatetime | None = None


class ReportScope(SafeModel):
    organization: str
    project: str


class UsageBucket(SafeModel):
    window: Window
    model: str | None
    input_tokens: int
    output_tokens: int
    input_cached_tokens: int | None
    num_model_requests: int


class CostBucket(SafeModel):
    window: Window
    line_item: str | None
    amount: Decimal
    currency: Literal["USD"] = "USD"


class QuotaObservation(SafeModel):
    provider: str
    resource: str
    unit: str = "unknown"
    used: Decimal | None = None
    limit: Decimal | None = None
    remaining: Decimal | None = None
    usage_percent: Decimal | None = None
    window: Window | None = None
    resets_at: AwareDatetime | None = None
    observed_at: AwareDatetime
    source: str
    confidence: str | None = None
    status: Literal["verified", "observed", "experimental", "unavailable"]
    is_stale: bool = False
    raw_payload_ref: str | None = None
    issue: ReportingIssue | None = None
    scope: ReportScope | None = None
    usage: tuple[UsageBucket, ...] = ()
    costs: tuple[CostBucket, ...] = ()
