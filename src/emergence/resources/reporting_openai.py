"""OpenAI Admin usage and cost reports; a projection is published only after the last page."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Final, Literal

from anyio import fail_after
from pydantic import ValidationError, field_validator

from .reporting_http import ReportingHTTP
from .reporting_parsers import Counter, WireModel, WireObject, WireValue
from .schemas import CostBucket, ReportingError, ReportingIssue, ReportScope, UsageBucket, Window

_MAX_PAGES: Final = 400
_DAY: Final = 86400
type ReportKind = Literal["usage", "costs"]


class _Bucket(WireModel):
    object: Literal["bucket"]
    start_time: Counter
    end_time: Counter
    results: list[WireObject]


class _Page(WireModel):
    object: Literal["page"]
    data: list[_Bucket]
    has_more: bool
    next_page: str | None


class _UsageRow(WireModel):
    object: Literal["organization.usage.completions.result"]
    project_id: str
    model: str | None
    input_tokens: Counter
    output_tokens: Counter
    input_cached_tokens: Counter | None = None
    num_model_requests: Counter


class _Amount(WireModel):
    currency: Literal["usd"]
    value: Decimal

    @field_validator("value", mode="before")
    @classmethod
    def exact_value(cls, value: WireValue) -> Decimal:
        match value:
            case bool():
                raise ValueError("expected a monetary JSON number")
            case int() | Decimal():
                exact = Decimal(value)
                exponent = exact.as_tuple().exponent
                if (
                    exact.is_finite()
                    and isinstance(exponent, int)
                    and -1000 <= exponent <= 1000
                    and exact.adjusted() <= 1000
                ):
                    return exact
                raise ValueError("monetary exponent outside supported range")
            case _:
                raise ValueError("expected a monetary JSON number")


class _CostRow(WireModel):
    object: Literal["organization.costs.result"]
    project_id: str
    line_item: str | None
    amount: _Amount


@dataclass(frozen=True, slots=True)
class AdminReport:
    scope: ReportScope
    window: Window
    pages: tuple[WireObject, ...]
    usage: tuple[UsageBucket, ...] = ()
    costs: tuple[CostBucket, ...] = ()


@dataclass(frozen=True, slots=True)
class OpenAIAdmin:
    http: ReportingHTTP
    base_url: str
    scope: ReportScope
    credential: str
    lookback_days: int
    safe_text: Callable[[str], str]

    async def fetch(self, kind: ReportKind, now: datetime) -> AdminReport:
        try:
            with fail_after(self.http.timeout):
                return await self._fetch(kind, now)
        except TimeoutError:
            raise ReportingError(ReportingIssue.TIMEOUT) from None
        except ValidationError:
            raise ReportingError(ReportingIssue.INVALID_SCHEMA) from None

    async def _fetch(self, kind: ReportKind, now: datetime) -> AdminReport:
        end = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=self.lookback_days)
        start_second, end_second = int(start.timestamp()), int(end.timestamp())
        route, dimension = (
            ("usage/completions", "model") if kind == "usage" else ("costs", "line_item")
        )
        url = f"{self.base_url.rstrip('/')}/organization/{route}"
        headers = {
            "Authorization": f"Bearer {self.credential}",
            "OpenAI-Organization": self.scope.organization,
        }
        query = [
            ("start_time", str(start_second)),
            ("end_time", str(end_second)),
            ("bucket_width", "1d"),
            ("limit", "31"),
            ("project_ids", self.scope.project),
            ("group_by", "project_id"),
            ("group_by", dimension),
        ]
        pages: list[WireObject] = []
        cursors: set[str] = set()
        windows: set[int] = set()
        usage: list[UsageBucket] = []
        costs: list[CostBucket] = []
        cursor: str | None = None
        remaining = self.http.limit
        for _ in range(_MAX_PAGES):
            raw, size = await self.http.get(
                url,
                headers=headers,
                params=query if cursor is None else [*query, ("page", cursor)],
                remaining=remaining,
            )
            remaining -= size
            if not isinstance(raw.get("has_more"), bool) or "next_page" not in raw:
                raise ReportingError(ReportingIssue.INVALID_PAGINATION)
            page = _Page.model_validate(raw)
            pages.append(raw)
            for bucket in page.data:
                if (
                    not start_second <= bucket.start_time < bucket.end_time <= end_second
                    or bucket.start_time % _DAY
                    or bucket.end_time - bucket.start_time != _DAY
                    or bucket.start_time in windows
                ):
                    raise ReportingError(ReportingIssue.INVALID_SCHEMA)
                windows.add(bucket.start_time)
                self._project(kind, bucket, usage, costs)
            cursor = page.next_page
            if not page.has_more:
                if cursor is not None:
                    raise ReportingError(ReportingIssue.INVALID_PAGINATION)
                return AdminReport(
                    self.scope,
                    Window(start=start, end=end),
                    tuple(pages),
                    tuple(usage),
                    tuple(costs),
                )
            if not cursor or not cursor.strip() or cursor in cursors or not page.data:
                raise ReportingError(ReportingIssue.INVALID_PAGINATION)
            cursors.add(cursor)
        raise ReportingError(ReportingIssue.INVALID_PAGINATION)

    def _project(
        self, kind: ReportKind, bucket: _Bucket, usage: list[UsageBucket], costs: list[CostBucket]
    ) -> None:
        window = Window(
            start=datetime.fromtimestamp(bucket.start_time, UTC),
            end=datetime.fromtimestamp(bucket.end_time, UTC),
        )
        labels: set[str | None] = set()
        for result in bucket.results:
            if result.get("project_id") != self.scope.project:
                raise ReportingError(ReportingIssue.SCOPE_MISMATCH)
            match kind:
                case "usage":
                    row = _UsageRow.model_validate(result)
                    label = row.model
                    usage.append(
                        UsageBucket(
                            window=window,
                            model=label,
                            input_tokens=row.input_tokens,
                            output_tokens=row.output_tokens,
                            input_cached_tokens=row.input_cached_tokens,
                            num_model_requests=row.num_model_requests,
                        )
                    )
                case "costs":
                    cost = _CostRow.model_validate(result)
                    label = cost.line_item
                    costs.append(
                        CostBucket(window=window, line_item=label, amount=cost.amount.value)
                    )
            if label in labels or (label is not None and self.safe_text(label) != label):
                raise ReportingError(ReportingIssue.INVALID_SCHEMA)
            labels.add(label)
