"""Persist report projections, archive payloads and reconcile measured rollups."""

import errno
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from hashlib import sha256
from typing import Self
from uuid import uuid4

from pydantic import Field, JsonValue
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.config import SafeModel, Settings
from emergence.storage.archive import RecordKey, Spool, SpoolError
from emergence.storage.database import Database
from emergence.storage.models import archive_references, quota_observations, usage_transactions
from emergence.storage.repositories import ArchiveReference

from .reporting_parsers import WireValue
from .schemas import QuotaObservation, ReportingIssue, Window


class _ArchiveProjection(SafeModel):
    normalized: tuple[QuotaObservation, ...] = Field(min_length=1)
    raw_pages: list[JsonValue] = Field(exclude=True)

    @classmethod
    async def load(cls, spool: Spool, reference: ArchiveReference) -> Self | None:
        try:
            data = await spool.read(reference)
        except SpoolError as error:
            if error.code not in {
                "archive_mount_absent",
                "archive_mount_unknown",
                "archive_io_timeout",
            }:
                raise
            return None
        except OSError as error:
            # MountedArchive opens the immutable record by ID relative to its
            # directory. A missing record there is evidence loss, not a lost mount.
            if error.errno == errno.ENOENT and error.filename == reference.record_id:
                raise SpoolError("missing_record") from error
            if error.errno not in {
                errno.ENOENT,
                errno.ESTALE,
                errno.ENETDOWN,
                errno.ENETUNREACH,
                errno.EHOSTUNREACH,
                errno.ETIMEDOUT,
                errno.EIO,
                errno.ENOTCONN,
            }:
                raise
            return None
        return cls.model_validate_json(data.split(b"\n", 2)[2])


def _archive_value(value: WireValue) -> JsonValue:
    match value:
        case Decimal():
            # The existing JSON spool has no Decimal encoder. Preserve digits and exponent.
            return str(value)
        case dict():
            return {key: _archive_value(item) for key, item in value.items()}
        case list():
            return [_archive_value(item) for item in value]
        case _:
            return value


def _stamp(value: datetime | None) -> str | None:
    return value.isoformat(timespec="microseconds") if value is not None else None


async def _reconcile(
    session: AsyncSession,
    observation: QuotaObservation,
    pages: tuple[WireValue, ...],
    reference: ArchiveReference,
) -> None:
    scope = observation.scope
    if (
        scope is None
        or observation.source != "reporting:openai"
        or observation.resource != "costs"
        or observation.status != "verified"
    ):
        return
    dimensions: defaultdict[Window, dict[str, Decimal]] = defaultdict(dict)
    for bucket in observation.costs:
        identity = json.dumps(
            {
                "source": observation.source,
                "organization": scope.organization,
                "project": scope.project,
                "line_item": bucket.line_item,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        dimensions[bucket.window][identity] = bucket.amount
    # The paginated parser validated these pages before publication. Flattened
    # costs retain nonempty windows only; an explicit empty results list is
    # also a revision, but a day missing from the response is not.
    for page in pages:
        if isinstance(page, dict) and isinstance(data := page.get("data"), list):
            for item in data:
                match item:
                    case {
                        "object": "bucket",
                        "results": [],
                        "start_time": int(start),
                        "end_time": int(end),
                    }:
                        dimensions.setdefault(
                            Window(
                                start=datetime.fromtimestamp(start, UTC),
                                end=datetime.fromtimestamp(end, UTC),
                            ),
                            {},
                        )
                    case _:
                        continue

    ledger = usage_transactions.c
    for window, deltas in dimensions.items():
        first, last = _stamp(window.start), _stamp(window.end)
        previous = await session.execute(
            select(ledger.scope, ledger.measured_cost).where(
                ledger.provider_id == observation.provider,
                ledger.currency == "USD",
                ledger.window_start == first,
                ledger.window_end == last,
                ledger.request_id.is_(None),
                ledger.measured_cost.is_not(None),
                func.json_extract(ledger.scope, "$.source") == observation.source,
                func.json_extract(ledger.scope, "$.organization") == scope.organization,
                func.json_extract(ledger.scope, "$.project") == scope.project,
            )
        )
        # SQLite SUM would convert these exact amounts into binary floats.
        with localcontext(prec=2048):
            for identity, amount in previous:
                deltas[identity] = deltas.get(identity, Decimal(0)) - amount
        for identity, delta in deltas.items():
            if not delta:
                continue
            reconciliation = sha256(
                json.dumps([reference.record_id, identity, first, last, "USD"]).encode()
            ).hexdigest()
            await session.execute(
                usage_transactions.insert().values(
                    id=reconciliation,
                    reconciliation_id=reconciliation,
                    provider_id=observation.provider,
                    currency="USD",
                    measured_cost=delta,
                    scope=identity,
                    window_start=first,
                    window_end=last,
                    detail_archive_record_id=reference.record_id,
                    occurred_at=_stamp(observation.observed_at),
                )
            )


class ReportingStore:
    def __init__(self, database: Database, spool: Spool) -> None:
        self.database = database
        self.spool = spool

    async def append(
        self,
        observations: tuple[QuotaObservation, ...],
        stale_after: int,
        pages: tuple[WireValue, ...] = (),
    ) -> tuple[QuotaObservation, ...]:
        envelope: JsonValue = {
            "normalized": [obs.model_dump(mode="json") for obs in observations],
            "raw_pages": [_archive_value(page) for page in pages],
        }
        reference = (
            await self.spool.publish(RecordKey(uuid4().hex, "reporting"), envelope)
            if pages or any(obs.issue is not None for obs in observations)
            else None
        )
        async with self.database.write() as session:
            for observation in observations:
                window = observation.window or Window()
                await session.execute(
                    quota_observations.insert().values(
                        provider_id=observation.provider,
                        resource=observation.resource,
                        unit=observation.unit,
                        used=observation.used,
                        quota_limit=observation.limit,
                        remaining=observation.remaining,
                        usage_percent=observation.usage_percent,
                        window_start=_stamp(window.start),
                        window_end=_stamp(window.end),
                        resets_at=_stamp(observation.resets_at),
                        observed_at=_stamp(observation.observed_at),
                        fresh_until=_stamp(
                            observation.observed_at + timedelta(seconds=stale_after)
                        ),
                        source=observation.source,
                        confidence=observation.confidence,
                        status=observation.status,
                        raw_archive_record_id=reference.record_id if reference else None,
                    )
                )
                if reference is not None:
                    await _reconcile(session, observation, pages, reference)
        return tuple(
            obs.model_copy(
                update={"raw_payload_ref": reference.record_id if pages and reference else None}
            )
            for obs in observations
        )

    async def latest(self, settings: Settings, now: datetime) -> tuple[QuotaObservation, ...]:
        quota = quota_observations.c
        archive = archive_references.c
        enabled = tuple(name for name, provider in settings.providers.items() if provider.enabled)
        ranked = (
            select(
                quota_observations,
                func.row_number()
                .over(
                    partition_by=(
                        quota.provider_id,
                        quota.source,
                        quota.resource,
                        quota.status == "unavailable",
                    ),
                    order_by=(quota.observed_at.desc(), quota.id.desc()),
                )
                .label("rank"),
            )
            .where(quota.provider_id.in_(enabled))
            .subquery()
        )
        query = (
            select(ranked, archive.checksum, archive.size_bytes)
            .outerjoin(archive_references, ranked.c.raw_archive_record_id == archive.record_id)
            .where(ranked.c.rank == 1)
            .order_by(ranked.c.observed_at.desc(), ranked.c.id.desc())
        )
        async with self.database.session() as session:
            rows = (await session.execute(query)).all()

        results: list[QuotaObservation] = []
        successful: set[tuple[str, str, str]] = set()
        envelopes: dict[str, _ArchiveProjection | None] = {}
        for row in rows:
            # Z.ai's unstable categories share one endpoint health state.
            key = (
                row.provider_id,
                row.source,
                "quota" if row.source == "reporting:zai" else row.resource,
            )
            if row.status == "unavailable" and key in successful:
                continue
            observed_at = datetime.fromisoformat(row.observed_at)
            fresh_until = datetime.fromisoformat(row.fresh_until) if row.fresh_until else None
            stale = fresh_until is None or not observed_at <= now < fresh_until
            record_id: str | None = row.raw_archive_record_id
            archived_detail = row.source.startswith("reporting:") and record_id is not None
            envelope = None
            if archived_detail and record_id is not None:
                if record_id not in envelopes:
                    reference = ArchiveReference(record_id, row.checksum, row.size_bytes)
                    envelopes[record_id] = await _ArchiveProjection.load(self.spool, reference)
                envelope = envelopes[record_id]
            if envelope is not None:
                matches = [obs for obs in envelope.normalized if obs.resource == row.resource]
                if len(matches) != 1:
                    raise SpoolError("report_projection_mismatch")
                observation = matches[0].model_copy(
                    update={"raw_payload_ref": record_id if envelope.raw_pages else None}
                )
                intended = settings.providers[observation.provider]
                if observation.scope is not None and (
                    observation.scope.organization != intended.organization
                    or observation.scope.project != intended.project
                ):
                    continue
            else:
                observation = QuotaObservation(
                    provider=row.provider_id,
                    resource=row.resource,
                    unit=row.unit,
                    used=row.used,
                    limit=row.quota_limit,
                    remaining=row.remaining,
                    usage_percent=row.usage_percent,
                    window=Window(start=row.window_start, end=row.window_end)
                    if row.window_start or row.window_end
                    else None,
                    resets_at=row.resets_at,
                    observed_at=observed_at,
                    source=row.source,
                    confidence=row.confidence,
                    status="unavailable" if archived_detail else row.status,
                    raw_payload_ref=record_id,
                    issue=ReportingIssue.ARCHIVE_DETAIL_UNAVAILABLE if archived_detail else None,
                )
            if observation.issue is ReportingIssue.UNSUPPORTED:
                stale = False
            if observation.status != "unavailable":
                successful.add(key)
            results.append(observation.model_copy(update={"is_stale": stale}))
        return tuple(results)
