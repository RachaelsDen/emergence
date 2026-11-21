"""Prometheus text exposition over durable state and bounded process timing."""

from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import func, select

from emergence.config import SafeModel
from emergence.inference.gateway import Gateway
from emergence.resources.quota_polling import QuotaMonitor
from emergence.storage.capacity import live_claims
from emergence.storage.models import archive_outbox, requests, usage_transactions


class StorageStatus(SafeModel):
    used_bytes: int
    limit_bytes: int
    free_bytes: int
    headroom_bytes: int
    reserved_bytes: int
    pending_records: int
    archived_records: int


async def storage_status(gateway: Gateway) -> StorageStatus:
    stats = await gateway.spool.stats()
    async with gateway.database.session() as session:
        claims = await live_claims(session)
        pending = await session.scalar(
            select(func.count())
            .select_from(archive_outbox)
            .where(archive_outbox.c.state == "pending")
        )
        archived = await session.scalar(
            select(func.count())
            .select_from(archive_outbox)
            .where(archive_outbox.c.state == "archived")
        )
    return StorageStatus(
        used_bytes=stats.used_bytes,
        limit_bytes=gateway.settings.spool_limit_bytes,
        free_bytes=stats.free_bytes,
        headroom_bytes=gateway.settings.disk_headroom_bytes,
        reserved_bytes=sum(claim.remaining_bytes for claim in claims),
        pending_records=pending if pending is not None else 0,
        archived_records=archived if archived is not None else 0,
    )


@dataclass(frozen=True, slots=True)
class Timing:
    count: int
    seconds: float


class InferenceMetrics:
    """Accumulate process durations only for configured provider/model labels."""

    def __init__(self) -> None:
        self.timings: dict[tuple[str, str], Timing] = {}

    def observe(self, provider: str, model: str, seconds: float) -> None:
        previous = self.timings.get((provider, model), Timing(0, 0.0))
        self.timings[provider, model] = Timing(previous.count + 1, previous.seconds + seconds)


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _sample(name: str, value: int | float | Decimal, **labels: str) -> str:
    dimensions = ",".join(f'{key}="{_label(value)}"' for key, value in labels.items())
    return f"{name}{{{dimensions}}} {value}"


async def render_metrics(gateway: Gateway, monitor: QuotaMonitor, timing: InferenceMetrics) -> str:
    storage = await storage_status(gateway)
    lines = [
        "# TYPE emergence_requests_total counter",
        "# TYPE emergence_observed_tokens_total counter",
        "# TYPE emergence_inference_duration_seconds summary",
        "# TYPE emergence_budget_available_usd gauge",
        "# TYPE emergence_reserved_usd gauge",
        "# TYPE emergence_safety_exposure_usd gauge",
        "# TYPE emergence_reporting_stale gauge",
        "# TYPE emergence_reporting_unavailable gauge",
        "# TYPE emergence_spool_used_bytes gauge",
        "# TYPE emergence_spool_reserved_bytes gauge",
        "# TYPE emergence_spool_limit_bytes gauge",
        "# TYPE emergence_spool_free_bytes gauge",
        "# TYPE emergence_archive_pending_records gauge",
        "# TYPE emergence_archive_delivered_records gauge",
        _sample("emergence_spool_used_bytes", storage.used_bytes),
        _sample("emergence_spool_reserved_bytes", storage.reserved_bytes),
        _sample("emergence_spool_limit_bytes", storage.limit_bytes),
        _sample("emergence_spool_free_bytes", storage.free_bytes),
        _sample("emergence_archive_pending_records", storage.pending_records),
        _sample("emergence_archive_delivered_records", storage.archived_records),
    ]
    async with gateway.database.session() as session:
        counts = await session.execute(
            select(
                requests.c.provider_id, requests.c.model_id, func.count().label("total")
            ).group_by(requests.c.provider_id, requests.c.model_id)
        )
        for row in counts:
            lines.append(
                _sample(
                    "emergence_requests_total",
                    row.total,
                    provider=row.provider_id,
                    model=row.model_id,
                )
            )
        tokens = await session.execute(
            select(
                usage_transactions.c.provider_id,
                usage_transactions.c.model_id,
                usage_transactions.c.request_id.is_not(None).label("request_usage"),
                func.sum(usage_transactions.c.input_tokens).label("input"),
                func.sum(usage_transactions.c.cached_input_tokens).label("cached"),
                func.sum(usage_transactions.c.output_tokens).label("output"),
            ).group_by(
                usage_transactions.c.provider_id,
                usage_transactions.c.model_id,
                usage_transactions.c.request_id.is_not(None),
            )
        )
        for row in tokens:
            for dimension, count in (
                ("input", row.input),
                ("cached_input", row.cached),
                ("output", row.output),
            ):
                if count is not None:
                    lines.append(
                        _sample(
                            "emergence_observed_tokens_total",
                            count,
                            provider=row.provider_id,
                            model=row.model_id or "unattributed",
                            source="request" if row.request_usage else "provider_report",
                            dimension=dimension,
                        )
                    )
    for (provider, model), observation in timing.timings.items():
        lines.extend(
            (
                _sample(
                    "emergence_inference_duration_seconds_count",
                    observation.count,
                    provider=provider,
                    model=model,
                ),
                _sample(
                    "emergence_inference_duration_seconds_sum",
                    observation.seconds,
                    provider=provider,
                    model=model,
                ),
            )
        )
    for name, config in gateway.settings.providers.items():
        if not config.enabled:
            continue
        balance = await gateway.ledger.balance(name)
        for metric, amount in (
            ("emergence_budget_available_usd", balance.available),
            ("emergence_reserved_usd", balance.reserved),
            ("emergence_safety_exposure_usd", balance.safety_exposure),
        ):
            if amount is not None:
                lines.append(_sample(metric, amount, provider=name))
    observations = await monitor.latest()
    for provider in gateway.settings.providers:
        relevant = tuple(item for item in observations if item.provider == provider)
        if relevant:
            lines.append(
                _sample(
                    "emergence_reporting_stale",
                    int(any(item.is_stale for item in relevant)),
                    provider=provider,
                )
            )
            lines.append(
                _sample(
                    "emergence_reporting_unavailable",
                    int(any(item.status == "unavailable" for item in relevant)),
                    provider=provider,
                )
            )
    return "\n".join(lines) + "\n"
