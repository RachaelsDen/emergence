"""Authenticated operator API over the existing gateway and accounting services."""

from collections import deque
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from time import monotonic
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import Response
from pydantic import AwareDatetime, BeforeValidator, Field, JsonValue, TypeAdapter
from pydantic_core import PydanticCustomError
from sqlalchemy import select

from emergence.config import ProviderKind, QuotaPolicy, SafeModel, Settings
from emergence.inference.gateway import GatewayResult
from emergence.inference.schemas import InferenceRequest
from emergence.providers.base import FeatureBound
from emergence.resources.ledger import BudgetBalance
from emergence.resources.pricing import stamp
from emergence.resources.schemas import QuotaObservation
from emergence.storage.archive import RecordKey
from emergence.storage.delivery import ArchiveStatus
from emergence.storage.models import (
    archive_references,
    request_events,
    requests,
    usage_transactions,
)
from emergence.storage.repositories import ArchiveReference
from emergence.telemetry.metrics import (
    InferenceMetrics,
    StorageStatus,
    render_metrics,
    storage_status,
)

from .dependencies import ERROR_DESCRIPTIONS, Authorization, ErrorResponse, Identity, Services


def _money(value: JsonValue) -> Decimal:
    match value:
        case str() as text:
            try:
                return Decimal(text)
            except InvalidOperation:
                raise PydanticCustomError("money", "amount must be a decimal string") from None
        case _:
            raise PydanticCustomError("money", "amount must be a decimal string")


type Money = Annotated[
    Decimal,
    BeforeValidator(_money, json_schema_input_type=str),
    Field(allow_inf_nan=False, max_digits=64, decimal_places=24),
]
type Reason = Annotated[str, Field(min_length=1, max_length=2000)]


class Adjustment(SafeModel):
    provider: str
    amount: Money
    kind: Literal["grant", "reduction", "correction"]
    reason: Reason


class Resolution(SafeModel):
    amount: Money | None
    reason: Reason


class ProviderStatus(SafeModel):
    provider: str
    kind: ProviderKind
    enabled: bool
    quota_policy: QuotaPolicy
    inference_credential_configured: bool
    reporting_credential_configured: bool
    reporting_key_expires_at: AwareDatetime | None


class Providers(SafeModel):
    providers: tuple[ProviderStatus, ...]


class ModelStatus(SafeModel):
    provider: str
    alias: str
    upstream_model_id: str
    max_input_tokens: int
    max_output_tokens: int
    features: tuple[FeatureBound, ...]


class Models(SafeModel):
    models: tuple[ModelStatus, ...]


class Resources(SafeModel):
    observations: tuple[QuotaObservation, ...]
    storage: StorageStatus
    archive: ArchiveStatus


class Budgets(SafeModel):
    budgets: tuple[BudgetBalance, ...]


class UsageRecord(SafeModel):
    id: str
    reconciliation_id: str
    request_id: str | None
    provider_id: str
    model_id: str | None
    budget_period_id: str | None
    currency: str
    measured_cost: Decimal | None
    estimated_cost: Decimal | None
    virtual_charge: Decimal | None
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    scope: str | None
    window_start: AwareDatetime | None
    window_end: AwareDatetime | None
    detail_archive_record_id: str | None
    occurred_at: AwareDatetime


class Usage(SafeModel):
    items: tuple[UsageRecord, ...]
    limit: int
    offset: int
    has_more: bool


class HistoryRecord(SafeModel):
    kind: Literal["input", "output"]
    record_id: str
    checksum: str
    payload: JsonValue


class HistoryEvent(SafeModel):
    id: int
    request_id: str
    sequence: int
    kind: str
    occurred_at: AwareDatetime
    archive_record_id: str | None
    outcome_code: str | None


class History(SafeModel):
    request_id: str
    records: tuple[HistoryRecord, ...]
    events: tuple[HistoryEvent, ...]


def install_routes(app: FastAPI, settings: Settings, services: Callable[[], Services]) -> None:
    auth = Authorization(settings)
    router = APIRouter(
        prefix="/api/v1",
        responses={
            code: {"model": ErrorResponse, "description": description}
            for code, description in ERROR_DESCRIPTIONS.items()
        },
    )
    inference_times: deque[float] = deque()
    timing = InferenceMetrics()

    @router.get("/providers")
    async def providers(
        _identity: Annotated[Identity, Depends(auth.reader)],
    ) -> Providers:
        services()
        return Providers(
            providers=tuple(
                ProviderStatus(
                    provider=name,
                    kind=config.kind,
                    enabled=config.enabled,
                    quota_policy=config.quota_policy,
                    inference_credential_configured=(
                        settings.secret(config.credential_env) is not None
                    ),
                    reporting_credential_configured=(
                        settings.secret(
                            config.reporting_credential_env
                            or (
                                config.credential_env
                                if config.kind is not ProviderKind.OPENAI
                                else None
                            )
                        )
                        is not None
                    ),
                    reporting_key_expires_at=config.reporting_key_expires_at,
                )
                for name, config in settings.providers.items()
            )
        )

    @router.get("/models")
    async def models(_identity: Annotated[Identity, Depends(auth.reader)]) -> Models:
        gateway = services().gateway
        return Models(
            models=tuple(
                ModelStatus(
                    provider=name,
                    alias=alias,
                    upstream_model_id=model.id,
                    max_input_tokens=model.max_input_tokens,
                    max_output_tokens=model.max_output_tokens,
                    features=gateway.registry.get(name).features(model),
                )
                for name, config in settings.providers.items()
                if config.enabled
                for alias, model in config.models.items()
            )
        )

    @router.get("/resources")
    async def resources(_identity: Annotated[Identity, Depends(auth.reader)]) -> Resources:
        current = services()
        return Resources(
            observations=await current.monitor.latest(),
            storage=await storage_status(current.gateway),
            archive=await current.archive.status(),
        )

    @router.get("/budgets")
    async def budgets(
        _identity: Annotated[Identity, Depends(auth.reader)], provider: str | None = None
    ) -> Budgets:
        gateway = services().gateway
        names = (
            (provider,)
            if provider is not None
            else tuple(name for name, config in settings.providers.items() if config.enabled)
        )
        if any(name not in settings.providers for name in names):
            raise HTTPException(404, "provider_not_found")
        return Budgets(budgets=tuple([await gateway.ledger.balance(name) for name in names]))

    @router.post("/budgets/adjustments")
    async def adjust(
        body: Adjustment, identity: Annotated[Identity, Depends(auth.operator)]
    ) -> BudgetBalance:
        if body.provider not in settings.providers:
            raise HTTPException(404, "provider_not_found")
        return await services().gateway.ledger.adjust(
            body.provider, body.amount, body.kind, body.reason, identity.actor
        )

    @router.get("/usage")
    async def usage(
        _identity: Annotated[Identity, Depends(auth.operator)],
        provider: str | None = None,
        model: str | None = None,
        start: AwareDatetime | None = None,
        end: AwareDatetime | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0, le=1000000)] = 0,
    ) -> Usage:
        if start is not None and end is not None and start >= end:
            raise HTTPException(400, "invalid_time_range")
        query = select(usage_transactions)
        if provider is not None:
            query = query.where(usage_transactions.c.provider_id == provider)
        if model is not None:
            query = query.where(usage_transactions.c.model_id == model)
        if start is not None:
            query = query.where(usage_transactions.c.occurred_at >= stamp(start))
        if end is not None:
            query = query.where(usage_transactions.c.occurred_at < stamp(end))
        query = (
            query.order_by(usage_transactions.c.occurred_at.desc(), usage_transactions.c.id.desc())
            .limit(limit + 1)
            .offset(offset)
        )
        async with services().gateway.database.session() as session:
            rows = (await session.execute(query)).mappings().all()
        return Usage(
            items=tuple(UsageRecord.model_validate(row) for row in rows[:limit]),
            limit=limit,
            offset=offset,
            has_more=len(rows) > limit,
        )

    @router.post("/inference")
    async def infer(
        body: InferenceRequest,
        identity: Annotated[Identity, Depends(auth.operator)],
        idempotency_key: Annotated[str, Header(min_length=1, max_length=256)],
    ) -> GatewayResult:
        now = monotonic()
        while inference_times and inference_times[0] <= now - 60:
            inference_times.popleft()
        if len(inference_times) >= settings.requests_per_minute:
            raise HTTPException(429, "inference_rate_limited", headers={"Retry-After": "60"})
        inference_times.append(now)
        started = monotonic()
        try:
            return await services().gateway.infer(body, idempotency_key, identity.actor)
        finally:
            if settings.metrics_enabled:
                config = settings.providers.get(body.provider)
                provider = body.provider if config is not None else "unknown"
                model = (
                    body.model if config is not None and body.model in config.models else "unknown"
                )
                timing.observe(provider, model, monotonic() - started)

    @router.get("/inference/{request_id}")
    async def inference_status(
        request_id: str, _identity: Annotated[Identity, Depends(auth.operator)]
    ) -> GatewayResult:
        return await services().gateway.get(request_id)

    @router.post("/inference/{request_id}/resolution")
    async def resolve(
        request_id: str,
        body: Resolution,
        identity: Annotated[Identity, Depends(auth.operator)],
    ) -> GatewayResult:
        return await services().gateway.resolve(
            request_id, body.amount, body.reason, identity.actor
        )

    @router.get("/history/{request_id}")
    async def history(
        request_id: str, _identity: Annotated[Identity, Depends(auth.operator)]
    ) -> History:
        gateway = services().gateway
        records: list[HistoryRecord] = []
        async with gateway.database.session() as session:
            exists = await session.scalar(select(requests.c.id).where(requests.c.id == request_id))
            if exists is None:
                raise HTTPException(404, "request_not_found")
            events = tuple(
                HistoryEvent.model_validate(row)
                for row in (
                    await session.execute(
                        select(request_events)
                        .where(request_events.c.request_id == request_id)
                        .order_by(request_events.c.sequence)
                    )
                ).mappings()
            )
            for kind in ("input", "output"):
                record_id = RecordKey(request_id, kind).record_id
                row = (
                    await session.execute(
                        select(archive_references).where(
                            archive_references.c.record_id == record_id
                        )
                    )
                ).one_or_none()
                if row is not None:
                    reference = ArchiveReference(row.record_id, row.checksum, row.size_bytes)
                    records.append(
                        HistoryRecord(
                            kind=kind,
                            record_id=record_id,
                            checksum=reference.checksum,
                            payload=TypeAdapter(JsonValue).validate_json(
                                (await gateway.spool.read(reference)).split(b"\n", 2)[2]
                            ),
                        )
                    )
        return History(request_id=request_id, records=tuple(records), events=events)

    app.include_router(router)

    if settings.metrics_enabled:

        @app.get("/metrics", response_class=Response)
        async def metrics(_identity: Annotated[Identity, Depends(auth.reader)]) -> Response:
            current = services()
            return Response(
                await render_metrics(current.gateway, current.monitor, timing),
                media_type="text/plain; version=0.0.4",
            )
