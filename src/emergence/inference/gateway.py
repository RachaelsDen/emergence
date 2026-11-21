"""Durable admission, one-shot dispatch, and restart-safe accounting."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from uuid import uuid4

from pydantic import AwareDatetime, JsonValue, ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.config import ProviderKind, SafeModel, Settings
from emergence.inference.errors import GatewayError as GatewayError
from emergence.inference.quota import check_quota
from emergence.inference.schemas import InferenceRequest, InferenceResult, TokenUsage
from emergence.providers.base import ProviderFailure
from emergence.providers.registry import Registry
from emergence.resources.ledger import BudgetLedger
from emergence.resources.pricing import stamp
from emergence.resources.schemas import utc_now
from emergence.storage.archive import RecordKey, Spool, SpoolError
from emergence.storage.capacity import live_claims
from emergence.storage.database import Database
from emergence.storage.gateway_state import request_capacity
from emergence.storage.models import (
    archive_references,
    audit_events,
    providers,
    quota_observations,
    request_events,
    request_reservations,
    requests,
    reservation_events,
)
from emergence.storage.repositories import ArchiveReference
from emergence.telemetry.logging import Redactor


class RequestState(StrEnum):
    ADMITTED = "admitted"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    RESOLVED = "resolved"


class FailureRecord(SafeModel):
    code: str
    ambiguous: bool
    status_code: int | None = None
    raw_response: JsonValue = None


class ArchivedOutcome(SafeModel):
    request_id: str
    observed_at: AwareDatetime
    result: InferenceResult | None = None
    failure: FailureRecord | None = None


class GatewayResult(SafeModel):
    request_id: str
    state: RequestState
    result: InferenceResult | None = None
    error_code: str | None = None
    replayed: bool = False


async def _event(
    session: AsyncSession,
    request_id: str,
    state: RequestState,
    now: datetime,
    *,
    reference: str | None = None,
    code: str | None = None,
) -> None:
    previous = await session.scalar(
        select(func.max(request_events.c.sequence)).where(request_events.c.request_id == request_id)
    )
    await session.execute(
        request_events.insert().values(
            request_id=request_id,
            sequence=0 if previous is None else previous + 1,
            kind=state.value,
            occurred_at=stamp(now),
            archive_record_id=reference,
            outcome_code=code,
        )
    )


async def _state(session: AsyncSession, request_id: str) -> tuple[RequestState, str | None]:
    latest = (
        await session.execute(
            select(request_events.c.kind, request_events.c.outcome_code)
            .where(request_events.c.request_id == request_id)
            .order_by(request_events.c.sequence.desc())
            .limit(1)
        )
    ).one_or_none()
    if latest is None:
        raise GatewayError("request_not_found", 404)
    try:
        return RequestState(latest.kind), latest.outcome_code
    except ValueError:
        raise GatewayError("request_state_invalid") from None


class Gateway:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        spool: Spool,
        registry: Registry,
        ledger: BudgetLedger,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.settings = settings
        self.database = database
        self.spool = spool
        self.registry = registry
        self.ledger = ledger
        self.clock = clock
        self.redactor = Redactor(settings.secrets())

    async def initialize(self) -> None:
        """Run once before accepting traffic in the single core process."""
        await self.ledger.initialize()
        await self.recover()

    async def infer(
        self, invocation: InferenceRequest, idempotency_key: str, actor: str = "operator"
    ) -> GatewayResult:
        canonical = json.dumps(
            invocation.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(canonical) > self.settings.request_limit_bytes:
            raise GatewayError("request_too_large", 413)
        fingerprint = sha256(canonical).hexdigest()
        identity = sha256(idempotency_key.encode("utf-8")).hexdigest()
        request_id = uuid4().hex
        now = self.clock().astimezone(UTC)
        existing_id: str | None = None
        async with self.database.write() as session:
            existing = (
                await session.execute(
                    select(requests.c.id, requests.c.input_fingerprint).where(
                        requests.c.idempotency_scope == actor,
                        requests.c.idempotency_key == identity,
                    )
                )
            ).one_or_none()
            if existing is not None:
                if existing.input_fingerprint != fingerprint:
                    raise GatewayError("idempotency_conflict", 409)
                existing_id = existing.id
            else:
                provider = self.registry.get(invocation.provider)
                model = provider.preflight(
                    invocation, cost_enforced=provider.config.kind is not ProviderKind.LLAMACPP
                )
                if (
                    provider.config.kind is not ProviderKind.LLAMACPP
                    and self.settings.secret(provider.config.credential_env) is None
                ):
                    raise GatewayError("inference_credential_unavailable")
                violation = await session.scalar(
                    select(audit_events.c.id)
                    .join(requests, requests.c.id == audit_events.c.request_id)
                    .where(
                        audit_events.c.action == "billable_bound_violation",
                        requests.c.provider_id == invocation.provider,
                        requests.c.model_id == invocation.model,
                    )
                    .limit(1)
                )
                if violation is not None:
                    raise GatewayError("billable_bound_unverified", 409)
                claims = await live_claims(session)
                if sum(claim.provider == invocation.provider for claim in claims) >= (
                    provider.config.concurrency
                ):
                    raise GatewayError("provider_concurrency_exhausted", 429)
                await session.execute(
                    requests.insert().values(
                        id=request_id,
                        idempotency_scope=actor,
                        idempotency_key=identity,
                        input_fingerprint=fingerprint,
                        provider_id=invocation.provider,
                        model_id=invocation.model,
                        created_at=stamp(now),
                    )
                )
                amount = await self.ledger.reserve(session, request_id, invocation, now)
                await check_quota(self.settings, session, invocation, now, amount)
                input_key = RecordKey(request_id, "input")
                input_ref = self.spool.preview(
                    input_key,
                    {"request_id": request_id, "request": invocation.model_dump(mode="json")},
                )
                # ASCII escaping (<=3x valid UTF-8 JSON), normalized/raw duplication,
                # and envelope fields fit within 12x the bounded wire response.
                # Known-secret replacement is one-pass and idempotent.
                capacity = (
                    input_ref.size_bytes
                    + provider.config.response_limit_bytes * 12 * self.redactor.expansion_factor
                    + 65536
                )
                await self.spool.ensure_capacity(session, capacity)
                await session.execute(
                    request_capacity.insert().values(
                        request_id=request_id,
                        allocated_bytes=capacity,
                        input_record_id=input_key.record_id,
                        output_record_id=RecordKey(request_id, "output").record_id,
                        input_tokens_bound=model.max_input_tokens,
                        output_tokens_bound=invocation.max_output_tokens,
                        created_at=stamp(now),
                    )
                )
                await _event(session, request_id, RequestState.ADMITTED, now)
        if existing_id is not None:
            return await self.get(existing_id, replayed=True)
        try:
            input_ref = await self.spool.publish(
                RecordKey(request_id, "input"),
                {"request_id": request_id, "request": invocation.model_dump(mode="json")},
                claim_request_id=request_id,
            )
        except (SpoolError, OSError, SQLAlchemyError):
            await self._fail_before_dispatch(request_id, "input_archive_failed")
            raise GatewayError("input_archive_failed", request_id=request_id) from None
        try:
            async with self.database.write() as session:
                amount = await session.scalar(
                    select(request_reservations.c.amount).where(
                        request_reservations.c.request_id == request_id
                    )
                )
                await check_quota(
                    self.settings, session, invocation, self.clock(), amount, exclude=request_id
                )
                await _event(
                    session,
                    request_id,
                    RequestState.DISPATCHED,
                    self.clock(),
                    reference=input_ref.record_id,
                )
                if amount is not None:
                    sequence: int = (
                        await session.execute(
                            select(func.max(reservation_events.c.sequence)).where(
                                reservation_events.c.request_id == request_id
                            )
                        )
                    ).scalar_one()
                    await session.execute(
                        reservation_events.insert().values(
                            request_id=request_id,
                            sequence=sequence + 1,
                            kind="dispatched",
                            occurred_at=stamp(self.clock()),
                        )
                    )
        except GatewayError as error:
            await self._fail_before_dispatch(request_id, error.code)
            raise GatewayError(error.code, error.status, request_id) from None
        try:
            result = await self.registry.get(invocation.provider).infer(invocation)
            outcome = ArchivedOutcome(
                request_id=request_id, observed_at=self.clock(), result=result
            )
        except ProviderFailure as failure:
            outcome = ArchivedOutcome(
                request_id=request_id,
                observed_at=self.clock(),
                failure=FailureRecord(
                    code=failure.code,
                    ambiguous=failure.ambiguous,
                    status_code=failure.status_code,
                    raw_response=failure.raw_response,
                ),
            )
        try:
            reference = await self.spool.publish(
                RecordKey(request_id, "output"),
                outcome.model_dump(mode="json"),
                claim_request_id=request_id,
            )
        except (SpoolError, OSError, SQLAlchemyError):
            async with self.database.write() as session:
                await _event(
                    session,
                    request_id,
                    RequestState.AMBIGUOUS,
                    self.clock(),
                    code="output_archive_failed",
                )
            raise GatewayError("output_archive_failed", request_id=request_id) from None
        verified_outcome = await self._read_outcome(reference)
        async with self.database.write() as session:
            state, code = await self._finish(session, verified_outcome, reference.record_id)
        return GatewayResult(
            request_id=request_id, state=state, result=verified_outcome.result, error_code=code
        )

    async def _fail_before_dispatch(self, request_id: str, code: str) -> None:
        async with self.database.write() as session:
            await self.ledger.release(session, request_id, self.clock(), code)
            await _event(session, request_id, RequestState.FAILED, self.clock(), code=code)

    async def _finish(
        self, session: AsyncSession, outcome: ArchivedOutcome, reference: str | None
    ) -> tuple[RequestState, str | None]:
        now = self.clock()
        previous_state, previous_code = await _state(session, outcome.request_id)
        state, code = RequestState.AMBIGUOUS, None
        if outcome.result is not None:
            result = outcome.result
            bounds = (
                await session.execute(
                    select(request_capacity).where(
                        request_capacity.c.request_id == outcome.request_id
                    )
                )
            ).one()
            exceeded = (
                result.usage.input_tokens is not None
                and result.usage.input_tokens > bounds.input_tokens_bound
            ) or (
                result.usage.output_tokens is not None
                and result.usage.output_tokens > bounds.output_tokens_bound
            )
            if exceeded:
                code = "billable_bound_violation"
                previous = await session.scalar(
                    select(audit_events.c.id).where(
                        audit_events.c.request_id == outcome.request_id,
                        audit_events.c.action == code,
                    )
                )
                if previous is None:
                    await session.execute(
                        audit_events.insert().values(
                            occurred_at=stamp(now),
                            actor_id="gateway",
                            action=code,
                            request_id=outcome.request_id,
                            provider_id=result.provider,
                        )
                    )
            if await self.ledger.settle(
                session, outcome.request_id, result.usage, outcome.observed_at
            ):
                state = RequestState.COMPLETED
            else:
                code = code or "usage_unknown"
        elif outcome.failure is not None:
            failure = outcome.failure
            code = failure.code
            match failure.raw_response:
                case {"error": {"code": "insufficient_quota" | "billing_hard_limit_reached"}}:
                    if failure.status_code in {402, 429} and (
                        previous_state is not RequestState.AMBIGUOUS
                    ):
                        await session.execute(
                            quota_observations.insert().values(
                                provider_id=(
                                    await session.execute(
                                        select(requests.c.provider_id).where(
                                            requests.c.id == outcome.request_id
                                        )
                                    )
                                ).scalar_one(),
                                resource="inference",
                                unit="availability",
                                remaining=Decimal(0),
                                observed_at=stamp(outcome.observed_at),
                                source="inference",
                                confidence="provider_rejection",
                                status="observed",
                                raw_archive_record_id=reference,
                            )
                        )
                case _:
                    pass
            if failure.ambiguous:
                kind: str = (
                    await session.execute(
                        select(providers.c.kind)
                        .join(requests, requests.c.provider_id == providers.c.id)
                        .where(requests.c.id == outcome.request_id)
                    )
                ).scalar_one()
                if kind != ProviderKind.LLAMACPP.value:
                    await self.ledger.settle(session, outcome.request_id, TokenUsage(), now)
            else:
                await self.ledger.release(session, outcome.request_id, now, failure.code)
                state = RequestState.FAILED
        if previous_state is not state or previous_code != code:
            await _event(session, outcome.request_id, state, now, reference=reference, code=code)
        return state, code

    async def _read_outcome(self, reference: ArchiveReference) -> ArchivedOutcome:
        raw = await self.spool.read(reference)
        try:
            return ArchivedOutcome.model_validate_json(raw.split(b"\n", 2)[2])
        except ValidationError:
            raise GatewayError("request_archive_invalid") from None

    async def _outcome(self, request_id: str) -> ArchivedOutcome | None:
        expected = RecordKey(request_id, "output").record_id
        async with self.database.session() as session:
            row = (
                await session.execute(
                    select(archive_references).where(archive_references.c.record_id == expected)
                )
            ).one_or_none()
        if row is None:
            return None
        return await self._read_outcome(
            ArchiveReference(row.record_id, row.checksum, row.size_bytes)
        )

    async def get(self, request_id: str, *, replayed: bool = False) -> GatewayResult:
        async with self.database.session() as session:
            state, code = await _state(session, request_id)
        outcome = await self._outcome(request_id)
        return GatewayResult(
            request_id=request_id,
            state=state,
            result=outcome.result if outcome is not None else None,
            error_code=code,
            replayed=replayed,
        )

    async def recover(self) -> None:
        """Reconcile archived outcomes and never replay an uncertain provider call."""
        async with self.database.session() as session:
            pending = await live_claims(session)
        for claim in pending:
            outcome = await self._outcome(claim.request_id)
            if outcome is not None:
                async with self.database.write() as session:
                    await self._finish(session, outcome, claim.output_record_id)
            elif claim.state == "admitted":
                await self._fail_before_dispatch(claim.request_id, "interrupted_before_dispatch")
            elif claim.state == "dispatched":
                async with self.database.write() as session:
                    await self._finish(
                        session,
                        ArchivedOutcome(
                            request_id=claim.request_id,
                            observed_at=self.clock(),
                            failure=FailureRecord(
                                code="interrupted_after_dispatch", ambiguous=True
                            ),
                        ),
                        None,
                    )

    async def resolve(
        self, request_id: str, amount: Decimal | None, reason: str, actor: str
    ) -> GatewayResult:
        outcome = await self._outcome(request_id)
        async with self.database.write() as session:
            state, _ = await _state(session, request_id)
            if state is not RequestState.AMBIGUOUS:
                raise GatewayError("request_not_ambiguous", 409)
            await self.ledger.resolve(session, request_id, amount, self.clock(), reason, actor)
            await _event(
                session,
                request_id,
                RequestState.RESOLVED,
                self.clock(),
                code="operator_reconciliation",
            )
        return GatewayResult(
            request_id=request_id,
            state=RequestState.RESOLVED,
            result=outcome.result if outcome is not None else None,
            error_code="operator_reconciliation",
        )
