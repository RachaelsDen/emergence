"""Durable admission, one-shot dispatch, and restart-safe accounting."""

import json
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from emergence.config import ProviderKind, Settings
from emergence.inference.errors import GatewayError as GatewayError
from emergence.inference.outcomes import OutcomeStore
from emergence.inference.quota import check_quota
from emergence.inference.request_state import ArchivedOutcome as ArchivedOutcome
from emergence.inference.request_state import FailureRecord, append_event
from emergence.inference.request_state import GatewayResult as GatewayResult
from emergence.inference.request_state import RequestState as RequestState
from emergence.inference.schemas import InferenceRequest
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
    audit_events,
    request_reservations,
    requests,
    reservation_events,
)
from emergence.telemetry.logging import Redactor


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
        self.outcomes = OutcomeStore(database, spool, ledger, clock)
        self._active_requests: set[str] = set()

    async def initialize(self) -> None:
        """Run once before accepting traffic in the single core process."""
        await self.ledger.initialize()
        await self.recover()

    async def infer(
        self, invocation: InferenceRequest, idempotency_key: str, actor: str = "operator"
    ) -> GatewayResult:
        request_id = uuid4().hex
        self._active_requests.add(request_id)
        try:
            return await self._infer(invocation, idempotency_key, actor, request_id)
        finally:
            self._active_requests.remove(request_id)

    async def _infer(
        self, invocation: InferenceRequest, idempotency_key: str, actor: str, request_id: str
    ) -> GatewayResult:
        canonical = json.dumps(
            invocation.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(canonical) > self.settings.request_limit_bytes:
            raise GatewayError("request_too_large", 413)
        fingerprint = sha256(canonical).hexdigest()
        identity = sha256(idempotency_key.encode("utf-8")).hexdigest()
        now = self.clock().astimezone(UTC)
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
                await append_event(session, request_id, RequestState.ADMITTED, now)
        if existing is not None:
            return await self.outcomes.replay(
                existing.id, active=existing.id in self._active_requests
            )
        try:
            input_ref = await self.spool.publish(
                RecordKey(request_id, "input"),
                {"request_id": request_id, "request": invocation.model_dump(mode="json")},
                claim_request_id=request_id,
            )
        except (SpoolError, OSError, SQLAlchemyError):
            await self.outcomes.fail_before_dispatch(request_id, "input_archive_failed")
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
                await append_event(
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
            await self.outcomes.fail_before_dispatch(request_id, error.code)
            raise GatewayError(error.code, error.status, request_id) from None
        except SQLAlchemyError:
            # If the writer is still unavailable, an idempotent replay can
            # fail the abandoned admission once storage recovers.
            with suppress(SQLAlchemyError):
                await self.outcomes.fail_before_dispatch(request_id, "local_storage_unavailable")
            raise GatewayError("local_storage_unavailable", request_id=request_id) from None
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
            # If this also fails, replay can recover the inactive dispatch.
            with suppress(SQLAlchemyError):
                await self.outcomes.mark_unarchived_dispatch(request_id, "output_archive_failed")
            raise GatewayError("output_archive_failed", request_id=request_id) from None
        verified_outcome = await self.outcomes.read_outcome(reference)
        async with self.database.write() as session:
            state, code = await self.outcomes.finish(session, verified_outcome, reference.record_id)
        return GatewayResult(
            request_id=request_id, state=state, result=verified_outcome.result, error_code=code
        )

    async def get(self, request_id: str, *, replayed: bool = False) -> GatewayResult:
        return await self.outcomes.get(request_id, replayed=replayed)

    async def recover(self) -> None:
        await self.outcomes.recover()

    async def resolve(
        self, request_id: str, amount: Decimal | None, reason: str, actor: str
    ) -> GatewayResult:
        return await self.outcomes.resolve(request_id, amount, reason, actor)
