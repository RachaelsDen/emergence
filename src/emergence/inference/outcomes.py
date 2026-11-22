"""Finalize verified outcomes and recover durable request accounting."""

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.config import ProviderKind
from emergence.inference.archived_outcomes import parse_archived_outcome
from emergence.inference.errors import GatewayError
from emergence.inference.request_state import (
    ArchivedOutcome,
    FailureRecord,
    GatewayResult,
    RequestState,
    append_event,
    read_state,
)
from emergence.inference.schemas import TokenUsage
from emergence.resources.ledger import BudgetLedger
from emergence.resources.pricing import stamp
from emergence.storage.archive import RecordKey, Spool
from emergence.storage.capacity import live_claims
from emergence.storage.database import Database
from emergence.storage.gateway_state import request_capacity
from emergence.storage.models import (
    archive_references,
    audit_events,
    providers,
    quota_observations,
    requests,
)
from emergence.storage.repositories import ArchiveReference


class OutcomeStore:
    """Own durable outcome accounting without provider dispatch."""

    def __init__(
        self,
        database: Database,
        spool: Spool,
        ledger: BudgetLedger,
        clock: Callable[[], datetime],
    ) -> None:
        self.database = database
        self.spool = spool
        self.ledger = ledger
        self.clock = clock

    async def fail_before_dispatch(self, request_id: str, code: str) -> None:
        async with self.database.write() as session:
            state, _ = await read_state(session, request_id)
            if state is not RequestState.ADMITTED:
                return
            await self.ledger.release(session, request_id, self.clock(), code)
            await append_event(session, request_id, RequestState.FAILED, self.clock(), code=code)

    async def finish(
        self, session: AsyncSession, outcome: ArchivedOutcome, reference: str | None
    ) -> tuple[RequestState, str | None]:
        now = self.clock()
        previous_state, previous_code = await read_state(session, outcome.request_id)
        if previous_state in {RequestState.COMPLETED, RequestState.FAILED, RequestState.RESOLVED}:
            return previous_state, previous_code
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
            await append_event(
                session, outcome.request_id, state, now, reference=reference, code=code
            )
        return state, code

    async def read_outcome(self, reference: ArchiveReference) -> ArchivedOutcome:
        raw = await self.spool.read(reference)
        return parse_archived_outcome(raw.split(b"\n", 2)[2])

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
        return await self.read_outcome(
            ArchiveReference(row.record_id, row.checksum, row.size_bytes)
        )

    async def mark_unarchived_dispatch(
        self, request_id: str, code: str
    ) -> tuple[RequestState, str | None]:
        async with self.database.write() as session:
            state, previous_code = await read_state(session, request_id)
            if state is not RequestState.DISPATCHED:
                return state, previous_code
            return await self.finish(
                session,
                ArchivedOutcome(
                    request_id=request_id,
                    observed_at=self.clock(),
                    failure=FailureRecord(code=code, ambiguous=True),
                ),
                None,
            )

    async def replay(self, request_id: str, *, active: bool) -> GatewayResult:
        """Verify archive evidence outside the writer, then reconcile current state."""
        async with self.database.session() as session:
            state, code = await read_state(session, request_id)
        if state is RequestState.ADMITTED and not active:
            await self.fail_before_dispatch(request_id, "interrupted_before_dispatch")
            return await self.get(request_id, replayed=True)
        outcome = await self._outcome(request_id)
        if outcome is not None and state not in {
            RequestState.COMPLETED,
            RequestState.FAILED,
            RequestState.RESOLVED,
        }:
            async with self.database.write() as session:
                state, code = await self.finish(
                    session, outcome, RecordKey(request_id, "output").record_id
                )
        elif outcome is None and state is RequestState.DISPATCHED and not active:
            state, code = await self.mark_unarchived_dispatch(
                request_id, "interrupted_after_dispatch"
            )
        return GatewayResult(
            request_id=request_id,
            state=state,
            result=outcome.result if outcome is not None else None,
            error_code=code,
            replayed=True,
        )

    async def get(self, request_id: str, *, replayed: bool = False) -> GatewayResult:
        async with self.database.session() as session:
            state, code = await read_state(session, request_id)
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
                    await self.finish(session, outcome, claim.output_record_id)
            elif claim.state == "admitted":
                await self.fail_before_dispatch(claim.request_id, "interrupted_before_dispatch")
            elif claim.state == "dispatched":
                await self.mark_unarchived_dispatch(claim.request_id, "interrupted_after_dispatch")

    async def resolve(
        self, request_id: str, amount: Decimal | None, reason: str, actor: str
    ) -> GatewayResult:
        outcome = await self._outcome(request_id)
        async with self.database.write() as session:
            state, _ = await read_state(session, request_id)
            if state is not RequestState.AMBIGUOUS:
                raise GatewayError("request_not_ambiguous", 409)
            await self.ledger.resolve(session, request_id, amount, self.clock(), reason, actor)
            await append_event(
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
