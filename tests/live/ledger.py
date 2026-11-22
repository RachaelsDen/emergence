"""An independent per-call ceiling for explicitly authorized smoke runs."""

from datetime import datetime
from decimal import Decimal
from typing import override

from sqlalchemy.ext.asyncio import AsyncSession

from emergence.config import Settings
from emergence.inference.schemas import InferenceRequest
from emergence.resources.ledger import BudgetError, BudgetLedger
from emergence.storage.database import Database


class CappedSmokeLedger(BudgetLedger):
    def __init__(self, settings: Settings, database: Database, cap: Decimal) -> None:
        super().__init__(settings, database)
        self._cap = cap

    @override
    async def reserve(
        self, session: AsyncSession, request_id: str, request: InferenceRequest, now: datetime
    ) -> Decimal | None:
        amount = await super().reserve(session, request_id, request, now)
        if amount is not None and amount > self._cap:
            raise BudgetError("live_request_cap_exceeded")
        return amount
