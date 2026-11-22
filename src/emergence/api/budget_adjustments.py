"""Authenticated, persistently identified financial adjustment operations."""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException

from emergence.config import Settings
from emergence.resources.budgets import AdjustmentReceipt

from .dependencies import ERROR_DESCRIPTIONS, Authorization, ErrorResponse, Identity, Services
from .routes import Adjustment


def install_budget_adjustments(
    app: FastAPI, settings: Settings, services: Callable[[], Services]
) -> None:
    auth = Authorization(settings)
    router = APIRouter(
        prefix="/api/v1",
        responses={
            code: {"model": ErrorResponse, "description": description}
            for code, description in ERROR_DESCRIPTIONS.items()
        },
    )

    @router.post("/budgets/adjustments")
    async def adjust(
        body: Adjustment,
        identity: Annotated[Identity, Depends(auth.operator)],
        idempotency_key: Annotated[
            str, Header(alias="Idempotency-Key", pattern=r"^[\x21-\x7e]{1,128}$")
        ],
    ) -> AdjustmentReceipt:
        if (config := settings.providers.get(body.provider)) is None or not config.enabled:
            raise HTTPException(404, "provider_not_found")
        return await services().gateway.ledger.adjust(
            body.provider,
            body.amount,
            body.kind,
            body.reason,
            identity.actor,
            idempotency_key=idempotency_key,
        )

    app.include_router(router)
