"""Synthetic provider transport with real migrations, storage, and financial admission."""

from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final

import httpx

from emergence.config import (
    ModelConfig,
    PriceConfig,
    ProviderConfig,
    ProviderKind,
    QuotaPolicy,
    Settings,
)
from emergence.inference.gateway import Gateway
from emergence.inference.schemas import InferenceRequest, Message
from emergence.providers.registry import Registry
from emergence.resources.ledger import BudgetLedger
from emergence.storage.archive import Spool
from emergence.storage.database import Database
from emergence.storage.repositories import ensure_catalog
from emergence.telemetry.logging import Redactor

NOW: Final = datetime(2026, 11, 20, 12, tzinfo=UTC)
type Handler = (
    Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
)


@dataclass
class Clock:
    at: datetime

    def __call__(self) -> datetime:
        return self.at


def settings_for(kind: ProviderKind = ProviderKind.ZAI) -> Settings:
    return Settings(
        disk_headroom_bytes=1,
        spool_limit_bytes=1024 * 1024,
        providers={
            "p": ProviderConfig(
                kind=kind,
                base_url="https://fixture.invalid/v1",
                credential_env="TEST_E005_INFERENCE",
                quota_policy=QuotaPolicy.LOCAL_LIMITS_ONLY,
                concurrency=4,
                response_limit_bytes=2048,
                organization="org-fixture" if kind is ProviderKind.OPENAI else None,
                project="project-fixture" if kind is ProviderKind.OPENAI else None,
                safety_limit_confirmed=True,
                models={
                    "m": ModelConfig(
                        id="configured-model",
                        max_input_tokens=20,
                        max_output_tokens=20,
                        input_overhead_tokens=0,
                        input_bound_verified=True,
                        input_bound_evidence="synthetic test contract",
                        output_bound_verified=True,
                        bound_evidence="synthetic test contract",
                        reasoning_in_output_bound_verified=True,
                        reasoning_bound_evidence="synthetic shared generated-token limit",
                    )
                },
            )
        },
        prices=()
        if kind is ProviderKind.LLAMACPP
        else (
            PriceConfig(
                version="fixture-v1",
                provider="p",
                model="m",
                effective_at=datetime(2026, 1, 1, tzinfo=UTC),
                input_per_million=Decimal("1000"),
                output_per_million=Decimal("1000"),
                source="synthetic test price, not a provider quote",
            ),
        ),
    )


def invocation(output: int = 10, content: str = "hello") -> InferenceRequest:
    return InferenceRequest(
        provider="p",
        model="m",
        messages=(Message(role="user", content=content),),
        max_output_tokens=output,
    )


def answer(inputs: int | None = 5, outputs: int | None = 3) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "synthetic",
            "model": "configured-model",
            "choices": [{"message": {"content": "bounded answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": inputs, "completion_tokens": outputs},
        },
    )


@asynccontextmanager
async def gateway_for(
    path: Path,
    settings: Settings,
    handler: Handler,
    clock: Callable[[], datetime] = lambda: NOW,
) -> AsyncGenerator[tuple[Gateway, BudgetLedger, Database, Spool]]:
    async with (
        Database(path) as database,
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
    ):
        await ensure_catalog(database, settings)
        spool = Spool(
            path.parent / "gateway-spool",
            database,
            Redactor(settings.secrets()),
            limit_bytes=settings.spool_limit_bytes,
            headroom_bytes=settings.disk_headroom_bytes,
        )
        await spool.recover()
        ledger = BudgetLedger(settings, database)
        gateway = Gateway(settings, database, spool, Registry(settings, client), ledger, clock)
        await gateway.initialize()
        yield gateway, ledger, database, spool
