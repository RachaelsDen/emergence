"""Shared actual-app fixture for operator recovery HTTP contracts."""

from collections import deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, tzinfo
from functools import partial
from pathlib import Path

import httpx
import pytest
from _gateway_fixture import NOW, Clock, answer, settings_for
from httpx import AsyncClient
from test_api import OPERATOR, READER

from emergence.api.app import create_app
from emergence.config import ProviderKind, Settings


@dataclass(frozen=True, slots=True)
class Harness:
    settings: Settings
    clock: Clock
    replies: deque[httpx.Response]
    calls: list[str]

    @asynccontextmanager
    async def open(self) -> AsyncGenerator[AsyncClient]:
        app = create_app(self.settings)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://core",
                headers={"Authorization": f"Bearer {OPERATOR}"},
            ) as client,
        ):
            assert (await client.get("/health/ready")).status_code == 200
            yield client


@pytest.fixture
def harness(migrated_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    clock = Clock(NOW)

    class FrozenTime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return clock().astimezone(tz)

    monkeypatch.setattr("emergence.resources.schemas.datetime", FrozenTime)
    monkeypatch.setattr("emergence.resources.ledger.datetime", FrozenTime)
    for name, value in (
        ("TEST_API_OPERATOR", OPERATOR),
        ("TEST_API_READER", READER),
        ("TEST_E005_INFERENCE", "synthetic-recovery-upstream"),
    ):
        monkeypatch.setenv(name, value)
    base = settings_for(ProviderKind.OPENAI)
    settings = base.model_copy(
        update={
            "providers": {
                **base.providers,
                "disabled": base.providers["p"].model_copy(update={"enabled": False}),
                "local": settings_for(ProviderKind.LLAMACPP).providers["p"],
            },
            "database_path": migrated_path,
            "spool_path": migrated_path.parent / "recovery-spool",
            "operator_token_env": "TEST_API_OPERATOR",
            "reader_token_env": "TEST_API_READER",
        }
    )
    result = Harness(settings, clock, deque(), [])

    def respond(request: httpx.Request) -> httpx.Response:
        result.calls.append(request.url.path)
        return result.replies.popleft() if result.replies else answer()

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(AsyncClient, transport=httpx.MockTransport(respond))
    )
    return result
