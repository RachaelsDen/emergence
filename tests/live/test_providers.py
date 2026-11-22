"""Real calls are never part of ordinary tests and never replay an uncertain outcome."""

import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx
import pytest
from alembic import command
from alembic.config import Config
from anyio import to_thread

from emergence.config import ProviderKind, load_settings
from emergence.inference.gateway import Gateway, RequestState
from emergence.inference.schemas import InferenceRequest, Message
from emergence.providers.registry import Registry
from emergence.resources.pricing import effective
from emergence.resources.quota_polling import QuotaMonitor
from emergence.storage.archive import Spool
from emergence.storage.database import Database
from emergence.storage.repositories import ensure_catalog
from emergence.telemetry.logging import Redactor
from tests.live.contract import LiveConfigurationError, LiveOptions
from tests.live.ledger import CappedSmokeLedger

PROJECT: Final = Path(__file__).resolve().parents[2]


@asynccontextmanager
async def prepared_run(
    options: LiveOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[tuple[Gateway, QuotaMonitor, LiveOptions]]:
    settings = options.bounded_settings(
        await to_thread.run_sync(load_settings, options.config), tmp_path
    )
    monkeypatch.setenv("EMERGENCE_DATABASE_PATH", str(settings.database_path))
    await to_thread.run_sync(command.upgrade, Config(str(PROJECT / "alembic.ini")), "head")
    async with (
        Database(settings.database_path) as database,
        httpx.AsyncClient(trust_env=False, follow_redirects=False) as inference_client,
        httpx.AsyncClient(trust_env=False, follow_redirects=False) as reporting_client,
    ):
        await ensure_catalog(database, settings)
        spool = Spool(
            settings.spool_path,
            database,
            Redactor(settings.secrets()),
            limit_bytes=settings.spool_limit_bytes,
            headroom_bytes=settings.disk_headroom_bytes,
        )
        await spool.recover()
        gateway = Gateway(
            settings,
            database,
            spool,
            Registry(settings, inference_client),
            CappedSmokeLedger(settings, database, options.request_cap),
        )
        await gateway.initialize()
        for name, config in settings.providers.items():
            alias = options.model(settings, name)
            request = InferenceRequest(
                provider=name,
                model=alias,
                messages=(Message(role="user", content="Reply with OK."),),
                max_output_tokens=options.max_output_tokens,
            )
            gateway.registry.get(name).preflight(request, cost_enforced=True)
            if config.kind is not ProviderKind.LLAMACPP:
                if settings.secret(config.credential_env) is None:
                    raise LiveConfigurationError("live_inference_credential_missing")
                async with database.session() as session:
                    await effective(session, name, alias, datetime.now(UTC))
        yield gateway, QuotaMonitor(settings, database, spool, reporting_client), options


@pytest.fixture
async def live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[tuple[Gateway, QuotaMonitor, LiveOptions]]:
    options = LiveOptions.from_environment(os.environ)
    async with prepared_run(options, tmp_path, monkeypatch) as run:
        yield run


@pytest.mark.live
@pytest.mark.anyio
async def test_selected_provider_inference(
    live_run: tuple[Gateway, QuotaMonitor, LiveOptions],
) -> None:
    gateway, _, options = live_run
    for name in options.providers:
        alias = options.model(gateway.settings, name)
        result = await gateway.infer(
            InferenceRequest(
                provider=name,
                model=alias,
                messages=(Message(role="user", content="Reply with OK."),),
                max_output_tokens=options.max_output_tokens,
            ),
            f"live-smoke:{name}",
        )
        print(
            json.dumps(
                {
                    "provider": gateway.redactor.text(name),
                    "model": gateway.redactor.text(alias),
                    "request_id": result.request_id,
                    "state": result.state.value,
                    "error": result.error_code,
                    "resolved_model": (
                        gateway.redactor.text(result.result.upstream_model)
                        if result.result is not None
                        else None
                    ),
                    "date": datetime.now(UTC).isoformat(),
                }
            )
        )
        assert result.state is RequestState.COMPLETED, result.error_code
        assert result.result is not None
        assert result.result.usage.input_tokens is not None
        assert result.result.usage.output_tokens is not None


@pytest.mark.live
@pytest.mark.anyio
async def test_selected_provider_reporting(
    live_run: tuple[Gateway, QuotaMonitor, LiveOptions],
) -> None:
    if os.environ.get("EMERGENCE_LIVE_REPORTING") != "1":
        pytest.skip("set EMERGENCE_LIVE_REPORTING=1 to opt into account reporting reads")
    gateway, monitor, options = live_run
    for name in options.providers:
        observations = await monitor.poll_once(name)
        print(
            json.dumps(
                {
                    "provider": gateway.redactor.text(name),
                    "reporting": [item.status for item in observations],
                    "issues": [item.issue.value if item.issue else None for item in observations],
                    "date": datetime.now(UTC).isoformat(),
                }
            )
        )
        assert observations
        assert all(item.status != "unavailable" for item in observations)
