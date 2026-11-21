"""Application factory and local-dependency health checks."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from hashlib import sha256

import httpx
from anyio import create_task_group
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import JsonValue, TypeAdapter
from sqlalchemy import insert, select
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

from emergence.api.dependencies import RequestBodyLimit, Services
from emergence.api.provider_recovery import install_provider_recovery
from emergence.api.routes import install_routes
from emergence.config import ConfigurationError, Settings, load_settings
from emergence.inference.gateway import Gateway, GatewayError
from emergence.providers.base import ProviderFailure
from emergence.providers.registry import Registry
from emergence.resources.ledger import BudgetError, BudgetLedger
from emergence.resources.quota_polling import QuotaMonitor
from emergence.storage.archive import RecordKey, Spool, SpoolError
from emergence.storage.database import Database
from emergence.storage.delivery import ArchiveDelivery
from emergence.storage.models import audit_events
from emergence.storage.repositories import ensure_catalog
from emergence.telemetry.logging import Redactor, configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings if settings is not None else load_settings()
    redactor = Redactor(config.secrets())
    database = Database(config.database_path)
    spool = Spool(
        config.spool_path,
        database,
        redactor,
        limit_bytes=config.spool_limit_bytes,
        headroom_bytes=config.disk_headroom_bytes,
        archive_path=config.archive_path,
        archive_mount_path=config.archive_mount_path,
    )
    ready = False
    services: Services | None = None

    def active_services() -> Services:
        if not ready or services is None:
            raise HTTPException(503, "local_storage_not_ready")
        return services

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        nonlocal ready, services
        configure_logging(redactor)
        async with database:
            ready = await database.readiness()
            if ready:
                try:
                    await spool.recover()
                    await ensure_catalog(database, config)
                    serialized = config.model_dump_json()
                    fingerprint = sha256(serialized.encode("utf-8")).hexdigest()
                    reference = await spool.publish(
                        RecordKey("operator-configuration", fingerprint),
                        TypeAdapter(JsonValue).validate_json(serialized),
                    )
                    async with database.write() as session:
                        previous = await session.scalar(
                            select(audit_events.c.detail_archive_record_id)
                            .where(audit_events.c.action == "configuration_changed")
                            .order_by(audit_events.c.id.desc())
                            .limit(1)
                        )
                        if previous != reference.record_id:
                            await session.execute(
                                insert(audit_events).values(
                                    occurred_at=datetime.now(UTC).isoformat(),
                                    actor_id="operator.configuration",
                                    action="configuration_changed",
                                    detail_archive_record_id=reference.record_id,
                                )
                            )
                except (ConfigurationError, SpoolError) as error:
                    ready = False
                    logging.getLogger("emergence").error(error.code)
                except (OSError, SQLAlchemyError):
                    ready = False
            async with (
                httpx.AsyncClient(trust_env=False, follow_redirects=False) as reporting_client,
                httpx.AsyncClient(trust_env=False, follow_redirects=False) as inference_client,
                create_task_group() as tasks,
            ):
                monitor = QuotaMonitor(config, database, spool, reporting_client)
                gateway = Gateway(
                    config,
                    database,
                    spool,
                    Registry(config, inference_client),
                    BudgetLedger(config, database),
                )
                delivery = ArchiveDelivery(spool, database)
                services = Services(gateway=gateway, monitor=monitor, archive=delivery)
                if ready:
                    try:
                        await gateway.initialize()
                    except (BudgetError, GatewayError, SpoolError) as error:
                        ready = False
                        logging.getLogger("emergence").error(error.code)
                    except (OSError, SQLAlchemyError):
                        ready = False
                logging.getLogger("emergence").info(
                    "service_started" if ready else "local_storage_not_ready"
                )
                if ready:
                    tasks.start_soon(monitor.run)
                    tasks.start_soon(delivery.run)
                try:
                    yield
                finally:
                    ready = False
                    tasks.cancel_scope.cancel()
                    logging.getLogger("emergence").info("service_stopped")

    app = FastAPI(title="Emergence Core", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        RequestBodyLimit,
        limit=config.request_limit_bytes,
        timeout=config.request_body_timeout_seconds,
    )
    install_routes(app, config, active_services)
    install_provider_recovery(app, config, active_services)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, error: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"error": error.detail},
            headers=error.headers,
        )

    @app.exception_handler(GatewayError)
    async def gateway_error(_request: Request, error: GatewayError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status,
            content={"error": error.code, "request_id": error.request_id},
        )

    @app.exception_handler(BudgetError)
    async def budget_error(_request: Request, error: BudgetError) -> JSONResponse:
        status = 402 if error.code in {"budget_insufficient", "safety_limit_exceeded"} else 409
        return JSONResponse(status_code=status, content={"error": error.code})

    @app.exception_handler(ProviderFailure)
    async def provider_error(_request: Request, error: ProviderFailure) -> JSONResponse:
        status = 404 if error.code == "model_unavailable" else 400
        return JSONResponse(status_code=status, content={"error": error.code})

    @app.exception_handler(SpoolError)
    async def spool_error(_request: Request, error: SpoolError) -> JSONResponse:
        return JSONResponse(status_code=507, content={"error": error.code})

    @app.exception_handler(SQLAlchemyError)
    async def database_error(_request: Request, _error: SQLAlchemyError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": "local_storage_unavailable"})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": "invalid_request"})

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def readiness() -> JSONResponse:
        healthy = ready and await database.readiness()
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={"status": "ready" if healthy else "not_ready"},
        )

    return app
