"""Independent provider polling, with bounded concurrency and durable snapshots."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from random import uniform
from typing import Final

import httpx
from anyio import CapacityLimiter, Lock, create_task_group, sleep, sleep_forever
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from emergence.config import ProviderConfig, ProviderKind, Settings
from emergence.storage.archive import Spool, SpoolError
from emergence.storage.database import Database
from emergence.storage.models import requests
from emergence.storage.repositories import ArchiveReferenceConflict
from emergence.telemetry.logging import Redactor

from .reporting_http import ReportingHTTP
from .reporting_openai import OpenAIAdmin
from .reporting_parsers import WireObject, parse_minimax, parse_zai
from .reporting_store import ReportingStore
from .schemas import QuotaObservation, ReportingError, ReportingIssue, ReportScope, utc_now

_LOG: Final = logging.getLogger(__name__)
_EXPIRY_WARNING: Final = timedelta(days=7)


class QuotaMonitor:
    """Use a dedicated, auth-free reporting HTTP client owned by the caller."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        spool: Spool,
        client: httpx.AsyncClient,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._settings = settings
        self._database = database
        self._store = ReportingStore(database, spool)
        self._client = client
        self._clock = clock
        self._limiter = CapacityLimiter(settings.poll_concurrency)
        self._providers = {name: cfg for name, cfg in settings.providers.items() if cfg.enabled}
        self._locks = {name: Lock() for name in self._providers}

    async def poll_once(self, provider_name: str) -> tuple[QuotaObservation, ...]:
        """Poll one enabled provider; expected upstream failures become observations."""
        config = self._providers[provider_name]
        async with self._locks[provider_name], self._limiter:
            now = self._clock().astimezone(UTC)
            return await self._poll(provider_name, config, now)

    async def latest(self) -> tuple[QuotaObservation, ...]:
        """Read persisted observations and compute freshness using the current clock."""
        return await self._store.latest(self._settings, self._clock().astimezone(UTC))

    async def run(self) -> None:
        """Run until cancelled, even with no enabled providers."""
        async with create_task_group() as group:
            for provider, config in self._providers.items():
                group.start_soon(self._loop, provider, config.poll_interval_seconds)
            await sleep_forever()

    async def _loop(self, provider: str, interval: float) -> None:
        # At service startup do not wait an entire interval for the first observation.
        await sleep(uniform(0, min(interval * 0.1, 1.0)))
        failures = 0
        while True:
            try:
                observations = await self.poll_once(provider)
                failed = any(
                    obs.status == "unavailable" and obs.issue != ReportingIssue.UNSUPPORTED
                    for obs in observations
                )
                if all(obs.issue == ReportingIssue.UNSUPPORTED for obs in observations):
                    return
            except (SQLAlchemyError, SpoolError, ArchiveReferenceConflict, OSError):
                _LOG.warning("reporting persistence unavailable")
                failed = True
            failures = min(failures + 1, 4) if failed else 0
            await sleep(interval * 2**failures * uniform(0.9, 1.1))

    def _credential(self, config: ProviderConfig, now: datetime) -> str:
        if config.reporting_key_expires_at is not None and now >= config.reporting_key_expires_at:
            raise ReportingError(ReportingIssue.EXPIRED_CREDENTIAL)
        name = config.reporting_credential_env
        if config.kind is not ProviderKind.OPENAI and name is None:
            name = config.credential_env
        secret = self._settings.secret(name)
        if secret is None:
            raise ReportingError(ReportingIssue.MISSING_CREDENTIAL)
        if config.kind is ProviderKind.OPENAI and any(
            secret == self._settings.secret(provider.credential_env)
            for provider in self._settings.providers.values()
        ):
            raise ReportingError(ReportingIssue.SHARED_CREDENTIAL)
        return secret.get_secret_value()

    async def _failure(
        self,
        name: str,
        resource: str,
        config: ProviderConfig,
        now: datetime,
        issue: ReportingIssue,
        *,
        pages: tuple[WireObject, ...] = (),
        scope: ReportScope | None = None,
    ) -> tuple[QuotaObservation, ...]:
        if issue in {ReportingIssue.EXPIRED_CREDENTIAL, ReportingIssue.EXPIRING_CREDENTIAL}:
            _LOG.warning("reporting credential alert: %s", issue.value)
        return await self._store.append(
            (
                QuotaObservation(
                    provider=name,
                    resource=resource,
                    source=f"reporting:{config.kind}",
                    observed_at=now,
                    status="unavailable",
                    issue=issue,
                    scope=scope,
                ),
            ),
            config.stale_after_seconds,
            pages,
        )

    async def _poll(
        self, name: str, config: ProviderConfig, now: datetime
    ) -> tuple[QuotaObservation, ...]:
        if config.kind is ProviderKind.LLAMACPP:
            async with self._database.session() as session:
                count = await session.scalar(
                    select(func.count()).select_from(requests).where(requests.c.provider_id == name)
                )
            return await self._store.append(
                (
                    QuotaObservation(
                        provider=name,
                        resource="accepted_requests",
                        source="local",
                        unit="requests",
                        used=Decimal(count) if count is not None else None,
                        observed_at=now,
                        confidence="recorded_core_requests",
                        status="observed",
                    ),
                ),
                config.stale_after_seconds,
            )
        if config.kind is not ProviderKind.OPENAI and config.quota_url is None:
            return await self._failure(name, "quota", config, now, ReportingIssue.UNSUPPORTED)
        http = ReportingHTTP(self._client, config.timeout_seconds, config.response_limit_bytes)
        if config.kind is ProviderKind.OPENAI:
            return await self._openai(name, config, now, http)
        url = config.quota_url
        assert url is not None  # The unsupported path above has already returned.
        pages: tuple[WireObject, ...] = ()
        try:
            credential = self._credential(config, now)
            raw, _ = await http.get(url, headers={"Authorization": f"Bearer {credential}"})
            pages = (raw,)
            redactor = Redactor(self._settings.secrets())
            if config.kind is ProviderKind.ZAI:
                categories = parse_zai(raw)
                if any(redactor.text(category) != category for category, _ in categories):
                    raise ReportingError(ReportingIssue.INVALID_SCHEMA)
                observations = tuple(
                    QuotaObservation(
                        provider=name,
                        resource=category,
                        unit="percent" if percent is not None else "unknown",
                        usage_percent=percent,
                        source="reporting:zai",
                        observed_at=now,
                        confidence="undocumented_schema",
                        status="experimental",
                    )
                    for category, percent in categories
                )
            else:
                parse_minimax(raw)
                observations = (
                    QuotaObservation(
                        provider=name,
                        resource="quota",
                        source="reporting:minimax",
                        observed_at=now,
                        confidence="unobserved_schema",
                        status="experimental",
                    ),
                )
        except ReportingError as error:
            return await self._failure(name, "quota", config, now, error.issue, pages=pages)
        return await self._store.append(observations, config.stale_after_seconds, pages)

    async def _openai(
        self, name: str, config: ProviderConfig, now: datetime, http: ReportingHTTP
    ) -> tuple[QuotaObservation, ...]:
        redactor = Redactor(self._settings.secrets())
        try:
            credential = self._credential(config, now)
            org, project = config.organization, config.project
            if (
                not org
                or not project
                or org != org.strip()
                or project != project.strip()
                or redactor.text(org) != org
                or redactor.text(project) != project
            ):
                raise ReportingError(ReportingIssue.MISSING_SCOPE)
            scope = ReportScope(organization=org, project=project)
        except ReportingError as error:
            usage_failure = await self._failure(name, "usage", config, now, error.issue)
            cost_failure = await self._failure(name, "costs", config, now, error.issue)
            return usage_failure + cost_failure

        alert = None
        expiry = config.reporting_key_expires_at
        if expiry is not None and expiry - now <= _EXPIRY_WARNING:
            alert = ReportingIssue.EXPIRING_CREDENTIAL
            _LOG.warning("reporting credential alert: %s", alert.value)
        admin = OpenAIAdmin(
            http,
            config.reporting_base_url,
            scope,
            credential,
            config.reporting_lookback_days,
            redactor.text,
        )
        observations: list[QuotaObservation] = []
        for kind in ("usage", "costs"):
            try:
                report = await admin.fetch(kind, now)
            except ReportingError as error:
                observations.extend(
                    await self._failure(name, kind, config, now, error.issue, scope=scope)
                )
                continue
            observations.extend(
                await self._store.append(
                    (
                        QuotaObservation(
                            provider=name,
                            resource=kind,
                            unit="USD" if kind == "costs" else "tokens",
                            source="reporting:openai",
                            observed_at=now,
                            confidence="provider_report_may_lag",
                            status="verified",
                            issue=alert,
                            scope=scope,
                            window=report.window,
                            usage=report.usage,
                            costs=report.costs,
                        ),
                    ),
                    config.stale_after_seconds,
                    report.pages,
                )
            )
        return tuple(observations)
