"""Resolved reporting credentials stay isolated from API and inference roles."""

from datetime import datetime

from emergence.config import ProviderConfig, ProviderKind, Settings

from .schemas import ReportingError, ReportingIssue


def reporting_credential(settings: Settings, config: ProviderConfig, now: datetime) -> str:
    if config.reporting_key_expires_at is not None and now >= config.reporting_key_expires_at:
        raise ReportingError(ReportingIssue.EXPIRED_CREDENTIAL)
    name = config.reporting_credential_env
    if config.kind is not ProviderKind.OPENAI and name is None:
        name = config.credential_env
    secret = settings.secret(name)
    if secret is None:
        raise ReportingError(ReportingIssue.MISSING_CREDENTIAL)
    if any(
        secret == settings.secret(role)
        for role in (settings.operator_token_env, settings.reader_token_env)
    ):
        raise ReportingError(ReportingIssue.SHARED_CREDENTIAL)
    if config.kind is ProviderKind.OPENAI and any(
        secret == settings.secret(provider.credential_env)
        for provider in settings.providers.values()
    ):
        raise ReportingError(ReportingIssue.SHARED_CREDENTIAL)
    if config.kind is not ProviderKind.OPENAI and any(
        secret == settings.secret(provider.reporting_credential_env)
        for provider in settings.providers.values()
        if provider.kind is ProviderKind.OPENAI
    ):
        raise ReportingError(ReportingIssue.SHARED_CREDENTIAL)
    return secret.get_secret_value()
