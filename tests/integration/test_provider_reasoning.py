"""Reasoning defaults require a shared billable cap before durable admission."""

from pathlib import Path

import httpx
import pytest
from _gateway_fixture import answer, gateway_for, invocation, settings_for
from pydantic import JsonValue, TypeAdapter
from sqlalchemy import func, select
from test_migrations import migrated_path as migrated_path

from emergence.config import ProviderKind
from emergence.inference.gateway import RequestState
from emergence.providers.base import ProviderFailure
from emergence.storage.models import request_reservations, requests


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("kind", "reasoning"),
    [
        (ProviderKind.OPENAI, False),
        (ProviderKind.OPENAI, True),
        (ProviderKind.MINIMAX, False),
        (ProviderKind.MINIMAX, True),
        (ProviderKind.ZAI, True),
    ],
)
@pytest.mark.parametrize("enforce_budgets", [True, False], ids=["enforced", "shadow"])
@pytest.mark.parametrize(
    ("verified", "evidence"),
    [(False, "synthetic contract"), (True, None), (True, ""), (True, " \t\n")],
)
async def test_unverified_reasoning_rejects_before_admission(
    migrated_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: ProviderKind,
    reasoning: bool,
    enforce_budgets: bool,
    verified: bool,
    evidence: str | None,
) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-inference-key")
    original = settings_for(kind)
    provider = original.providers["p"]
    model = provider.models["m"].model_copy(
        update={
            "reasoning_in_output_bound_verified": verified,
            "reasoning_bound_evidence": evidence,
        }
    )
    settings = original.model_copy(
        update={
            "enforce_budgets": enforce_budgets,
            "providers": {"p": provider.model_copy(update={"models": {"m": model}})},
        }
    )
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    async with gateway_for(migrated_path, settings, respond) as (gateway, _, database, _):
        with pytest.raises(ProviderFailure, match="unverified_billable_bound") as failure:
            await gateway.infer(
                invocation().model_copy(update={"reasoning": reasoning}), "unverified-reasoning"
            )
        assert not failure.value.ambiguous
        assert calls == []
        features = gateway.registry.get("p").features(model)
        assert not next(feature for feature in features if feature.dimension == "reasoning").bounded
        async with database.session() as session:
            assert await session.scalar(select(func.count()).select_from(requests)) == 0
            assert await session.scalar(select(func.count()).select_from(request_reservations)) == 0


@pytest.mark.anyio
@pytest.mark.parametrize("reasoning", [False, True], ids=["default", "explicit"])
async def test_verified_openai_reasoning_uses_shared_output_cap(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch, reasoning: bool
) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-inference-key")
    settings = settings_for(ProviderKind.OPENAI)
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        payload = TypeAdapter(dict[str, JsonValue]).validate_json(request.content)
        if (
            payload.get("max_completion_tokens") != 10
            or "max_tokens" in payload
            or payload.get("model") != "configured-model"
            or (
                payload.get("reasoning_effort") != "low"
                if reasoning
                else "reasoning_effort" in payload
            )
        ):
            return httpx.Response(400, text="invalid reasoning bound")
        return httpx.Response(
            200,
            json={
                "model": "configured-model",
                "choices": [{"message": {"content": "bounded reasoning"}}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "completion_tokens_details": {"reasoning_tokens": 7},
                },
            },
        )

    async with gateway_for(migrated_path, settings, respond) as (gateway, ledger, database, _):
        result = await gateway.infer(
            invocation().model_copy(update={"reasoning": reasoning}), "verified-reasoning"
        )
        assert result.state is RequestState.COMPLETED
        assert result.result is not None
        assert result.result.usage.output_tokens == 10
        assert result.result.usage.reasoning_tokens == 7
        assert calls == ["/v1/chat/completions"]
        features = gateway.registry.get("p").features(settings.providers["p"].models["m"])
        reasoning_bound = next(feature for feature in features if feature.dimension == "reasoning")
        assert reasoning_bound.bounded
        assert reasoning_bound.included_in == "output"
        assert reasoning_bound.mechanism == "max_completion_tokens"
        assert (await ledger.balance("p")).reserved == 0
        async with database.session() as session:
            assert await session.scalar(select(func.count()).select_from(requests)) == 1
            assert await session.scalar(select(func.count()).select_from(request_reservations)) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("kind", [ProviderKind.ZAI, ProviderKind.LLAMACPP])
async def test_disabled_optional_reasoning_needs_no_reasoning_evidence(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch, kind: ProviderKind
) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-inference-key")
    original = settings_for(kind)
    provider = original.providers["p"]
    model = provider.models["m"].model_copy(
        update={"reasoning_in_output_bound_verified": False, "reasoning_bound_evidence": None}
    )
    settings = original.model_copy(
        update={"providers": {"p": provider.model_copy(update={"models": {"m": model}})}}
    )
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        payload = TypeAdapter(dict[str, JsonValue]).validate_json(request.content)
        if kind is ProviderKind.ZAI and payload.get("thinking") != {"type": "disabled"}:
            return httpx.Response(400, text="reasoning must be disabled")
        return answer()

    async with gateway_for(migrated_path, settings, respond) as (gateway, _, _, _):
        result = await gateway.infer(invocation(), "disabled-reasoning")

    assert result.state is RequestState.COMPLETED
    assert calls == ["/v1/chat/completions"]
