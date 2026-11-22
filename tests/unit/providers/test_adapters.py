"""Synthetic wire fixtures test normalization, feature admission, and error semantics."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

import httpx
import pytest
from anyio import fail_after, sleep_forever
from pydantic import JsonValue, TypeAdapter

from emergence.config import ModelConfig, ProviderConfig, ProviderKind, QuotaPolicy, Settings
from emergence.inference.schemas import InferenceRequest, Message
from emergence.providers.base import ChatAdapter, ProviderFailure
from emergence.providers.registry import Registry

FIXTURE = Path(__file__).parents[2] / "fixtures" / "providers" / "chat.json"


def configured(kind: ProviderKind) -> Settings:
    return Settings(
        providers={
            "provider": ProviderConfig(
                kind=kind,
                base_url="https://fixture.invalid/v1",
                quota_policy=QuotaPolicy.LOCAL_LIMITS_ONLY,
                credential_env="FIXTURE_INFERENCE_KEY",
                reporting_credential_env="FIXTURE_REPORTING_KEY",
                models={
                    "alias": ModelConfig(
                        id="configured-model",
                        max_output_tokens=32,
                        input_bound_verified=True,
                        input_bound_evidence="synthetic fixture byte/overhead contract",
                        output_bound_verified=True,
                        bound_evidence="synthetic fixture completion limit",
                        reasoning_in_output_bound_verified=True,
                        reasoning_bound_evidence="synthetic fixture includes reasoning in output",
                    )
                },
            )
        }
    )


def request() -> InferenceRequest:
    return InferenceRequest(
        provider="provider",
        model="alias",
        messages=(Message(role="user", content="hello"),),
        max_output_tokens=16,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("kind", list(ProviderKind))
async def test_adapter_normalizes_usage_and_preserves_vendor_metadata(
    kind: ProviderKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIXTURE_INFERENCE_KEY", "inference-secret")
    monkeypatch.setenv("FIXTURE_REPORTING_KEY", "reporting-secret")

    def respond(incoming: httpx.Request) -> httpx.Response:
        # This wire server enforces auth and the vendor-specific output parameter.
        payload = TypeAdapter(dict[str, JsonValue]).validate_json(incoming.content)
        expected_parameter = (
            "max_completion_tokens"
            if kind in {ProviderKind.OPENAI, ProviderKind.MINIMAX}
            else "max_tokens"
        )
        if (
            incoming.headers.get("Authorization") != "Bearer inference-secret"
            or payload.get(expected_parameter) != 16
            or payload.get("model") != "configured-model"
        ):
            return httpx.Response(400, text="invalid protocol")
        return httpx.Response(200, content=FIXTURE.read_bytes())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = Registry(configured(kind), client).get("provider")
        provider.preflight(request(), cost_enforced=True)
        result = await provider.infer(request())
    assert result.content == "fixture answer"
    assert result.usage.input_tokens == 24
    assert result.usage.cached_input_tokens == 9
    assert result.usage.output_tokens == 7
    assert result.usage.reasoning_tokens == 2
    assert result.raw_usage == {
        "prompt_tokens": 24,
        "completion_tokens": 7,
        "total_tokens": 31,
        "prompt_tokens_details": {"cached_tokens": 9},
        "completion_tokens_details": {"reasoning_tokens": 2},
        "vendor_detail": "preserved",
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "ambiguous"),
    [(401, False), (429, False), (500, True), (200, True)],
)
async def test_http_and_non_json_failures_have_honest_charge_uncertainty(
    status: int, ambiguous: bool
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, text="not JSON"))
    ) as client:
        provider = Registry(configured(ProviderKind.ZAI), client).get("provider")
        with pytest.raises(ProviderFailure) as failure:
            await provider.infer(request())
    assert failure.value.ambiguous is ambiguous
    assert failure.value.code == ("provider_schema_error" if status == 200 else "provider_rejected")


@pytest.mark.anyio
async def test_timeout_does_not_retry_or_claim_free_usage() -> None:
    calls: list[str] = []

    def timeout(incoming: httpx.Request) -> httpx.Response:
        calls.append(incoming.url.path)
        raise httpx.ReadTimeout("secret-bearing remote failure", request=incoming)

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
        with pytest.raises(ProviderFailure) as failure:
            await Registry(configured(ProviderKind.ZAI), client).get("provider").infer(request())
    assert calls == ["/v1/chat/completions"]
    assert failure.value.code == "provider_transport_error"
    assert failure.value.ambiguous
    assert "secret-bearing" not in str(failure.value)


@pytest.mark.anyio
async def test_missing_usage_stays_unknown() -> None:
    response = {
        "model": "configured-model",
        "choices": [{"message": {"content": "answer"}}],
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        result = (
            await Registry(configured(ProviderKind.ZAI), client).get("provider").infer(request())
        )
    assert not result.usage.complete
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None


@pytest.mark.anyio
@pytest.mark.parametrize("feature", ["tools", "extensions", "bound"])
async def test_uncertain_billable_features_reject_before_dispatch(
    feature: Literal["tools", "extensions", "bound"],
) -> None:
    calls: list[str] = []

    def respond(incoming: httpx.Request) -> httpx.Response:
        calls.append(incoming.url.path)
        return httpx.Response(200, content=FIXTURE.read_bytes())

    config = configured(ProviderKind.ZAI)
    invocation = request()
    match feature:
        case "tools":
            invocation = request().model_copy(update={"tools": ("tool",)})
        case "extensions":
            invocation = request().model_copy(update={"extensions": {"search": True}})
        case "bound":
            original = config.providers["provider"]
            model = original.models["alias"].model_copy(update={"output_bound_verified": False})
            config = config.model_copy(
                update={
                    "providers": {
                        "provider": original.model_copy(update={"models": {"alias": model}})
                    }
                }
            )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ProviderFailure):
            Registry(config, client).get("provider").preflight(invocation, cost_enforced=True)
    assert calls == []


@pytest.mark.anyio
async def test_provider_raw_payload_is_redacted_before_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FIXTURE_INFERENCE_KEY", "SENTINEL_SECRET")
    response = {
        "model": "configured-model",
        "choices": [{"message": {"content": "echo SENTINEL_SECRET"}}],
        "api_key": "unknown-sensitive-value",
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        result = (
            await Registry(configured(ProviderKind.ZAI), client).get("provider").infer(request())
        )
    assert result.content == "echo [REDACTED]"
    assert "SENTINEL_SECRET" not in result.model_dump_json()
    assert "unknown-sensitive-value" not in result.model_dump_json()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid_usage",
    [
        {"prompt_tokens": -1},
        {"prompt_tokens": "not-a-count"},
        {"prompt_tokens": False, "completion_tokens": False},
        {"completion_tokens": "0"},
        {"prompt_tokens": 0.0},
        {"completion_tokens": []},
        {"total_tokens": "0"},
        {"prompt_tokens_details": {"cached_tokens": False}},
        {"completion_tokens_details": {"reasoning_tokens": "0"}},
    ],
)
async def test_usage_schema_drift_retains_an_ambiguous_outcome(
    invalid_usage: dict[str, JsonValue],
) -> None:
    response = {
        "model": "configured-model",
        "choices": [{"message": {"content": "answer"}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, **invalid_usage},
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        with pytest.raises(ProviderFailure) as failure:
            await Registry(configured(ProviderKind.ZAI), client).get("provider").infer(request())
    assert failure.value.code == "provider_schema_error"
    assert failure.value.ambiguous
    evidence_usage = {
        field: str(value) if isinstance(value, float) else value
        for field, value in invalid_usage.items()
    }
    assert failure.value.raw_response == {
        **response,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, **evidence_usage},
    }


@pytest.mark.anyio
async def test_integer_zero_is_complete_usage() -> None:
    response = {
        "model": "configured-model",
        "choices": [{"message": {"content": "answer"}}],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        result = (
            await Registry(configured(ProviderKind.ZAI), client).get("provider").infer(request())
        )
    assert result.usage.complete
    assert (
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.cached_input_tokens,
        result.usage.reasoning_tokens,
    ) == (0, 0, 0, 0)


@pytest.mark.anyio
async def test_header_resolution_rechecks_after_inference_key_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = "synthetic-rotation-admin"
    monkeypatch.setenv("FIXTURE_INFERENCE_KEY", "synthetic-rotation-inference")
    monkeypatch.setenv("FIXTURE_REPORTING_KEY", shared)
    settings = configured(ProviderKind.OPENAI)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=FIXTURE.read_bytes()))
    ) as client:
        adapter = ChatAdapter("provider", settings.providers["provider"], settings, client)
        adapter.preflight(request(), cost_enforced=True)
        monkeypatch.setenv("FIXTURE_INFERENCE_KEY", shared)
        with pytest.raises(ProviderFailure, match="shared_inference_credential") as failure:
            adapter.headers()
    assert shared not in str(failure.value)


@pytest.mark.anyio
async def test_unexpected_tool_output_cannot_be_treated_as_a_plain_completion() -> None:
    response = {
        "model": "configured-model",
        "choices": [{"message": {"tool_calls": [{"function": {"name": "unrequested"}}]}}],
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        with pytest.raises(ProviderFailure) as failure:
            await Registry(configured(ProviderKind.ZAI), client).get("provider").infer(request())
    assert failure.value.code == "unexpected_tool_response"
    assert failure.value.ambiguous


@pytest.mark.anyio
async def test_oversized_response_is_bounded_and_reports_truncation() -> None:
    config = configured(ProviderKind.ZAI)
    config = config.model_copy(
        update={
            "providers": {
                "provider": config.providers["provider"].model_copy(
                    update={"response_limit_bytes": 16}
                )
            }
        }
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"x" * 64))
    ) as client:
        with pytest.raises(ProviderFailure) as failure:
            await Registry(config, client).get("provider").infer(request())
    assert failure.value.code == "provider_response_too_large"
    assert failure.value.ambiguous
    assert failure.value.raw_response == "x" * 16


@pytest.mark.anyio
@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
async def test_nonidentity_response_is_rejected_without_reading_body(encoding: str) -> None:
    class UnreadableBody(httpx.AsyncByteStream):
        def __aiter__(self) -> AsyncIterator[bytes]:
            raise AssertionError("the encoded stream must not be read")

    def respond(incoming: httpx.Request) -> httpx.Response:
        assert incoming.headers["accept-encoding"] == "identity"
        return httpx.Response(200, headers={"Content-Encoding": encoding}, stream=UnreadableBody())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(
            ProviderFailure, match="provider_response_encoding_unsupported"
        ) as caught:
            await Registry(configured(ProviderKind.ZAI), client).get("provider").infer(request())
    assert caught.value.ambiguous


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "body"),
    [
        (200, b'{"usage":{},"usage":null}'),
        (200, rb'{"usage":{"prompt_tokens":1,"prompt_\u0074okens":2}}'),
        (429, b'{"error":{"code":"rate_limit","code":"insufficient_quota"}}'),
    ],
)
async def test_duplicate_members_preserve_conflicting_text_as_ambiguous_evidence(
    status: int, body: bytes
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, content=body))
    ) as client:
        with pytest.raises(ProviderFailure, match="provider_schema_error") as caught:
            await Registry(configured(ProviderKind.ZAI), client).get("provider").infer(request())
    assert caught.value.ambiguous
    assert caught.value.raw_response == body.decode()


@pytest.mark.anyio
async def test_inference_has_a_total_deadline_independent_of_transport_phase_timeouts() -> None:
    config = configured(ProviderKind.ZAI)
    config = config.model_copy(
        update={
            "providers": {
                "provider": config.providers["provider"].model_copy(
                    update={"timeout_seconds": 0.01}
                )
            }
        }
    )

    async def stalled(_: httpx.Request) -> httpx.Response:
        await sleep_forever()
        raise AssertionError("stalled transport unexpectedly returned")

    async with httpx.AsyncClient(transport=httpx.MockTransport(stalled)) as client:
        with fail_after(0.5), pytest.raises(ProviderFailure, match="provider_transport_error"):
            await Registry(config, client).get("provider").infer(request())


@pytest.mark.anyio
@pytest.mark.parametrize("kind", [ProviderKind.ZAI, ProviderKind.OPENAI])
async def test_reasoning_request_reaches_the_supported_vendor_control(kind: ProviderKind) -> None:
    def respond(incoming: httpx.Request) -> httpx.Response:
        payload = TypeAdapter(dict[str, JsonValue]).validate_json(incoming.content)
        selected = (
            payload.get("thinking") == {"type": "enabled"}
            if kind is ProviderKind.ZAI
            else payload.get("reasoning_effort") == "low"
        )
        return httpx.Response(
            200,
            json={
                "model": "configured-model",
                "choices": [
                    {"message": {"content": "reasoning configured" if selected else "ordinary"}}
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = (
            await Registry(configured(kind), client)
            .get("provider")
            .infer(request().model_copy(update={"reasoning": True}))
        )
    assert result.content == "reasoning configured"


@pytest.mark.anyio
async def test_local_reasoning_control_is_explicitly_unsupported() -> None:
    calls: list[str] = []

    def respond(incoming: httpx.Request) -> httpx.Response:
        calls.append(incoming.url.path)
        return httpx.Response(200, content=FIXTURE.read_bytes())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ProviderFailure, match="unsupported_reasoning_control"):
            await (
                Registry(configured(ProviderKind.LLAMACPP), client)
                .get("provider")
                .infer(request().model_copy(update={"reasoning": True}))
            )
    assert calls == []
