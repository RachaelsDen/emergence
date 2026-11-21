"""Chat-completion transport shared only where the wire protocol is compatible."""

from dataclasses import dataclass
from typing import Literal, Protocol

import httpx
from anyio import fail_after
from pydantic import JsonValue, SecretStr, TypeAdapter, ValidationError

from emergence.config import ModelConfig, ProviderConfig, ProviderKind, SafeModel, Settings
from emergence.inference.schemas import InferenceRequest, InferenceResult, TokenUsage
from emergence.telemetry.logging import Redactor

from .wire import Completion


class FeatureBound(SafeModel):
    dimension: Literal["input", "cached_input", "output", "reasoning", "tools", "extras"]
    unit: str
    bounded: bool
    mechanism: str
    evidence: str | None
    included_in: str | None = None


@dataclass(slots=True)
class ProviderFailure(Exception):
    code: str
    ambiguous: bool = False
    status_code: int | None = None
    raw_response: JsonValue = None

    def __str__(self) -> str:
        return self.code


class Provider(Protocol):
    name: str
    config: ProviderConfig

    def features(self, model: ModelConfig) -> tuple[FeatureBound, ...]: ...
    def preflight(self, request: InferenceRequest, *, cost_enforced: bool) -> ModelConfig: ...
    async def infer(self, request: InferenceRequest) -> InferenceResult: ...


class ChatAdapter:
    """One injected HTTP client; no credential-bearing errors or automatic retries."""

    output_parameter: str = "max_tokens"
    mandatory_reasoning: bool = False
    supports_reasoning_control: bool = False

    def __init__(
        self, name: str, config: ProviderConfig, settings: Settings, client: httpx.AsyncClient
    ) -> None:
        self.name = name
        self.config = config
        self._settings = settings
        self._client = client
        self._redactor = Redactor(settings.secrets())

    def features(self, model: ModelConfig) -> tuple[FeatureBound, ...]:
        """Declare token overlap and which provider limits have supporting evidence."""
        reasoning_bounded = (
            model.output_bound_verified
            and model.reasoning_in_output_bound_verified
            and bool(model.reasoning_bound_evidence)
        )
        return (
            FeatureBound(
                dimension="input",
                unit="token",
                bounded=model.input_bound_verified and bool(model.input_bound_evidence),
                mechanism="UTF-8 byte bound plus verified per-message protocol overhead",
                evidence=model.input_bound_evidence,
            ),
            FeatureBound(
                dimension="cached_input",
                unit="token",
                bounded=model.input_bound_verified and bool(model.input_bound_evidence),
                mechanism="reserve all input at uncached price; cache count overlaps input",
                evidence=model.input_bound_evidence,
                included_in="input",
            ),
            FeatureBound(
                dimension="output",
                unit="token",
                bounded=model.output_bound_verified and bool(model.bound_evidence),
                mechanism=self.output_parameter,
                evidence=model.bound_evidence,
            ),
            FeatureBound(
                dimension="reasoning",
                unit="token",
                bounded=reasoning_bounded,
                mechanism=self.output_parameter,
                evidence=model.reasoning_bound_evidence,
                included_in="output",
            ),
            FeatureBound(
                dimension="tools", unit="call", bounded=False, mechanism="disabled", evidence=None
            ),
            FeatureBound(
                dimension="extras",
                unit="vendor",
                bounded=False,
                mechanism="disabled",
                evidence=None,
            ),
        )

    def preflight(self, request: InferenceRequest, *, cost_enforced: bool) -> ModelConfig:
        model = self.config.models.get(request.model)
        if not self.config.enabled or model is None:
            raise ProviderFailure("model_unavailable")
        if request.tools or request.extensions:
            raise ProviderFailure("unbounded_feature")
        if request.reasoning and not (self.supports_reasoning_control or self.mandatory_reasoning):
            raise ProviderFailure("unsupported_reasoning_control")
        if request.max_output_tokens > model.max_output_tokens:
            raise ProviderFailure("output_limit_exceeded")
        input_bound = sum(
            len(message.content.encode("utf-8")) + model.input_overhead_tokens
            for message in request.messages
        )
        if input_bound > model.max_input_tokens:
            raise ProviderFailure("input_limit_exceeded")
        if cost_enforced:
            required = {"input", "output"}
            if request.reasoning or self.mandatory_reasoning:
                required.add("reasoning")
            if any(
                not feature.bounded
                for feature in self.features(model)
                if feature.dimension in required
            ):
                raise ProviderFailure("unverified_billable_bound")
        self._inference_credential()
        return model

    def payload(self, request: InferenceRequest, model: ModelConfig) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "model": model.id,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            self.output_parameter: request.max_output_tokens,
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload

    def _inference_credential(self) -> SecretStr | None:
        """Resolve a header-safe credential isolated from API and reporting roles."""
        credential = self._settings.secret(self.config.credential_env)
        if credential is not None and any(
            not "!" <= char <= "~" for char in credential.get_secret_value()
        ):
            raise ProviderFailure("invalid_inference_credential")
        protected = (
            self._settings.operator_token_env,
            self._settings.reader_token_env,
            *(
                provider.reporting_credential_env
                for provider in self._settings.providers.values()
                if provider.kind is ProviderKind.OPENAI
            ),
        )
        if credential is not None and any(
            credential == self._settings.secret(name) for name in protected
        ):
            raise ProviderFailure("shared_inference_credential")
        return credential

    def headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        # Use the exact value just checked, not a second lookup after preflight.
        credential = self._inference_credential()
        if credential is not None:
            headers["Authorization"] = f"Bearer {credential.get_secret_value()}"
        if self.config.kind is ProviderKind.OPENAI:
            if self.config.organization:
                headers["OpenAI-Organization"] = self.config.organization
            if self.config.project:
                headers["OpenAI-Project"] = self.config.project
        return headers

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        model = self.preflight(request, cost_enforced=False)
        body = bytearray()
        try:
            with fail_after(self.config.timeout_seconds):
                async with self._client.stream(
                    "POST",
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers=self.headers(),
                    json=self.payload(request, model),
                    timeout=self.config.timeout_seconds,
                    follow_redirects=False,
                ) as response:
                    async for chunk in response.aiter_bytes():
                        if len(body) + len(chunk) > self.config.response_limit_bytes:
                            remaining = self.config.response_limit_bytes - len(body)
                            body.extend(chunk[:remaining])
                            raise ProviderFailure(
                                "provider_response_too_large",
                                ambiguous=True,
                                raw_response=self._redactor.text(
                                    body.decode("utf-8", errors="replace")
                                ),
                            )
                        body.extend(chunk)
                    content = bytes(body)
                    status_code = response.status_code
        except (TimeoutError, httpx.RequestError):
            raise ProviderFailure(
                "provider_transport_error",
                ambiguous=True,
                raw_response=self._redactor.text(body.decode("utf-8", errors="replace")),
            ) from None
        try:
            raw = self._redactor.value(TypeAdapter(JsonValue).validate_json(content))
        except ValidationError:
            raw = self._redactor.text(content.decode("utf-8", errors="replace"))
        if not 200 <= status_code < 300:
            # Auth/rate-limit rejection is known not to be a successful inference;
            # other errors are conservative because providers can fail after work.
            raise ProviderFailure(
                "provider_rejected",
                ambiguous=status_code not in {400, 401, 403, 404, 422, 429},
                status_code=status_code,
                raw_response=raw,
            )
        try:
            completion = Completion.model_validate(raw)
            usage = completion.usage
            normalized = TokenUsage(
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
                cached_input_tokens=(
                    usage.prompt_tokens_details.cached_tokens
                    if usage and usage.prompt_tokens_details
                    else None
                ),
                reasoning_tokens=(
                    usage.completion_tokens_details.reasoning_tokens
                    if usage and usage.completion_tokens_details
                    else None
                ),
            )
        except ValidationError:
            raise ProviderFailure(
                "provider_schema_error", ambiguous=True, raw_response=raw
            ) from None
        if completion.model != model.id:
            raise ProviderFailure("provider_model_mismatch", ambiguous=True, raw_response=raw)
        choice = completion.choices[0]
        if choice.message.tool_calls:
            raise ProviderFailure("unexpected_tool_response", ambiguous=True, raw_response=raw)
        raw_usage: JsonValue = None
        match raw:
            case dict():
                raw_usage = raw.get("usage")
            case _:
                raise ProviderFailure("provider_schema_error", ambiguous=True, raw_response=raw)
        return InferenceResult(
            provider=self.name,
            model=request.model,
            upstream_model=completion.model,
            upstream_request_id=completion.id,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=normalized,
            raw_usage=raw_usage,
            raw_response=raw,
        )
