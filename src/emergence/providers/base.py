"""Chat-completion transport shared only where the wire protocol is compatible."""

import json

import httpx
from anyio import fail_after
from pydantic import JsonValue, SecretStr, TypeAdapter, ValidationError

from emergence.config import ModelConfig, ProviderConfig, ProviderKind, Settings
from emergence.inference.schemas import InferenceRequest, InferenceResult, TokenUsage
from emergence.telemetry.logging import Redactor

from .contracts import FeatureBound as FeatureBound
from .contracts import Provider as Provider
from .contracts import ProviderFailure as ProviderFailure
from .wire import Completion


class _DuplicateMember(ValueError):
    pass


def _check_unique_members(pairs: list[tuple[str, object]]) -> None:
    if len({key for key, _ in pairs}) != len(pairs):
        raise _DuplicateMember


class ChatAdapter:
    """One injected HTTP client; no credential-bearing errors or automatic retries."""

    output_parameter: str = "max_tokens"
    mandatory_reasoning: bool = False
    default_reasoning_possible: bool = False
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
        input_bounded = model.input_bound_verified and bool(
            (model.input_bound_evidence or "").strip()
        )
        output_bounded = model.output_bound_verified and bool((model.bound_evidence or "").strip())
        reasoning_bounded = (
            output_bounded
            and model.reasoning_in_output_bound_verified
            and bool((model.reasoning_bound_evidence or "").strip())
        )
        return (
            FeatureBound(
                dimension="input",
                unit="token",
                bounded=input_bounded,
                mechanism="UTF-8 byte bound plus verified per-message protocol overhead",
                evidence=model.input_bound_evidence,
            ),
            FeatureBound(
                dimension="cached_input",
                unit="token",
                bounded=input_bounded,
                mechanism="reserve all input at uncached price; cache count overlaps input",
                evidence=model.input_bound_evidence,
                included_in="input",
            ),
            FeatureBound(
                dimension="output",
                unit="token",
                bounded=output_bounded,
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
            if request.reasoning or self.mandatory_reasoning or self.default_reasoning_possible:
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
        headers = {"Accept-Encoding": "identity"}
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
                    self.config.inference_url,
                    headers=self.headers(),
                    json=self.payload(request, model),
                    timeout=self.config.timeout_seconds,
                    follow_redirects=False,
                ) as response:
                    if response.headers.get("content-encoding", "identity").strip().lower() != (
                        "identity"
                    ):
                        raise ProviderFailure(
                            "provider_response_encoding_unsupported", ambiguous=True
                        )

                    def append(chunk: bytes) -> None:
                        remaining = self.config.response_limit_bytes - len(body)
                        if len(chunk) > remaining:
                            body.extend(chunk[:remaining])
                            raise ProviderFailure(
                                "provider_response_too_large",
                                ambiguous=True,
                                raw_response=self._redactor.text(
                                    body.decode("utf-8", errors="replace")
                                ),
                            )
                        body.extend(chunk)

                    if response.is_stream_consumed:
                        append(response.content)  # An injected transport may preload the body.
                    else:
                        async for chunk in response.aiter_raw():
                            append(chunk)
                    content = bytes(body)
                    status_code = response.status_code
        except (TimeoutError, httpx.RequestError):
            raise ProviderFailure(
                "provider_transport_error",
                ambiguous=True,
                raw_response=self._redactor.text(body.decode("utf-8", errors="replace")),
            ) from None
        try:
            # Check the original bounded body before Pydantic collapses
            # duplicate names. Preserve its numeric parsing for valid objects.
            json.loads(
                content, object_pairs_hook=_check_unique_members, parse_int=str, parse_float=str
            )
            raw = self._redactor.value(TypeAdapter(JsonValue).validate_json(content))
        except _DuplicateMember:
            # Preserve both members as redacted text, not a collapsed object.
            raise ProviderFailure(
                "provider_schema_error",
                ambiguous=True,
                status_code=status_code,
                raw_response=self._redactor.text(content.decode("utf-8", errors="replace")),
            ) from None
        except (ValueError, RecursionError):
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
