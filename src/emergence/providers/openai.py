"""OpenAI inference protocol; Admin reporting credentials never enter this client."""

from pydantic import JsonValue

from emergence.config import ModelConfig
from emergence.inference.schemas import InferenceRequest
from emergence.providers.base import ChatAdapter


class OpenAIAdapter(ChatAdapter):
    output_parameter = "max_completion_tokens"
    supports_reasoning_control = True

    def payload(self, request: InferenceRequest, model: ModelConfig) -> dict[str, JsonValue]:
        payload = super().payload(request, model)
        if request.reasoning:
            payload["reasoning_effort"] = "low"
        return payload
