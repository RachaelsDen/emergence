"""Z.ai coding-plan chat protocol; endpoint and entitlement remain configuration."""

from pydantic import JsonValue

from emergence.config import ModelConfig
from emergence.inference.schemas import InferenceRequest
from emergence.providers.base import ChatAdapter


class ZaiAdapter(ChatAdapter):
    supports_reasoning_control = True

    def payload(self, request: InferenceRequest, model: ModelConfig) -> dict[str, JsonValue]:
        payload = super().payload(request, model)
        payload["thinking"] = {"type": "enabled" if request.reasoning else "disabled"}
        return payload
