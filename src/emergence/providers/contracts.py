"""Provider capability declarations and the gateway-facing inference contract."""

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import JsonValue

from emergence.config import ModelConfig, ProviderConfig, SafeModel
from emergence.inference.schemas import InferenceRequest, InferenceResult


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
