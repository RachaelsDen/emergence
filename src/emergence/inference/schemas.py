"""Validated operator requests and lossless redacted provider observations."""

from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator
from pydantic_core import PydanticCustomError

from emergence.config import SafeModel

TokenCount = Annotated[int, Field(strict=True, ge=0)]


class Message(SafeModel):
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(min_length=1)]


class InferenceRequest(SafeModel):
    provider: Annotated[str, Field(min_length=1, max_length=128)]
    model: Annotated[str, Field(min_length=1, max_length=128)]
    messages: Annotated[tuple[Message, ...], Field(min_length=1, max_length=128)]
    max_output_tokens: Annotated[int, Field(gt=0, le=131072)] = 128
    temperature: Annotated[float | None, Field(ge=0, le=2)] = None
    reasoning: bool = False
    tools: tuple[JsonValue, ...] = ()
    extensions: dict[str, JsonValue] = Field(default_factory=dict)


class TokenUsage(SafeModel):
    input_tokens: TokenCount | None = None
    cached_input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    reasoning_tokens: TokenCount | None = None

    @model_validator(mode="after")
    def validate_overlapping_counts(self) -> Self:
        if (
            self.cached_input_tokens is not None
            and self.input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise PydanticCustomError("usage_counts", "cached count exceeds input count")
        if (
            self.reasoning_tokens is not None
            and self.output_tokens is not None
            and self.reasoning_tokens > self.output_tokens
        ):
            raise PydanticCustomError("usage_counts", "reasoning count exceeds output count")
        return self

    @property
    def complete(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None


class InferenceResult(SafeModel):
    provider: str
    model: str
    upstream_model: str
    upstream_request_id: str | None
    content: str
    finish_reason: str | None
    usage: TokenUsage
    raw_usage: JsonValue
    raw_response: JsonValue
