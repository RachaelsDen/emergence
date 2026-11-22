"""Shared chat-completion wire schemas, separate from normalized usage."""

from typing import Annotated, Self

from pydantic import BaseModel, Field, JsonValue, model_validator

from emergence.inference.schemas import TokenCount


class _PromptDetails(BaseModel):
    cached_tokens: TokenCount | None = None


class _CompletionDetails(BaseModel):
    reasoning_tokens: TokenCount | None = None


class _Usage(BaseModel):
    prompt_tokens: TokenCount | None = None
    completion_tokens: TokenCount | None = None
    total_tokens: TokenCount | None = None
    prompt_tokens_details: _PromptDetails | None = None
    completion_tokens_details: _CompletionDetails | None = None

    @model_validator(mode="after")
    def validate_total_tokens(self) -> Self:
        if (
            self.total_tokens is not None
            and self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens != self.prompt_tokens + self.completion_tokens
        ):
            raise ValueError("total tokens do not match prompt and completion tokens")
        return self


class _ResponseMessage(BaseModel):
    content: str | None = None
    tool_calls: list[JsonValue] = Field(default_factory=list[JsonValue])
    reasoning_content: str | None = None


class _Choice(BaseModel):
    message: _ResponseMessage
    finish_reason: str | None = None


class Completion(BaseModel):
    id: str | None = None
    model: str
    choices: Annotated[list[_Choice], Field(min_length=1, max_length=1)]
    usage: _Usage | None = None
