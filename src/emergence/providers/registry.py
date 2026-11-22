"""Configured provider/model identities with no fallback selection."""

import httpx

from emergence.config import ProviderKind, Settings
from emergence.providers.base import Provider, ProviderFailure
from emergence.providers.llamacpp import LlamaCppAdapter
from emergence.providers.minimax import MinimaxAdapter
from emergence.providers.openai import OpenAIAdapter
from emergence.providers.zai import ZaiAdapter


class Registry:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.providers: dict[str, Provider] = {}
        for name, config in settings.providers.items():
            if not config.enabled:
                continue
            match config.kind:
                case ProviderKind.ZAI:
                    adapter = ZaiAdapter(name, config, settings, client)
                case ProviderKind.MINIMAX:
                    adapter = MinimaxAdapter(name, config, settings, client)
                case ProviderKind.OPENAI:
                    adapter = OpenAIAdapter(name, config, settings, client)
                case ProviderKind.LLAMACPP:
                    adapter = LlamaCppAdapter(name, config, settings, client)
            self.providers[name] = adapter

    def get(self, name: str) -> Provider:
        provider = self.providers.get(name)
        if provider is None:
            raise ProviderFailure("provider_unavailable")
        return provider
