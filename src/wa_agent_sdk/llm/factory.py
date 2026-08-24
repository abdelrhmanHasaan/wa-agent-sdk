"""Provider registry and factory."""

from __future__ import annotations

import difflib
from typing import Callable, Type

from ..config import LLMConfig
from ..exceptions import ProviderError
from .anthropic import AnthropicProvider
from .base import BaseChatProvider, ChatMessage, ChatResult, ToolCall, image_block, text_block
from .gemini import GeminiProvider
from .openai_compatible import OpenAICompatibleProvider

_REGISTRY: dict[str, Type[BaseChatProvider]] = {}


def register_provider(kind: str) -> Callable[[Type[BaseChatProvider]], Type[BaseChatProvider]]:
    """Class decorator to plug a custom provider backend into the SDK."""

    def deco(cls: Type[BaseChatProvider]) -> Type[BaseChatProvider]:
        _REGISTRY[kind.lower()] = cls
        return cls

    return deco


for _kind, _cls in (
    ("openai", OpenAICompatibleProvider),
    ("groq", OpenAICompatibleProvider),
    ("deepseek", OpenAICompatibleProvider),
    ("openrouter", OpenAICompatibleProvider),
    ("together", OpenAICompatibleProvider),
    ("mistral", OpenAICompatibleProvider),
    ("fireworks", OpenAICompatibleProvider),
    ("nvidia", OpenAICompatibleProvider),
    ("xai", OpenAICompatibleProvider),
    ("grok", OpenAICompatibleProvider),
    ("ollama", OpenAICompatibleProvider),
    ("lmstudio", OpenAICompatibleProvider),
    ("anthropic", AnthropicProvider),
    ("claude", AnthropicProvider),
    ("gemini", GeminiProvider),
    ("google", GeminiProvider),
):
    _REGISTRY[_kind] = _cls
    _cls.name = _cls.name  # keep explicit names


def create_provider(config: LLMConfig) -> BaseChatProvider:
    kind = config.kind
    cls = _REGISTRY.get(kind)
    if cls is None:
        matches = difflib.get_close_matches(kind, sorted(_REGISTRY), n=3)
        hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
        raise ProviderError(f"No provider backend registered for kind '{kind}'.{hint}")
    return cls(config)


def known_provider_kinds() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "BaseChatProvider",
    "ChatMessage",
    "ChatResult",
    "ToolCall",
    "image_block",
    "text_block",
    "register_provider",
    "create_provider",
    "known_provider_kinds",
]
