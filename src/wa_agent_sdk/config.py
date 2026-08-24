"""Declarative configuration for LLM providers and agent behaviour."""

from __future__ import annotations

import dataclasses
import difflib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ProviderAuthError, WaAgentError

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful WhatsApp assistant. Chat naturally and keep replies concise "
    "(this is a messaging app). Use plain language. When the user sends a document, its "
    "extracted text is provided to you: summarize it or answer questions about it faithfully. "
    "When the user sends an image, you can see it. Be honest when unsure."
)

_KNOWN_PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {"kind": "openai", "base_url": "https://api.openai.com/v1", "env": "OPENAI_API_KEY"},
    "groq": {"kind": "openai", "base_url": "https://api.groq.com/openai/v1", "env": "GROQ_API_KEY"},
    "deepseek": {"kind": "openai", "base_url": "https://api.deepseek.com/v1", "env": "DEEPSEEK_API_KEY"},
    "openrouter": {"kind": "openai", "base_url": "https://openrouter.ai/api/v1", "env": "OPENROUTER_API_KEY"},
    "together": {"kind": "openai", "base_url": "https://api.together.xyz/v1", "env": "TOGETHER_API_KEY"},
    "mistral": {"kind": "openai", "base_url": "https://api.mistral.ai/v1", "env": "MISTRAL_API_KEY"},
    "fireworks": {"kind": "openai", "base_url": "https://api.fireworks.ai/inference/v1", "env": "FIREWORKS_API_KEY"},
    "ollama": {"kind": "openai", "base_url": "http://localhost:11434/v1", "env": ""},
    "lmstudio": {"kind": "openai", "base_url": "http://localhost:1234/v1", "env": ""},
    "nvidia": {
        "kind": "openai",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env": "NVIDIA_API_KEY",
    },
    "xai": {"kind": "openai", "base_url": "https://api.x.ai/v1", "env": "XAI_API_KEY"},
    "grok": {"kind": "openai", "base_url": "https://api.x.ai/v1", "env": "XAI_API_KEY"},
    "anthropic": {"kind": "anthropic", "base_url": "https://api.anthropic.com", "env": "ANTHROPIC_API_KEY"},
    "gemini": {
        "kind": "gemini",
        "base_url": "https://generativelanguage.googleapis.com",
        "env": "GEMINI_API_KEY",
    },
}


def known_providers() -> list[str]:
    return sorted(_KNOWN_PROVIDERS)


@dataclass(frozen=True)
class LLMConfig:
    """Which AI provider powers the agent.

    Example:
        LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-...")
    """

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    request_timeout: float = 120.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not self.provider:
            raise WaAgentError("LLMConfig.provider must be set")
        if not self.model:
            raise WaAgentError("LLMConfig.model must be set")

    @property
    def info(self) -> dict[str, str]:
        meta = _KNOWN_PROVIDERS.get(self.provider.lower())
        if meta is None:
            matches = difflib.get_close_matches(self.provider.lower(), _KNOWN_PROVIDERS, n=3)
            hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
            known = ", ".join(known_providers())
            raise WaAgentError(
                f"Unknown provider {self.provider!r}.{hint} Known providers: {known}. "
                "Custom providers can be added with wa_agent_sdk.register_provider()."
            )
        return meta

    @property
    def kind(self) -> str:
        return self.info["kind"]

    @property
    def resolved_base_url(self) -> str:
        return (self.base_url or self.info["base_url"]).rstrip("/")

    @property
    def resolved_api_key(self) -> str:
        key = self.api_key or os.environ.get(self.info["env"], "")
        if not key and self.info["env"]:
            raise ProviderAuthError(
                f"No API key for provider '{self.provider}'. Pass LLMConfig(api_key=...) "
                f"or export {self.info['env']}."
            )
        return key


@dataclass(frozen=True)
class AgentConfig:
    """Everything that controls agent behaviour besides the LLM itself."""

    llm: LLMConfig
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    session_name: str = "default"
    sessions_dir: Path | None = None
    data_dir: Path | None = None

    provider_router: Any = None

    max_history_messages: int = 40
    max_context_chars: int = 60_000
    max_document_chars: int = 15_000
    max_image_side_px: int = 1600

    handle_images: bool = True
    handle_documents: bool = True
    enable_builtin_tools: bool = True
    max_tool_iterations: int = 6

    typing_indicator: bool = True
    mark_read: bool = True
    ignore_groups: bool = True
    allowed_chats: frozenset[str] | set[str] | list[str] | tuple[str, ...] | None = None

    human_batching: bool = True
    batch_window_seconds: float = 6.0
    batch_max_wait_seconds: float = 30.0

    enable_safety: bool = True
    require_trigger: str | None = None
    group_mention_only: bool = True
    reply_cooldown: float = 3.0
    global_hourly_limit: int = 80
    per_chat_daily_limit: int = 50
    new_chat_daily_limit: int = 12
    quiet_hours: tuple[str, str] | None = None
    humanize_min_delay: float = 0.8
    humanize_max_delay: float = 3.0
    campaign_min_delay: float = 6.0
    campaign_max_delay: float = 15.0

    qr_timeout: float = 180.0
    qr_max_attempts: int = 3
    max_bridge_restarts: int = 5
    log_level: str = "INFO"

    def resolved_sessions_dir(self) -> Path:
        d = self.sessions_dir or Path.cwd() / ".wa_sessions"
        return Path(d).expanduser().resolve()

    def resolved_data_dir(self) -> Path:
        d = self.data_dir or Path.cwd() / ".wa_data"
        return Path(d).expanduser().resolve()

    def is_chat_allowed(self, chat_jid: str) -> bool:
        if not self.allowed_chats:
            return True
        allowed = {c.split("@")[0].split(":")[0].lstrip("+") for c in self.allowed_chats}
        number = chat_jid.split("@")[0].split(":")[0].lstrip("+")
        return number in allowed or chat_jid in allowed_chats


def replace_config(cfg: AgentConfig, **overrides: object) -> AgentConfig:
    return dataclasses.replace(cfg, **overrides)
