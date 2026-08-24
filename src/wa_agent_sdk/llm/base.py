"""Abstract LLM provider interface plus shared message/result types."""

from __future__ import annotations

import abc
import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from ..config import LLMConfig
from ..exceptions import ProviderAuthError, ProviderError


@dataclass(slots=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatMessage:
    """Canonical chat message.

    ``content`` is either a plain string or a list of blocks produced by the
    helpers :func:`text_block` / :func:`image_block`. Tool responses use
    ``role="tool"`` together with ``tool_call_id`` and ``name``.
    """

    role: str
    content: Any = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_block(mime_type: str, data_b64: str) -> dict[str, Any]:
    return {"type": "image", "mime_type": mime_type, "data": data_b64}


@dataclass(slots=True)
class ChatResult:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class BaseChatProvider(abc.ABC):
    """Base class every provider backend implements.

    Subclasses only implement :meth:`_chat_once`; this base class adds retry
    handling with exponential backoff and consistent error mapping.
    """

    name: str = "base"
    supports_vision: bool = True
    supports_tools: bool = True

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.request_timeout, connect=15.0)
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        attempts = max(1, self.config.max_retries)
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await self._chat_once(messages, tools)
            except ProviderError as exc:
                last_error = exc
                retryable = exc.retryable
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                retryable = True

            if not retryable or attempt == attempts:
                raise last_error  # type: ignore[misc]
            sleep_for = delay * (1.0 + random.random() * 0.3)
            await asyncio.sleep(sleep_for)
            delay *= 2
        raise last_error  # type: ignore[misc]

    @abc.abstractmethod
    async def _chat_once(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None,
    ) -> ChatResult: ...

    async def _post_json(
        self,
        url: str,
        *,
        json_body: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        try:
            resp = await self.http.post(url, json=json_body, headers=headers)
        except httpx.HTTPStatusError:
            raise
        status = resp.status_code
        if status >= 400:
            body = resp.text[:600]
            if status in (401, 403):
                raise ProviderAuthError(
                    f"{self.name}: authentication failed ({status}). Check your API key. {body}",
                    status=status,
                )
            raise ProviderError(
                f"{self.name}: HTTP {status}: {body}",
                retryable=status in RETRYABLE_STATUS,
                status=status,
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response: {resp.text[:400]}") from exc


ProviderFactory = Callable[[LLMConfig], "BaseChatProvider"]
