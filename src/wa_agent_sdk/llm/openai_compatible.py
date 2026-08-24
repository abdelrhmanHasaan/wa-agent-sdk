"""OpenAI-compatible provider.

Also covers Groq, DeepSeek, OpenRouter, Together, Mistral, Fireworks, Ollama
and LM Studio because they all expose the ``/chat/completions`` schema.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..exceptions import ProviderError
from .base import BaseChatProvider, ChatMessage, ChatResult, ToolCall


class OpenAICompatibleProvider(BaseChatProvider):
    name = "openai-compatible"
    supports_vision = True
    supports_tools = True

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.config.resolved_api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if "openrouter" in self.config.resolved_base_url:
            headers.setdefault("HTTP-Referer", "https://github.com/wa-agent-sdk")
            headers.setdefault("X-Title", "wa-agent-sdk")
        return headers

    def _convert(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "",
                        "content": msg.content or "",
                    }
                )
                continue
            if isinstance(msg.content, str) or msg.content is None:
                entry: dict[str, Any] = {
                    "role": msg.role,
                    "content": msg.content if msg.content is not None else "",
                }
                if msg.tool_calls:
                    entry["content"] = entry["content"] or None
                    entry["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.args)},
                        }
                        for c in msg.tool_calls
                    ]
                out.append(entry)
                continue

            parts: list[dict[str, Any]] = []
            text_chunks: list[str] = []
            for block in msg.content:
                if block.get("type") == "text":
                    text_chunks.append(block["text"])
                    parts.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image":
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{block['mime_type']};base64,{block['data']}"
                            },
                        }
                    )
            entry = {"role": msg.role, "content": parts or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.args)},
                    }
                    for c in msg.tool_calls
                ]
            out.append(entry)
        return out

    async def _chat_once(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._convert(messages),
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens
        if tools and self.supports_tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
                for t in tools
            ]

        url = f"{self.config.resolved_base_url}/chat/completions"
        try:
            data = await self._post_json(url, json_body=payload, headers=self._headers())
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"{self.name}: HTTP {exc.response.status_code}: {exc.response.text[:400]}"
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.name}: no choices in response: {str(data)[:400]}")
        message = choices[0].get("message") or {}

        tool_calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=raw.get("id") or "", name=fn.get("name", ""), args=args))

        return ChatResult(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=str(choices[0].get("finish_reason") or ""),
            input_tokens=int((data.get("usage") or {}).get("prompt_tokens") or 0),
            output_tokens=int((data.get("usage") or {}).get("completion_tokens") or 0),
        )
