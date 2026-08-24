"""Anthropic Claude provider."""

from __future__ import annotations

from typing import Any

from ..exceptions import ProviderError
from .base import BaseChatProvider, ChatMessage, ChatResult, ToolCall

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseChatProvider):
    name = "anthropic"
    supports_vision = True
    supports_tools = True

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.config.resolved_api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def _convert(self, messages: list[ChatMessage]) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        def blocks_of(msg: ChatMessage) -> list[dict[str, Any]]:
            if isinstance(msg.content, str) or msg.content is None:
                out: list[dict[str, Any]] = []
                text = msg.content or ""
                if text:
                    out.append({"type": "text", "text": text})
                for c in msg.tool_calls or []:
                    out.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.args})
                return out
            out = []
            for block in msg.content:
                if block.get("type") == "text":
                    out.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image":
                    out.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block["mime_type"],
                                "data": block["data"],
                            },
                        }
                    )
            return out

        for msg in messages:
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system_parts.append(msg.content)
                continue
            if msg.role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": msg.content or "",
                            }
                        ],
                    }
                )
                continue
            role = "assistant" if msg.role == "assistant" else "user"
            content = blocks_of(msg)
            if not content and role == "user":
                content = [{"type": "text", "text": "(continue)"}]
            converted.append({"role": role, "content": content})

        if converted and converted[0]["role"] == "assistant":
            converted.insert(0, {"role": "user", "content": [{"type": "text", "text": "(start)"}]})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, converted

    async def _chat_once(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None,
    ) -> ChatResult:
        system, converted = self._convert(messages)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens or 2048,
            "messages": converted,
            "temperature": self.config.temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]

        url = f"{self.config.resolved_base_url}/v1/messages"
        data = await self._post_json(url, json_body=payload, headers=self._headers())

        content_blocks = data.get("content") or []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        args=block.get("input") or {},
                    )
                )

        usage = data.get("usage") or {}
        result = ChatResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=str(data.get("stop_reason") or ""),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
        )
        if usage.get("is_error"):
            raise ProviderError(f"{self.name}: error response: {str(data)[:400]}")
        return result
