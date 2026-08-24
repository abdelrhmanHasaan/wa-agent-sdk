"""Google Gemini provider (generativeLanguage REST API)."""

from __future__ import annotations

from typing import Any

from ..exceptions import ProviderError
from .base import BaseChatProvider, ChatMessage, ChatResult, ToolCall


def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini rejects some JSON-Schema keywords; strip the unsafe ones."""
    if not isinstance(schema, dict):
        return schema
    clean = {
        k: v
        for k, v in schema.items()
        if k not in ("additionalProperties", "$defs", "$schema", "title", "examples")
    }
    for key, value in list(clean.items()):
        if isinstance(value, dict):
            clean[key] = _sanitize_schema(value)
        elif isinstance(value, list):
            clean[key] = [_sanitize_schema(v) if isinstance(v, dict) else v for v in value]
    return clean


class GeminiProvider(BaseChatProvider):
    name = "gemini"
    supports_vision = True
    supports_tools = True

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.config.resolved_api_key,
        }

    def _convert(self, messages: list[ChatMessage]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        system_instruction: dict[str, Any] | None = None
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                text = msg.content if isinstance(msg.content, str) else str(msg.content or "")
                system_instruction = {"parts": [{"text": text}]}
                continue

            if msg.role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.name or "tool",
                                    "response": {"result": msg.content or ""},
                                }
                            }
                        ],
                    }
                )
                continue

            role = "model" if msg.role == "assistant" else "user"
            parts: list[dict[str, Any]] = []
            if isinstance(msg.content, str) or msg.content is None:
                if msg.content:
                    parts.append({"text": msg.content})
            else:
                for block in msg.content:
                    if block.get("type") == "text":
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image":
                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": block["mime_type"],
                                    "data": block["data"],
                                }
                            }
                        )
            for call in msg.tool_calls or []:
                parts.append({"functionCall": {"name": call.name, "args": call.args}})
            if not parts:
                parts.append({"text": "(continue)"})
            contents.append({"role": role, "parts": parts})

        return system_instruction, contents

    async def _chat_once(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None,
    ) -> ChatResult:
        system_instruction, contents = self._convert(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": self.config.temperature},
        }
        if self.config.max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = self.config.max_tokens
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if tools and self.supports_tools:
            declarations = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": _sanitize_schema(
                        t.get("parameters", {"type": "object", "properties": {}})
                    ),
                }
                for t in tools
            ]
            payload["tools"] = [{"functionDeclarations": declarations}]

        model = self.config.model.removeprefix("models/")
        url = f"{self.config.resolved_base_url}/v1beta/models/{model}:generateContent"
        data = await self._post_json(url, json_body=payload, headers=self._headers())

        feedback = data.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise ProviderError(f"gemini: request blocked: {feedback['blockReason']}")

        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderError(f"gemini: no candidates in response: {str(data)[:400]}")
        parts = ((candidates[0].get("content") or {}).get("parts")) or []

        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        for idx, part in enumerate(parts):
            if "text" in part:
                text_chunks.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        id=f"gemini_{idx}",
                        name=fc.get("name", ""),
                        args=fc.get("args") or {},
                    )
                )

        return ChatResult(
            text="".join(text_chunks),
            tool_calls=tool_calls,
            finish_reason=str(candidates[0].get("finishReason") or ""),
        )
