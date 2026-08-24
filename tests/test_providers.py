"""LLM provider tests using httpx.MockTransport — no network needed."""

import asyncio
import json

import httpx
import pytest

from wa_agent_sdk import LLMConfig, WhatsAppAgent
from wa_agent_sdk.llm.anthropic import AnthropicProvider
from wa_agent_sdk.llm.base import ChatMessage, ChatResult, ToolCall, image_block, text_block
from wa_agent_sdk.llm.gemini import GeminiProvider
from wa_agent_sdk.llm.openai_compatible import OpenAICompatibleProvider

TOOLS_SCHEMA = [
    {
        "name": "calculate",
        "description": "math",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    }
]


def mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_openai_multimodal_and_tool_calls():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "calculate", "arguments": '{"expression": "6*7"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4},
        })

    prov = OpenAICompatibleProvider(LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test"))
    prov._client = mock_client(handler)
    msgs = [
        ChatMessage(role="system", content="be brief"),
        ChatMessage(role="user", content=[text_block("what's this?"), image_block("image/jpeg", "QUJD")]),
    ]
    res = prov.chat and __import__("asyncio").run(prov.chat(msgs, tools=TOOLS_SCHEMA))
    assert res.has_tool_calls and res.tool_calls[0].args == {"expression": "6*7"}
    assert res.input_tokens > 0
    assert captured["auth"] == "Bearer sk-test"
    part = captured["body"]["messages"][1]["content"][1]
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,QUJD")
    assert captured["body"]["tools"][0]["function"]["name"] == "calculate"


def test_anthropic_conversion_roundtrip():
    def handler(request):
        body = json.loads(request.content)
        assert request.headers.get("x-api-key") == "sk-ant"
        assert body["system"] == "be helpful"
        assert body["messages"][-1]["content"][0]["content"] == "42"
        return httpx.Response(200, json={
            "content": [
                {"type": "text", "text": "The answer is"},
                {"type": "tool_use", "id": "tu_9", "name": "calc", "input": {"x": 1}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 14, "output_tokens": 6},
        })

    prov = AnthropicProvider(LLMConfig(provider="anthropic", model="claude-sonnet-4-5", api_key="sk-ant"))
    prov._client = mock_client(handler)
    res = __import__("asyncio").run(prov.chat([
        ChatMessage(role="system", content="be helpful"),
        ChatMessage(role="user", content=[image_block("image/jpeg", "QUJD"), text_block("hi")]),
        ChatMessage(role="assistant", content=None, tool_calls=[ToolCall(id="tu_x", name="calc", args={})]),
        ChatMessage(role="tool", tool_call_id="tu_x", name="calc", content="42"),
    ], tools=TOOLS_SCHEMA))
    assert res.text == "The answer is" and res.tool_calls[0].name == "calc"
    assert (res.input_tokens, res.output_tokens) == (14, 6)


def test_gemini_function_response_shape():
    def handler(request):
        assert request.headers.get("x-goog-api-key") == "g-key"
        body = json.loads(request.content)
        fr = body["contents"][-1]["parts"][0]["functionResponse"]
        assert fr["response"]["result"] == "42"
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [
                {"text": "calc!"},
                {"functionCall": {"name": "calc", "args": {"x": 2}}},
            ]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 13, "candidatesTokenCount": 8},
        })

    prov = GeminiProvider(LLMConfig(provider="gemini", model="gemini-2.0-flash", api_key="g-key"))
    prov._client = mock_client(handler)
    res = asyncio.run(prov.chat([
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="assistant", content="", tool_calls=[ToolCall(id="i", name="calc", args={"x": 1})]),
        ChatMessage(role="tool", tool_call_id="i", name="calc", content="42"),
    ], tools=TOOLS_SCHEMA))
    assert res.text == "calc!" and res.tool_calls[0].args == {"x": 2}
    assert (res.input_tokens, res.output_tokens) == (13, 8)


def test_agent_tool_loop_executes_builtin_calculator():
    class ScriptedProvider(OpenAICompatibleProvider):
        def __init__(self):
            super().__init__(LLMConfig(provider="openai", model="m", api_key="k"))
            self.calls = 0

        async def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return ChatResult(tool_calls=[ToolCall(id="c1", name="calculate",
                                                       args={"expression": "(40+2)*10"})])
            assert any(m.role == "tool" and "420" in str(m.content) for m in messages)
            return ChatResult(text="It is 420.")

    agent = WhatsAppAgent(llm=LLMConfig(provider="openai", model="m", api_key="k"))
    agent._provider = ScriptedProvider()
    agent.memory.append("chat1", ChatMessage(role="user", content="what is (40+2)*10"))
    assert __import__("asyncio").run(agent._generate("chat1")) == "It is 420."


def test_retry_on_429_then_success():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        })

    prov = OpenAICompatibleProvider(LLMConfig(provider="openai", model="m", api_key="k", max_retries=4))
    prov._client = mock_client(handler)
    res = __import__("asyncio").run(prov.chat([ChatMessage(role="user", content="hi")]))
    assert res.text == "ok" and attempts["n"] == 3


def test_auth_error_not_retried():
    def handler(request):
        return httpx.Response(401, json={"error": "bad key"})

    from wa_agent_sdk.exceptions import ProviderAuthError

    prov = OpenAICompatibleProvider(LLMConfig(provider="openai", model="m", api_key="bad", max_retries=5))
    prov._client = mock_client(handler)
    with pytest.raises(ProviderAuthError):
        __import__("asyncio").run(prov.chat([ChatMessage(role="user", content="hi")]))
