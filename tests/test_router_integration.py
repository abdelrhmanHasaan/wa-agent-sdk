"""End-to-end: WhatsAppAgent replies through a ProviderRouter with failover."""

import asyncio
from pathlib import Path

from wa_agent_sdk import (
    LLMConfig,
    ModelEndpoint,
    ProviderRouter,
    Strategy,
    Tier,
    UsageTracker,
    WhatsAppAgent,
)
from wa_agent_sdk.exceptions import ProviderError
from wa_agent_sdk.llm.base import BaseChatProvider, ChatMessage, ChatResult


class Scripted(BaseChatProvider):
    async def _chat_once(self, messages, tools):
        if self.config.model == "flaky":
            raise ProviderError("primary is down")
        return ChatResult(text=f"via-{self.config.model}", input_tokens=10, output_tokens=4)


def build_agent(tmp: Path) -> WhatsAppAgent:
    router = ProviderRouter(
        [
            ModelEndpoint(name="flaky", tier=Tier.SMART, priority=10,
                          llm=LLMConfig(provider="openai", model="flaky", api_key="k")),
            ModelEndpoint(name="steady", tier=Tier.FAST, price_input_per_m=0.5,
                          llm=LLMConfig(provider="groq", model="steady", api_key="k")),
        ],
        strategy=Strategy.FAILOVER,
    )
    router._factory = lambda cfg: Scripted(cfg)
    return WhatsAppAgent(
        llm=LLMConfig(provider="openai", model="unused-default", api_key="k"),
        provider_router=router,
        data_dir=tmp,
    )


def test_agent_generates_through_router_with_failover_and_usage():
    tmp = Path(__import__("tempfile").mkdtemp())
    agent = build_agent(tmp)
    agent.memory.append("c1", ChatMessage(role="user", content="hello"))

    reply = asyncio.run(agent._generate("c1"))
    assert reply == "via-steady"           # primary failed -> failover answered
    assert agent._last_endpoint_name == "steady"

    report = agent.usage_summary()
    assert report["today"]["steady"]["calls"] == 1
    assert report["today"]["steady"]["input_tokens"] == 10

    # route-level LLM override bypasses the router entirely:
    route = agent.add_route("special", system_prompt="be weird")
    route.llm = LLMConfig(provider="openai", model="direct", api_key="k")
    Scripted.registry = {}  # not used by Scripted class; direct provider built by factory
    agent.memory.append("c2", ChatMessage(role="user", content="hi"))

    from wa_agent_sdk.llm.factory import create_provider
    original_factory = create_provider

    class DirectScripted(BaseChatProvider):
        async def _chat_once(self, messages, tools):
            return ChatResult(text="route-direct")

    monkey_provider = DirectScripted(route.llm)
    agent._extra_providers["openai:direct:" + route.llm.resolved_base_url] = monkey_provider

    reply2 = asyncio.run(agent._generate("c2", route=route))
    assert reply2 == "route-direct"
