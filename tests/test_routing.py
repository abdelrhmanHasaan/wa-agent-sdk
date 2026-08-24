"""ProviderRouter tests — strategies, budgets, circuit breaker, usage accounting."""

import asyncio
import json

import pytest

from wa_agent_sdk import (
    LLMConfig,
    ModelEndpoint,
    ProviderRouter,
    Strategy,
    Tier,
    UsageTracker,
    profile_query,
)
from wa_agent_sdk.exceptions import ProviderError
from wa_agent_sdk.llm.base import BaseChatProvider, ChatMessage, ChatResult


class FakeProvider(BaseChatProvider):
    """Programmable stand-in; keyed off the model name via a shared registry."""

    registry: dict = {}

    async def _chat_once(self, messages, tools):
        spec = FakeProvider.registry[self.config.model]
        if isinstance(spec, Exception):
            raise spec
        assert isinstance(spec, ChatResult)
        return spec


def make_router(endpoints, *, tracker=None, **kw) -> ProviderRouter:
    router = ProviderRouter(
        endpoints,
        provider_factory=lambda cfg: FakeProvider(cfg),
        failure_threshold=kw.pop("failure_threshold", 3),
        cooldown_seconds=kw.pop("cooldown_seconds", 300.0),
        **kw,
    )
    return router


def ep(name, *, model=None, tier=Tier.BALANCED, price_in=1.0, **kw):
    cfg = LLMConfig(provider="openai", model=model or name, api_key="k")
    return ModelEndpoint(name=name, llm=cfg, tier=tier,
                         price_input_per_m=price_in,
                         price_output_per_m=price_in, **kw)


def msgs(text="hi"):
    return [ChatMessage(role="user", content=text)]


@pytest.fixture(autouse=True)
def _clean_registry():
    FakeProvider.registry = {}
    yield
    FakeProvider.registry = {}


# ------------------------------------------------------------------ profiling

def test_profile_scores_simple_vs_complex():
    simple = profile_query(msgs("hi"), 0)
    assert simple.score < 0.34 and simple.target_tier is Tier.FAST

    big_doc = "<document filename='x.pdf'>" + "x" * 9000 + "</document>"
    heavy = profile_query([ChatMessage(role="user", content=big_doc)], 3)
    assert heavy.score >= 0.67 and heavy.has_document
    assert heavy.target_tier is Tier.SMART

    img = profile_query([ChatMessage(role="user", content=[
        {"type": "image", "mime_type": "image/jpeg", "data": "xx"}])], 0)
    assert img.has_image and img.needs_vision

    why = profile_query(msgs("why does this crash? analyze the stack trace"), 0)
    assert why.score > simple.score


# ------------------------------------------------------------------ strategies

def test_cheapest_picks_lowest_cost():
    FakeProvider.registry = {"cheap": ChatResult(text="c"), "pricey": ChatResult(text="p")}
    r = make_router([
        ep("pricey", tier=Tier.SMART, price_in=15),
        ep("cheap", tier=Tier.FAST, price_in=0.05),
    ], strategy=Strategy.CHEAPEST)
    res, used = asyncio.run(r.chat(msgs("hello there")))
    assert used.name == "cheap" and res.text == "c"


def test_smart_routes_by_complexity():
    FakeProvider.registry = {
        "mini": ChatResult(text="fast!"),
        "gpt": ChatResult(text="smart!"),
    }
    r = make_router([
        ep("mini", model="mini", tier=Tier.FAST, price_in=0.05),
        ep("gpt", model="gpt", tier=Tier.SMART, price_in=10),
    ], strategy=Strategy.SMART)

    _, used_low = asyncio.run(r.chat(msgs("ok")))
    assert used_low.tier is Tier.FAST

    heavy_text = "analyze " + ("detail " * 3000)
    _, used_high = asyncio.run(r.chat(msgs(heavy_text)))
    assert used_high.tier is Tier.SMART


def test_vision_requirement_filters_endpoints():
    FakeProvider.registry = {"blind": ChatResult(text="b"), "eyes": ChatResult(text="e")}
    r = make_router([
        ep("blind", price_in=0.01, supports_vision=False),
        ep("eyes", price_in=5.0),
    ], strategy=Strategy.CHEAPEST)
    image_msg = [ChatMessage(role="user", content=[
        {"type": "text", "text": "what is this"},
        {"type": "image", "mime_type": "image/jpeg", "data": "AA"}])]
    _, used = asyncio.run(r.chat(image_msg))
    assert used.name == "eyes"


def test_failover_on_provider_error():
    from wa_agent_sdk.exceptions import ProviderError as PE

    FakeProvider.registry = {
        "primary": PE("down", retryable=True),
        "backup": ChatResult(text="backup ok"),
    }
    r = make_router([
        ep("primary", priority=10),
        ep("backup", priority=1),
    ], strategy=Strategy.FAILOVER)
    res, used = asyncio.run(r.chat(msgs("hi")))
    assert used.name == "backup" and res.text == "backup ok"


def test_circuit_breaker_skips_failing_endpoint():
    from wa_agent_sdk.exceptions import ProviderError as PE

    calls = {"flaky": 0}

    class Counting(FakeProvider):
        async def _chat_once(self, messages, tools):
            if self.config.model == "flaky":
                calls["flaky"] += 1
                raise PE("boom")
            return ChatResult(text="fine")

    r = make_router([ep("flaky"), ep("solid")], strategy=Strategy.FAILOVER,
                    failure_threshold=1)
    r._factory = lambda cfg: Counting(cfg)

    first = asyncio.run(r.chat(msgs("a")))
    second = asyncio.run(r.chat(msgs("b")))
    assert first[1].name == "solid" and second[1].name == "solid"
    assert calls["flaky"] == 1  # quarantined after first failure, not retried


def test_daily_budget_skips_endpoint():
    tmp = __import__("pathlib").Path(__import__("tempfile").mkdtemp())
    tracker = UsageTracker(tmp / "usage.json")
    tracker.record("budget-hit", input_tokens=1000, output_tokens=1000, cost_usd=99.0)
    FakeProvider.registry = {"budget-hit": ChatResult(text="x"), "spare": ChatResult(text="y")}
    r = make_router([
        ep("budget-hit", daily_budget_usd=10.0, price_in=1),
        ep("spare", price_in=2),
    ], usage_tracker=tracker, strategy=Strategy.CHEAPEST)
    _, used = asyncio.run(r.chat(msgs("hi")))
    assert used.name == "spare"


def test_all_candidates_down_raises():
    from wa_agent_sdk.exceptions import ProviderError as PE

    FakeProvider.registry = {"only": PE("dead")}
    r = make_router([ep("only")])
    with pytest.raises(ProviderError, match="All 1 routed endpoints failed"):
        asyncio.run(r.chat(msgs("hi")))


def test_pinned_endpoint_reused_for_tool_loops():
    FakeProvider.registry = {"a": ChatResult(text="A"), "b": ChatResult(text="B")}
    r = make_router([ep("a"), ep("b")], strategy=Strategy.BALANCED)
    res1, ep1 = asyncio.run(r.chat(msgs("one")))
    res2, ep2 = asyncio.run(r.chat(msgs("two"), pinned=ep1.name))
    assert ep1.name == ep2.name


# --------------------------------------------------------------------- usage

def test_usage_tracker_records_and_persists():
    tmp = __import__("pathlib").Path(__import__("tempfile").mkdtemp())
    path = tmp / "provider_usage.json"
    t = UsageTracker(path)
    t.record("groq-mini", input_tokens=100, output_tokens=50, cost_usd=0.001)
    t.record("groq-mini", input_tokens=10, output_tokens=5, cost_usd=0.0001)

    s = t.summary()
    today = s["today"]["groq-mini"]
    assert today["calls"] == 2 and today["input_tokens"] == 110
    assert abs(today["cost_usd"] - 0.0011) < 1e-9
    assert s["lifetime"]["groq-mini"]["calls"] == 2

    t2 = UsageTracker(path)  # persisted across instances
    assert t2.spend_today("groq-mini") > 0


def test_router_records_real_token_usage():
    tmp = __import__("pathlib").Path(__import__("tempfile").mkdtemp())
    tracker = UsageTracker(tmp / "u.json")
    FakeProvider.registry = {"paid": ChatResult(text="ok", input_tokens=120, output_tokens=30)}
    r = make_router([ep("paid", price_in=1.0)], usage_tracker=tracker)
    asyncio.run(r.chat(msgs("hello")))
    stats = tracker.summary()["today"]["paid"]
    assert stats["input_tokens"] == 120 and stats["output_tokens"] == 30
    expected_cost = (120 * 1.0 + 30 * 1.0) / 1_000_000
    assert abs(stats["cost_usd"] - expected_cost) < 1e-9


def test_usage_summary_reports_budget_remaining():
    tmp = __import__("pathlib").Path(__import__("tempfile").mkdtemp())
    tracker = UsageTracker(tmp / "u.json")
    tracker.record("cap", input_tokens=0, output_tokens=0, cost_usd=2.0)
    FakeProvider.registry = {}
    r = make_router([ep("cap", daily_budget_usd=10.0)], usage_tracker=tracker)
    summary = r.usage_summary()
    assert summary["today"]["cap"]["budget_remaining"] == pytest.approx(8.0)


def test_auth_error_long_quarantine():
    from wa_agent_sdk.exceptions import ProviderAuthError

    FakeProvider.registry = {
        "badkey": ProviderAuthError("invalid key"),
        "good": ChatResult(text="ok"),
    }
    r = make_router([ep("badkey", priority=10), ep("good")], strategy=Strategy.FAILOVER)
    res, used = asyncio.run(r.chat(msgs("hi")))
    assert used.name == "good" and res.text == "ok"
