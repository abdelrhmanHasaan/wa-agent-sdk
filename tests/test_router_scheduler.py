"""Router, trigger, scheduler and campaign tests."""

import asyncio
import re

from wa_agent_sdk import (
    AgentRouter,
    IncomingMessage,
    LLMConfig,
    TriggerBoard,
    WhatsAppAgent,
)
from wa_agent_sdk.models import jid_to_number


def msg(text="hi", chat="15551234567@s.whatsapp.net", is_group=False, push_name=None):
    return IncomingMessage(id="m", chat_jid=chat, sender_jid=chat, text=text,
                           media_type="text", is_group=is_group, push_name=push_name)


# ------------------------------------------------------------------- router

def test_router_priority_and_fallback():
    router = AgentRouter()
    router.add_route("low", match="price", priority=1)
    router.add_route("high", match=re.compile(r"\bprice\b", re.I), priority=10)
    router.add_route("fallback")  # catch-all

    assert router.resolve(msg("what's the PRICE?")).name == "high"
    assert router.resolve(msg("hello world")).name == "fallback"
    empty = AgentRouter()
    assert empty.resolve(msg()) is None
    assert router.get("high").priority == 10


def test_router_callable_matcher():
    router = AgentRouter()
    router.add_route("vip", match=lambda m: m.push_name == "Boss", priority=5)
    assert router.resolve(msg(push_name="Boss")).name == "vip"
    assert router.resolve(msg(push_name="Staff")) is None  # no fallback registered


def test_trigger_board_static_and_dynamic():
    board = TriggerBoard()
    board.add("pricing", "cheap!")
    board.add("/help", "commands", exact=True)
    board.add("time", lambda m: f"now for {m.chat_jid.split('@')[0]}", priority=5)

    # both "time"(prio 5) and "pricing"(prio 0) match -> highest priority wins
    assert asyncio.run(board.match(msg("any TIME left for PRICING?"))) == "now for 15551234567"
    assert asyncio.run(board.match(msg('/help'))) == 'commands'
    assert asyncio.run(board.match(msg("/help me please"))) is None  # exact
    assert asyncio.run(board.match(msg("just pricing please"))) == "cheap!"


async def _trigger_async_reply():
    board = TriggerBoard()

    async def factory(m):
        return f"async {m.id}"

    board.add("go", factory)
    assert await board.match(msg("let's GO now")) == "async m"


# ---------------------------------------------------------------- scheduler

def test_scheduler_every_and_remind():
    sent = []

    async def sender(jid, text):
        sent.append((jid, text))

    async def run():
        from wa_agent_sdk.scheduler import Scheduler

        s = Scheduler(sender)
        job = s.every(0.05, "j@x", lambda j: "tick", run_immediately=True)
        await asyncio.sleep(0.22)
        ticks = len(sent)
        job.cancel()
        await asyncio.sleep(0.1)
        stable = len(sent)

        s.remind_after(0.05, "j@x", "one-shot")
        await asyncio.sleep(0.15)
        return ticks, stable, list(sent)

    ticks, stable, all_sent = asyncio.run(run())
    assert ticks >= 2 and len(sent) >= ticks  # kept ticking until cancelled
    assert any(t == "one-shot" for _, t in all_sent)


def test_scheduler_at_and_cancel_all():
    sent = []

    async def sender(jid, text):
        sent.append(text)

    from datetime import datetime, timedelta

    from wa_agent_sdk.scheduler import Scheduler

    async def run():
        s = Scheduler(sender)
        s.at(datetime.now() + timedelta(seconds=0.05), "j", "at-msg")
        s.every(10_000, "j", "never")
        await asyncio.sleep(0.12)
        cancelled = s.cancel_all()
        await asyncio.sleep(0.05)
        return cancelled, s.active

    cancelled, active = asyncio.run(run())
    assert cancelled == 1 and active == 0 and "at-msg" in sent


# ----------------------------------------------------------------- campaign

def test_send_campaign_pacing_and_optouts():
    agent = WhatsAppAgent(llm=LLMConfig(provider="openai", model="m", api_key="k"),
                          enable_safety=True,
                          campaign_min_delay=0.01, campaign_max_delay=0.02)
    outgoing = []

    async def fake_send(to, text):
        outgoing.append(to)

    agent.send_text = fake_send  # type: ignore[method-assign]
    agent.safety.set_blocked("15550000003@s.whatsapp.net", True)

    targets = [f"1555000000{i}@s.whatsapp.net" for i in range(6)]
    report = asyncio.run(agent.send_campaign(targets, "sale!"))

    assert report["sent"] == 5 and report["skipped_opted_out"] == 1 and report["failed"] == 0
    assert len(outgoing) == 5
    # every send was accounted against the safety budget
    stats = agent.safety.stats()
    assert stats["hourly"]["count"] >= 5


def test_campaign_dynamic_text():
    agent = WhatsAppAgent(llm=LLMConfig(provider="openai", model="m", api_key="k"))
    seen = []
    async def fake_send(to, text):
        seen.append(text)
    agent.send_text = fake_send  # type: ignore[method-assign]
    asyncio.run(agent.send_campaign(["15550000001@s.whatsapp.net"],
                                    lambda j: f"hey {jid_to_number(j)}"))
    assert seen == ["hey 15550000001"]
