"""Under-the-hood echo debugger — see every stage a message passes through.

Run me, send a message from ANOTHER phone, and I reply with live state.
No LLM is touched unless you send `/llm <question>`.

Console shows: raw bridge events, gate decisions, pipeline drops, heartbeats.
"""

import asyncio
import json
import logging
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from wa_agent_sdk import IncomingMessage, LLMConfig, WhatsAppAgent

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
for _noisy in ("httpx", "websockets", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

USE_LLM_FOR_ALL = os.environ.get("ECHO_USE_LLM") == "1"

agent = WhatsAppAgent(
    llm=LLMConfig(
        provider=os.environ.get("ECHO_PROVIDER", "nvidia"),
        model=os.environ.get("ECHO_MODEL", "meta/llama-3.1-70b-instruct"),
        api_key=os.environ.get("NVIDIA_API_KEY", ""),
    ),
    system_prompt="You are a debugging assistant. Be terse.",
    ignore_groups=False,       # surface group traffic too
    group_mention_only=False,
    qr_max_attempts=3,
)

state = {"bridge_events": 0, "raw_messages": 0, "gate_allowed": 0, "gate_denied": 0,
         "hooks_hit": 0, "echoes_sent": 0, "llm_replies": 0}


# ------------------------------------------------------------- raw bridge tap
async def wire_raw_tap():
    """Print EVERY event crossing the bridge before any processing."""
    bridge = agent._bridge
    original_emit = bridge._emit

    async def tapping_emit(event, payload):
        state["bridge_events"] += 1
        if event == "message":
            p = payload.get("payload") or {}
            state["raw_messages"] += 1
            print(
                f"\n📩 RAW #{state['raw_messages']} id={str(p.get('id'))[:10]} "
                f"chat={p.get('chat_jid')} type={p.get('media_type')} "
                f"from_me={p.get('from_me')} group={p.get('is_group')} "
                f"text={(p.get('text') or '')!r}"
            )
        else:
            print(f"[BRIDGE] {event} {json.dumps(payload, ensure_ascii=False)[:180]}")
        await original_emit(event, payload)

    bridge._emit = tapping_emit


# ------------------------------------------------------------ gate spy
_orig_gate = agent.safety.gate


def spying_gate(msg, *a, **kw):
    decision = _orig_gate(msg, *a, **kw)
    if decision.allowed:
        state["gate_allowed"] += 1
        print(f"🛂 GATE ALLOW msg={str(msg.id)[:10]} from={msg.sender_jid}")
    else:
        state["gate_denied"] += 1
        print(f"🛂 GATE DENY({decision.reason}) msg={str(msg.id)[:10]} "
              f"chat={msg.chat_jid} from_me={msg.from_me} group={msg.is_group}")
    return decision


agent.safety.gate = spying_gate


# ------------------------------------------------------------------ echo hook
@agent.on_message
async def echo_with_state(message: IncomingMessage):
    state["hooks_hit"] += 1
    text = (message.text or "").strip()

    if text.lower() in ("/quit", "!quit"):
        asyncio.get_event_loop().create_task(agent.stop())
        return "🛑 Debugger stopping…"

    if text.lower().startswith("/llm"):
        print(f"➡️  '/llm' detected — letting the REAL pipeline (provider="
              f"{agent.config.llm.provider}/{agent.config.llm.model}) answer this one.")
        return None  # fall through to the actual AI

    state["echoes_sent"] += 1
    return (
        "🧪 ECHO — pipeline works!\n"
        f"state: events={state['bridge_events']} msgs={state['raw_messages']} "
        f"gate_ok={state['gate_allowed']} denied={state['gate_denied']} "
        f"echoes={state['echoes_sent']}\n"
        f"your msg: id={message.id[:10]}… type={message.media_type.value} "
        f"from_me={message.from_me} group={message.is_group}\n"
        f"from: {message.display_sender}\n"
        f"said: {text!r}\n"
        "(try: /llm why is the sky blue)"
    )


# ------------------------------------------------------------------ lifecycle
async def heartbeat():
    while True:
        await asyncio.sleep(20)
        print(f"💓 alive — {json.dumps(state)}")


async def main():
    await agent.start()
    await wire_raw_tap()

    @agent.on_ready
    async def _announce():
        pass

    hb = asyncio.create_task(heartbeat())
    print(
        "\n🔍 DEBUGGER LIVE\n"
        "  1. Send ANY WhatsApp message from a DIFFERENT number to the linked one\n"
        "     (messages sent from the linked phone itself are ignored by design:\n"
        "      watch for RAW … from_me=True)\n"
        "  2. Watch console stages: RAW → GATE → hook → ECHO reply\n"
        "  3. /llm <question> exercises the real NVIDIA path\n"
        "  4. /quit stops the bot\n"
    )
    try:
        await agent.run_forever()
    finally:
        hb.cancel()
        print(f"🏁 FINAL STATE {json.dumps(state)}")
        await agent.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
