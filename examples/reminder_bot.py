"""Reminders & recurring messages via the built-in scheduler.

Commands (send to the bot from your phone):
    /remind 10m drink water
    /remind 2h call mom
    /daily 8h standup notes        (repeats every 8 hours)
"""

import re
import time

from wa_agent_sdk import IncomingMessage, LLMConfig, WhatsAppAgent

agent = WhatsAppAgent(
    llm=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="YOUR_KEY"),
    system_prompt="You are a cheerful personal assistant on WhatsApp.",
)

DURATION = re.compile(r"(\d+)\s*(s|sec|m|min|h|hr|d)\b", re.I)
UNITS = {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "hr": 3600, "d": 86400}


def parse_duration(text: str) -> float | None:
    match = DURATION.search(text)
    if not match:
        return None
    return float(match.group(1)) * UNITS[match.group(2).lower()]


@agent.on_message
async def commands(message: IncomingMessage):
    chat = message.chat_jid
    text = (message.text or "").strip()

    if text.lower().startswith("/remind"):
        delay = parse_duration(text)
        note = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else "your reminder"
        if not delay:
            return "Format: /remind 10m <what>  (supports s/m/h/d)"
        agent.scheduler.remind_after(delay, chat, f"⏰ Reminder: {note}")
        return f"Noted! I'll ping you in {int(delay // 60 or delay)}{'m' if delay >= 60 else 's'} ⏳"

    if text.lower().startswith("/every"):
        delay = parse_duration(text)
        note = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else "your recurring message"
        if not delay:
            return "Format: /every 8h <what>"
        agent.scheduler.every(delay, chat,
                              lambda jid: f"🔁 {note} ({time.strftime('%H:%M')})")
        return f"Recurring every {delay / 3600:.0f}h — send /stopall to cancel."

    if text.lower() == "/stopall":
        n = agent.scheduler.cancel_all()
        return f"Cancelled {n} scheduled job(s)."

    if text.lower().startswith("/ping"):
        return "pong"


if __name__ == "__main__":
    # Jobs can also be created from code before startup:
    # agent.scheduler.remind_after(3600, "15551234567@s.whatsapp.net", "Stretch!")
    agent.run()
