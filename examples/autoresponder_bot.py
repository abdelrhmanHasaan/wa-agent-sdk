"""Zero-LLM-cost autoresponder: FAQ triggers + business hours + opt-outs."""

import re

from wa_agent_sdk import LLMConfig, WhatsAppAgent

agent = WhatsAppAgent(
    llm=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="YOUR_KEY"),
    system_prompt="You are the front desk of Acme Store. Be brief.",
    quiet_hours=("22:00", "08:00"),   # never reply at night
    new_chat_daily_limit=8,           # strangers get fewer replies on day one
)

agent.add_trigger("pricing", "💰 Starter $9/mo · Growth $29/mo · Scale $99/mo. Which fits you?")
agent.add_trigger("hours", "We're open Mon–Fri 9:00–18:00, Sat 10:00–14:00.")
agent.add_trigger("location", "We're at 221B Market Street. Map: https://maps.example.com/acme")
agent.add_trigger(r"\b(refund|return)s?\b",
                  "Refunds are free within 30 days — reply here with your order ID and "
                  "I'll process it.")
agent.add_trigger("/help", "Commands: pricing, hours, location, refund. Or just ask me anything!",
                  exact=True)

# Regex + dynamic replies also work:
def tracking_reply(message):
    order_id = re.search(r"ACM-\d+", message.text, re.I).group(0)
    return f"Tracking {order_id}: shipped, arriving Thursday. 📦"

agent.add_trigger(re.compile(r"track\s+(ACM-\d+)", re.I), tracking_reply)

# Anything that isn't a trigger falls through to the AI (with full context).

if __name__ == "__main__":
    agent.run()
