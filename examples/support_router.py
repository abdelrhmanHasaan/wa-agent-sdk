"""Multi-persona support: one WhatsApp number, several specialised agents."""

import re

from wa_agent_sdk import LLMConfig, WhatsAppAgent, tool

agent = WhatsAppAgent(
    # Default/fallback model for anything unmatched:
    llm=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="YOUR_KEY"),
    system_prompt="You are Acme's general assistant.",
)


@tool(description="Look up an order by ID like 'ACM-1042'. Returns status and ETA.")
def get_order_status(order_id: str) -> str:
    return {"ACM-1042": "Shipped — arriving Thursday"}.get(
        order_id.upper(), f"Order {order_id} not found."
    )


@tool(description="Fetch an invoice by its ID (e.g. 'INV-204').")
def lookup_invoice(invoice_id: str) -> str:
    return f"Invoice {invoice_id}: $149, paid while you wait."


# 1) Billing persona — its own model AND its own tools
billing = agent.add_route(
    "billing",
    match="billing",
    priority=10,
    system_prompt=(
        "You are Acme's billing specialist. Help with invoices, refunds and payment "
        "methods. Use the tools to look up order/invoice IDs when mentioned."
    ),
    llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-5", api_key="YOUR_ANTHROPIC_KEY"),
)
billing.tools.extend([get_order_status, lookup_invoice])

# 2) Tech support — regex matcher (add re.I for case-insensitivity)
agent.add_route(
    "tech-support",
    match=re.compile(r"\b(error|bug|crash|not working)\b", re.I),
    priority=10,
    system_prompt="You are a patient senior support engineer. Diagnose step by step.",
    tools=[get_order_status],
)

# 3) Sales — list matcher = fires if ANY word appears
agent.add_route(
    "sales",
    match=["price", "buy", "demo"],
    priority=5,
    system_prompt="You are an enthusiastic but honest sales rep. Qualify the lead.",
)

# 4) VIPs — arbitrary Python predicate
agent.add_route(
    "vip",
    match=lambda msg: msg.push_name in ("Mom", "Big Client"),
    priority=20,
    system_prompt="You are talking to a VIP. Be extra warm and proactive.",
)

# Everything unmatched falls back to the agent-level llm + system_prompt.

if __name__ == "__main__":
    agent.run()
