"""A support bot with custom tools, hooks, image + document understanding.

Demonstrates:
  * registering your own tools (function-calling)
  * intercepting messages before the AI (slash commands, greetings)
  * restricting who can talk to the bot
"""

import httpx
from wa_agent_sdk import IncomingMessage, LLMConfig, WhatsAppAgent, tool

agent = WhatsAppAgent(
    llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="YOUR_GROQ_KEY"),
    system_prompt=(
        "You are 'Nova', the support agent for a small online store called Acme. "
        "Answer order/product questions using the tools provided. Be brief."
    ),
    ignore_groups=True,
    # allowed_chats={"15551234567"},   # uncomment to whitelist numbers
)


@tool(description="Look up an order by its ID (e.g. 'ACM-1042'). Returns status and ETA.")
async def get_order_status(order_id: str) -> str:
    """Fake demo backend."""
    fake_db = {
        "ACM-1042": "Shipped — arriving Thursday",
        "ACM-1043": "Packing in progress",
    }
    return fake_db.get(order_id.upper(), f"Order {order_id} not found.")


@tool(description="Return today's store hours as text.")
def store_hours() -> str:
    return "Mon–Fri 9:00–18:00, Sat 10:00–14:00."

agent.register_tool(get_order_status)
agent.register_tool(store_hours)


@tool(description="Fetch the current Bitcoin price in USD from CoinGecko.")
async def btc_price() -> str:
    async with httpx.AsyncClient(timeout=15) as client_:
        r = await client_.get("https://api.coingecko.com/api/v3/simple/price",
                              params={"ids": "bitcoin", "vs_currencies": "usd"})
        data = r.json()
        return f"BTC = ${data['bitcoin']['usd']}"


@agent.on_message
async def slash_commands(message: IncomingMessage):
    text = (message.text or "").strip().lower()
    if text == "/start":
        return (
            "Hi! I'm Nova 🤖\n"
            "Ask me about your order (e.g. ACM-1042), send me a PDF to summarize, "
            "or a photo to describe."
        )
    if text == "/reset":
        agent.memory.clear(message.chat_jid)
        return "Conversation cleared ✨"


if __name__ == "__main__":
    agent.run()
