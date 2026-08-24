"""Minimal agent: scan the QR once, then chat with your AI on WhatsApp."""

from wa_agent_sdk import LLMConfig, WhatsAppAgent

agent = WhatsAppAgent(
    llm=LLMConfig(
        provider="openai",           # openai | groq | deepseek | openrouter | anthropic | gemini ...
        model="gpt-4o-mini",
        api_key="YOUR_API_KEY",      # or set OPENAI_API_KEY and omit this line
    ),
    system_prompt="You are a friendly personal assistant. Keep replies short and useful.",
)


@agent.on_ready
async def announce():
    print("Agent is live! Send it a PDF, an image, or just say hi.")


if __name__ == "__main__":
    agent.run()  # prints the pairing QR, then serves messages until Ctrl+C
