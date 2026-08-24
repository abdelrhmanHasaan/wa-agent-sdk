"""Human-paced broadcast campaign with opt-out compliance.

Reads numbers from contacts.txt (one JID per line) and sends with randomized
6–15s delays. Anyone who ever replied STOP is skipped automatically.
"""

from pathlib import Path

from wa_agent_sdk import LLMConfig, WhatsAppAgent

agent = WhatsAppAgent(
    llm=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="YOUR_KEY"),
    campaign_min_delay=6.0,   # seconds between sends (randomized in range)
    campaign_max_delay=15.0,
)


def personalized(jid: str) -> str:
    return (
        "Hi 👋 Acme here: our autumn sale starts today — 20% off everything. "
        "Reply STOP to opt out."
    )


async def main():
    await agent.start()  # QR pairing (or instant reconnect if already linked)

    contacts_file = Path("contacts.txt")
    jids = [
        line.strip()
        for line in contacts_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and "@" in line
    ]
    report = await agent.send_campaign(jids[:50], personalized)   # keep batches small!
    print(report)   # {'sent': 47, 'skipped_opted_out': 3, 'failed': 0}

    await agent.stop()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
