# wa-agent-sdk

Build **WhatsApp AI agents** that run on **your own phone number** — no Meta
business verification, no WhatsApp Cloud API, no approval process.

The SDK pairs with your WhatsApp account by showing a one-time **QR code**
(via a bundled [Baileys](https://github.com/WhiskeySockets/Baileys) bridge),
then turns every incoming message into an LLM conversation. You only configure:

```python
provider + model + api key  →  scan QR  →  your agent is live
```

Out of the box it understands:

- 📄 **Documents** — PDF, DOCX, Markdown, TXT, CSV, JSON, HTML, YAML… (auto-extracted text)
- 🖼️ **Images** — routed to vision models (GPT-4o, Claude, Gemini, Llama-vision…)
- 🧰 **Tools** — function calling with auto JSON-schema generation (`@tool`)
- 💬 **Memory** — per-chat conversation history, typing indicators, read receipts

---

## How it works

```
┌──────────────────────┐   WebSocket (JSON)   ┌──────────────────┐   multi-device
│  Python SDK          │◄────────────────────►│  Node bridge     │◄───────────────► your phone
│  agents • LLM • tools│                      │  Baileys         │   (QR pairing)
└──────────────────────┘                      └──────────────────┘
```

The Node subprocess owns the WhatsApp Web connection; the Python side handles
agents, LLM providers, tools and media parsing. Sessions are persisted on disk,
so you scan the QR **once** — restarts reconnect automatically.

## Requirements

- Python ≥ 3.10
- Node.js ≥ 18 ([nodejs.org](https://nodejs.org)) — needed once by the bridge
- A WhatsApp account (use a spare number if unsure)

## Install

```bash
cd wa-agent-sdk
pip install -e .
wa-agent doctor      # sanity-check python/node/deps
```

Bridge npm dependencies install automatically on first run (or run
`npm install` inside `src/wa_agent_sdk/node_bridge/` yourself).

## Quick start

```python
from wa_agent_sdk import WhatsAppAgent, LLMConfig

agent = WhatsAppAgent(
    llm=LLMConfig(
        provider="openai",        # or groq / deepseek / openrouter / anthropic / gemini ...
        model="gpt-4o-mini",
        api_key="sk-...",         # or export OPENAI_API_KEY
    ),
    system_prompt="You are a helpful assistant. Be concise.",
)

agent.run()   # prints QR → scan in WhatsApp ▸ Linked devices → done
```

Or scaffold instantly: `wa-agent init`.

### Supported providers

| provider     | kind            | env var                |
|--------------|-----------------|------------------------|
| openai       | openai-compat   | `OPENAI_API_KEY`       |
| groq         | openai-compat   | `GROQ_API_KEY`         |
| deepseek     | openai-compat   | `DEEPSEEK_API_KEY`     |
| openrouter   | openai-compat   | `OPENROUTER_API_KEY`   |
| together / mistral / fireworks | openai-compat | … |
| ollama / lmstudio (local)      | openai-compat | none needed |
| anthropic    | anthropic       | `ANTHROPIC_API_KEY`    |
| gemini       | gemini          | `GEMINI_API_KEY`       |

Any OpenAI-compatible endpoint works via `LLMConfig(provider="custom",
base_url="...", api_key="...")` after `register_provider("custom")(MyProvider)`.

## What your agent can do

**Automatic ingestion** — when someone sends your number a PDF/DOCX/MD file,
the text is extracted and placed in the model's context; images are resized and
attached to vision models; captions are honoured.

**Your own tools** (the model calls them when needed):

```python
from wa_agent_sdk import tool

@tool(description="Look up an order by ID like 'ACM-1042'.")
async def get_order_status(order_id: str) -> str:
    return await my_backend.fetch(order_id)

agent.register_tool(get_order_status)
```

Built-ins included: `calculator`, `current_datetime`, `remember_note`,
`recall_notes`, `clear_notes` (per-chat persistent memory).

**Hooks & control**

```python
@agent.on_message
async def commands(message):           # runs before the AI
    if (message.text or "").strip() == "/ping":
        return "Pong!"                 # return a string to reply directly

@agent.on_ready
async def ready(): print("live!")

await agent.send_text("15551234567@s.whatsapp.net", "hello")
await agent.send_image(jid, "chart.png", caption="daily report")
await agent.send_document(jid, "invoice.pdf")
await agent.broadcast([jid1, jid2], "server maintenance tonight")
```

Useful `AgentConfig` options: `ignore_groups`, `allowed_chats`,
`max_history_messages`, `max_document_chars`, `handle_images`,
`enable_builtin_tools`, `typing_indicator`, `mark_read`, `qr_timeout`,
`session_name` (multiple numbers = multiple agents).

See [`examples/`](examples/) for complete runnable bots.

## Project layout

```
src/wa_agent_sdk/
├── client.py        WhatsAppAgent orchestration + hooks
├── bridge.py        spawns/supervises the Node bridge over WS
├── llm/             base + openai-compatible + anthropic + gemini providers
├── tools/           @tool registry, PDF/DOCX/MD parsers, image prep, built-ins
├── config.py        LLMConfig / AgentConfig
├── memory.py        per-chat history trimming
├── qr.py            terminal QR rendering
└── node_bridge/     bridge.mjs (Baileys) — installed & launched automatically
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `NodeNotFoundError` | Install Node.js LTS from nodejs.org |
| npm install fails behind proxy | set `HTTPS_PROXY`, then rerun |
| QR expired / never scanned | call `start()` again for a fresh code |
| Want to relink another number | delete `.wa_sessions/<session_name>/` or `bridge.logout()` |
| Bridge crash details | see `.wa_sessions/logs/bridge-*.log` |

## ⚠️ Disclaimer

This SDK uses WhatsApp's **multi-device web protocol** (Baileys), which is not
an official API. Automating a personal number can, in rare cases, violate
WhatsApp's Terms of Service and lead to temporary bans. Use a dedicated number,
avoid spam, and don't use this for bulk messaging.
