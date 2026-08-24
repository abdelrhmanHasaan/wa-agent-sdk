# wa-agent-sdk

Build **AI agents on WhatsApp** that run on **your own phone number** — no Meta
business verification, no WhatsApp Cloud API, no approval queue.

You configure three things, scan one QR code, and your agent is live:

```python
provider + model + api key  →  scan QR  →  agent answers your WhatsApp
```

| | |
|---|---|
| 📄 Documents | PDF · DOCX · Markdown · CSV · JSON · HTML · TXT auto-parsed into the model's context |
| 🖼️ Images | Routed to vision models (GPT-4o, Claude, Gemini, Llama-vision…) |
| 🧰 Tools | Function calling with automatic JSON-schema generation |
| 🛡️ Anti-ban | Rate limits, cooldowns, quiet hours, STOP compliance, humanized pacing |
| 🔀 Multi-agent | Route chats to different personas/models/tools on one number |
| ⚡ Triggers | Instant keyword replies with zero LLM cost |
| ⏰ Scheduler | Reminders & recurring messages |
| 📣 Campaigns | Human-paced bulk messaging with opt-out compliance |

> ⚠️ **Disclaimer** — wa-agent-sdk uses WhatsApp's multi-device web protocol
> (Baileys), which is not an official API. Automating a number can violate
> WhatsApp's Terms of Service; use a dedicated number, never spam, and read the
> [anti-ban guide](#-anti-ban-protection-stay-unbanned) before going live.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Requirements](#requirements)
3. [Install](#install)
4. [60-second quickstart](#60-second-quickstart)
5. [Linking your number (QR)](#linking-your-number-qr)
6. [Providers](#providers)
7. [Configuration reference](#configuration-reference)
8. [🛡️ Anti-ban protection](#️-anti-ban-protection-stay-unbanned)
9. [Use-case guides](#use-case-guides)
   - [9.1 Personal AI assistant](#91-personal-ai-assistant)
   - [9.2 Customer-support bot over your docs](#92-customer-support-bot-over-your-docs)
   - [9.3 Keyword autoresponder (zero token cost)](#93-keyword-autoresponder-zero-token-cost)
   - [9.4 Multi-persona router (support / sales / VIP)](#94-multi-persona-router-support--sales--vip)
   - [9.5 Reminders & scheduled messages](#95-reminders--scheduled-messages)
   - [9.6 Broadcast campaigns](#96-broadcast-campaigns)
   - [9.7 Vision bot (photos → answers)](#97-vision-bot-photos--answers)
   - [9.8 Group assistant (mention-only)](#98-group-assistant-mention-only)
   - [9.9 Fully local & private with Ollama](#99-fully-local--private-with-ollama)
   - [9.10 Custom tools deep dive](#910-custom-tools-deep-dive)
   - [9.11 Bring-your-own provider](#911-bring-your-own-provider)
10. [API reference](#api-reference)
11. [On-disk layout & debugging](#on-disk-layout--debugging)
12. [Troubleshooting](#troubleshooting)

---

## How it works

```
┌──────────────────────┐   WebSocket (JSON)    ┌───────────────────┐   WhatsApp
│  Python SDK          │◄─────────────────────►│  Node bridge      │   multi-device ◄──► your phone
│  agents·LLM·tools    │                       │  Baileys          │   (QR pairing)
│  safety·scheduler    │                       └───────────────────┘
└──────────────────────┘
```

A bundled Node subprocess owns the WhatsApp Web connection; Python handles
agents, LLM providers, tools, safety and scheduling. The session persists to
disk, so you scan the QR **once** — restarts reconnect automatically and the
bridge self-heals (exponential-backoff respawns).

## Requirements

- Python **≥ 3.10**
- Node.js **≥ 18** ([nodejs.org](https://nodejs.org)) — needed by the bridge only
- A WhatsApp account (use a spare number for production bots)

## Install

```bash
cd wa-agent-sdk
pip install -e .
wa-agent doctor     # verifies python/node/npm/python-deps at a glance
wa-agent init       # optional wizard that generates my_whatsapp_agent.py
```

Bridge npm dependencies (`baileys`, `ws`, `pino`) are installed automatically on
first run. Behind a corporate proxy? Set `HTTPS_PROXY` first.

## 60-second quickstart

```python
from wa_agent_sdk import WhatsAppAgent, LLMConfig

agent = WhatsAppAgent(
    llm=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-..."),
    system_prompt="You are a helpful personal assistant.",
)

agent.run()   # prints QR → WhatsApp ▸ Settings ▸ Linked devices ▸ Link a device
```

Message the number from another phone — the agent replies. Send it a PDF and
ask questions about it; send a photo and ask what's in it.

Run examples from [`examples/`](examples/): `basic_agent`, `autoresponder_bot`,
`support_router`, `reminder_bot`, `broadcast_campaign`, `support_agent`.

## Linking your number (QR)

- First `start()` renders a scannable ASCII QR in your terminal.
- Session credentials persist in `.wa_sessions/<session_name>/` — subsequent
  runs reconnect without a QR.
- Run **multiple numbers** by giving each agent its own `session_name`.
- To relink fresh: delete the session folder or call `await bridge.logout()`.
- QR codes expire after ~60 s in WhatsApp itself; the SDK automatically shows
  fresh ones while `start()` waits (`qr_timeout`, default 180 s).

## Providers

All providers accept `LLMConfig(provider=..., model=..., api_key=...)`.
`api_key` may be omitted if the listed env var is exported.

| provider | models (examples) | base URL | env var | vision | tools |
|---|---|---|---|---|---|
| `openai` | gpt-4o, gpt-4o-mini, o4-mini | api.openai.com/v1 | `OPENAI_API_KEY` | ✅ | ✅ |
| `anthropic` / `claude` | claude-sonnet-4-5, claude-opus-4-1 | api.anthropic.com | `ANTHROPIC_API_KEY` | ✅ | ✅ |
| `gemini` / `google` | gemini-2.0-flash, gemini-2.5-pro | generativelanguage.googleapis.com | `GEMINI_API_KEY` | ✅ | ✅ |
| `groq` | llama-3.3-70b-versatile | api.groq.com/openai/v1 | `GROQ_API_KEY` | some | ✅ |
| `deepseek` | deepseek-chat, deepseek-reasoner | api.deepseek.com/v1 | `DEEPSEEK_API_KEY` | ❌ | ✅ |
| `openrouter` | any routed model | openrouter.ai/api/v1 | `OPENROUTER_API_KEY` | varies | ✅ |
| `nvidia` | meta/llama-3.3-70b-instruct, nvidia/… | integrate.api.nvidia.com/v1 | `NVIDIA_API_KEY` | varies | ✅ |
| `xai` / `grok` | grok-3, grok-3-mini | api.x.ai/v1 | `XAI_API_KEY` | ✅ | ✅ |
| `together` | Qwen/Llama/Mixtral | api.together.xyz/v1 | `TOGETHER_API_KEY` | varies | ✅ |
| `mistral` | mistral-large-latest | api.mistral.ai/v1 | `MISTRAL_API_KEY` | some | ✅ |
| `fireworks` | firefunction, Llama … | api.fireworks.ai/inference/v1 | `FIREWORKS_API_KEY` | varies | ✅ |
| `ollama` (local) | llama3, qwen2.5, llava | localhost:11434/v1 | none | llava ✅ | ✅ |
| `lmstudio` (local) | whatever you loaded | localhost:1234/v1 | none | varies | ✅ |

Notes:
- **NVIDIA NIM**: get a key at build.nvidia.com; model names look like
  `meta/llama-3.3-70b-instruct` or `mistralai/mixtral-8x22b-instruct-v0.1`.
- **Grok (xAI)**: console.x.ai → `XAI_API_KEY`; works like OpenAI.
- **Ollama**: `ollama pull llama3` then just
  `LLMConfig(provider="ollama", model="llama3")` — no key, fully offline.
  Use `llava` for images.
- Any other OpenAI-compatible endpoint: see [§9.11](#911-bring-your-own-provider).
- Missing keys raise `ProviderAuthError` naming the expected env var; unknown
  providers raise a suggestion ("Did you mean: openai?").

## Configuration reference

```python
WhatsAppAgent(llm=..., **overrides)   # overrides = AgentConfig fields
```

**Behaviour**

| field | default | meaning |
|---|---|---|
| `system_prompt` | friendly assistant | persona for unmatched chats (routes can override) |
| `ignore_groups` | `True` | never respond inside group chats |
| `allowed_chats` | `None` | whitelist of JIDs/numbers; `None` = everyone |
| `typing_indicator` | `True` | show "typing…" while generating |
| `mark_read` | `True` | blue-double-check incoming messages we handle |
| `enable_builtin_tools` | `True` | calculator, clock, persistent per-chat notes |

**Media & memory**

| field | default | meaning |
|---|---|---|
| `handle_images` | `True` | attach photos to vision-capable models |
| `handle_documents` | `True` | extract text from PDF/DOCX/MD/… attachments |
| `max_document_chars` | `15_000` | per-document truncation cap |
| `max_image_side_px` | `1600` | longest-side downscale before upload |
| `max_history_messages` | `40` | per-chat rolling window |
| `max_context_chars` | `60_000` | hard char budget across history |
| `max_tool_iterations` | `6` | max model↔tool round-trips per reply |

**Anti-ban (see next section)**

| field | default |
|---|---|
| `enable_safety` | `True` |
| `require_trigger` | `None` (e.g. `"!bot"`) |
| `group_mention_only` | `True` |
| `reply_cooldown` | `3.0` s per chat |
| `global_hourly_limit` | `80` |
| `per_chat_daily_limit` | `50` |
| `new_chat_daily_limit` | `12` (stricter cap on day one) |
| `quiet_hours` | `None` (e.g. `("23:00", "07:00")`) |
| `humanize_min_delay` / `humanize_max_delay` | `0.8` / `3.0` s |
| `campaign_min_delay` / `campaign_max_delay` | `6.0` / `15.0` s |

**Infrastructure**: `session_name` ("default"), `sessions_dir`, `data_dir`,
`qr_timeout` (180 s), `max_bridge_restarts` (5), `log_level`.

## 🛡️ Anti-ban protection (stay unbanned)

WhatsApp bans **behaviour**, not code: instant replies, messaging strangers
non-stop, ignoring opt-outs, activity at night. Every counter-measure is on by
default and enforced in `_process_message` *before* any reply is composed:

1. **Opt-out compliance** — anyone sending *stop / unsubscribe / remove me /
   cancel* is blocklisted (persisted in `.wa_data/optouts.json`); they receive
   one confirmation and are never messaged again until they say *start /
   subscribe / unstop*. Campaigns skip blocked numbers automatically.
2. **Trigger gate** — set `require_trigger="!bot"` and the agent only reacts to
   messages starting with `!bot` (the prefix is stripped before the LLM sees it).
3. **Group mention-only** — in groups, reply solely when mentioned
   (`@YourBot …`), quoted, or triggered. Uses real mention metadata from the
   bridge.
4. **Rate limits** — per-chat cooldown, per-chat daily cap, global hourly cap.
   Counters persist across restarts (`.wa_data/safety.json`).
5. **New-number warm-up** — brand-new chats get `new_chat_daily_limit`
   (default 12/day) instead of the full daily allowance.
6. **Quiet hours** — no replies inside the window; supports overnight ranges.
7. **Humanized pacing** — every reply waits a randomized 0.8–3 s; campaigns
   wait 6–15 s between sends. Typing indicators make it feel natural.

Recommended presets:

| use case | suggested knobs |
|---|---|
| personal assistant (your own number) | defaults fine; safety mostly moot |
| support bot on business number | defaults + `require_trigger=None`, keep quiet hours |
| public-facing bot in many groups | `group_mention_only=True`, `per_chat_daily_limit=30` |
| marketing broadcasts | batches ≤ 50, `campaign_*_delay ≥ 6–15 s`, always include "Reply STOP to opt out" |

## Use-case guides

### 9.1 Personal AI assistant

The quickstart above *is* this. Nice extras:

```python
agent = WhatsAppAgent(
    llm=LLMConfig(provider="gemini", model="gemini-2.0-flash", api_key=os.environ["GEMINI_API_KEY"]),
    system_prompt="You know me well: concise, direct, occasional dry humour.",
    allowed_chats={"+15551234567"},        # only I can talk to it
)

@agent.on_ready
async def ready():
    print("linked as", agent.bot_jid)
```

Built-ins you get for free: `calculate`, `current_datetime`,
`remember_note`/`recall_notes`/`clear_notes` (persistent memory scoped per chat).

### 9.2 Customer-support bot over your docs

Send the bot your PDFs once; it answers from them forever (history carries the
extracted text):

```python
agent = WhatsAppAgent(
    llm=LLMConfig(provider="openai", model="gpt-4o", api_key="sk-..."),
    system_prompt="You support Acme products. Answer ONLY from provided documents; "
                  "if unsure, say so and offer escalation.",
    max_document_chars=25_000,
)
```

Customers literally message their `manual.pdf` + "what's the warranty?" —
or you pre-load knowledge at startup:

```python
from pathlib import Path
from wa_agent_sdk import ChatMessage, parse_document

@agent.on_ready
async def seed_knowledge():
    for pdf in Path("knowledge").glob("*.pdf"):
        doc = parse_document(pdf.name, pdf.read_bytes(), max_chars=40_000)
        agent.memory.append("kb", ChatMessage(role="user",
            content=f"<document>{doc.text}</document>"))
```

### 9.3 Keyword autoresponder (zero token cost)

Full example: [`examples/autoresponder_bot.py`](examples/autoresponder_bot.py).

```python
agent.add_trigger("pricing", "💰 Starter $9/mo · Growth $29/mo · Scale $99/mo")
agent.add_trigger(r"\brefund\b", "Free refunds within 30 days — what's your order ID?")
agent.add_trigger("/help", "pricing · hours · location · refund", exact=True)
agent.add_trigger(re.compile(r"track\s+(ACM-\d+)", re.I),
                  lambda msg: f"{re.search('ACM-\\d+', msg.text, re.I).group(0)}: shipped 📦")
```

Semantics: plain string = case-insensitive substring; compiled regex = used
verbatim (add `re.I` yourself); `exact=True` = whole message equality.
Unmatched text falls through to the LLM. Combine with `quiet_hours` so the bot
never texts customers at midnight.

### 9.4 Multi-persona router (support / sales / VIP)

Full example: [`examples/support_router.py`](examples/support_router.py).

```python
billing = agent.add_route(
    "billing", match="billing", priority=10,
    system_prompt="You are Acme's billing specialist…",
    llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-5", api_key="..."),
)
billing.tools.extend([lookup_invoice])           # route-specific tools!

agent.add_route("tech", match=re.compile(r"\berror|crash\b", re.I), priority=10, ...)
agent.add_route("vip", match=lambda m: m.push_name == "Big Client", priority=20, ...)
```

Matcher types: substring · regex · list-of-substrings (ANY) · predicate(msg).
Highest `priority` wins; a route with `match=None` is the catch-all; else the
agent-level prompt/model handles it. Each route may override the model — cheap
model for FAQ, expensive one for billing disputes.

### 9.5 Reminders & scheduled messages

Full example: [`examples/reminder_bot.py`](examples/reminder_bot.py).

```python
# from chat commands (handled in an on_message hook):
agent.scheduler.remind_after(600, chat_jid, "⏰ drink water")
agent.scheduler.every(8 * 3600, chat_jid, lambda jid: f"standup notes {time.strftime('%H:%M')}")
agent.scheduler.at(datetime(2026, 9, 1, 9, 0), chat_jid, "Invoice due today!")

n = agent.scheduler.cancel_all()      # or job.cancel()
print(f"cancelled {n} jobs")          # scheduler.active -> live count
```

Text sources may be static strings, sync or async callables receiving the JID.
Jobs live in-process (restart clears them — pair with a DB if you need durable
jobs). `stop()` cancels everything cleanly.

### 9.6 Broadcast campaigns

Full example: [`examples/broadcast_campaign.py`](examples/broadcast_campaign.py).

```python
report = await agent.send_campaign(
    jids,
    lambda jid: f"Hi! Acme autumn sale −20%. Reply STOP to opt out.",  # or plain str
    min_delay=6, max_delay=15,        # randomized seconds between sends
)
# {'sent': 47, 'skipped_opted_out': 3, 'failed': 0}
```

STOP-listed numbers are skipped automatically; every send is charged against
the same safety budget as normal replies. Keep batches small (≤ 50) and always
include an opt-out line — this is the single biggest ban-risk feature, so it's
built to be slow on purpose.

### 9.7 Vision bot (photos → answers)

Nothing to configure — pick a vision model:

```python
agent = WhatsAppAgent(llm=LLMConfig(provider="openai", model="gpt-4o", ...))
# customer sends receipt.jpg + "how much was this?"
# → image resized (EXIF-fixed, ≤1600px) → attached to the model → answer
```

Images are EXIF-rotated, alpha-flattened and downscaled before upload. With
Ollama use `llava`. Text-only models get "(image received but not shown)"
instead of crashing.

### 9.8 Group assistant (mention-only)

```python
agent = WhatsAppAgent(
    llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="..."),
    ignore_groups=False,            # engage with groups…
    group_mention_only=True,        # …but ONLY when @mentioned or quoted
    require_trigger=None,
)
```

Members type `@YourBot explain recursion` — everyone else's chatter is ignored
(mention metadata comes straight from WhatsApp). Add `require_trigger="!bot"`
for double-gating.

### 9.9 Fully local & private with Ollama

```bash
ollama pull llama3
ollama pull llava        # optional: image understanding
```

```python
agent = WhatsAppAgent(llm=LLMConfig(provider="ollama", model="llama3"))
# zero API keys, zero cloud — messages never leave your machine
```

### 9.10 Custom tools deep dive

```python
from wa_agent_sdk import tool

@tool(description="Look up an order like 'ACM-1042'. Returns status + ETA.")
async def get_order_status(order_id: str) -> str:
    return await backend.fetch(order_id)

agent.register_tool(get_order_status)              # sync functions work too
```

JSON schema is generated from the signature + annotations
(`str/int/float/bool/list/dict`, `Optional[...]`). Docstrings enrich it:

```python
def search(query: str, limit: int = 5) -> list:
    """Search the product catalogue.

    query: free-text search string
    limit: max results to return
    """
```

Errors raised inside tools are caught and returned to the model as
`Error: …` text so it can recover conversationally. Per-route toolsets live on
the route object ([§9.4](#94-multi-persona-router-support--sales--vip)).

### 9.11 Bring-your-own provider

Any endpoint speaking OpenAI `/chat/completions`:

```python
from wa_agent_sdk import LLMConfig, register_provider
from wa_agent_sdk.llm.openai_compatible import OpenAICompatibleProvider

@register_provider("myproxy")
class MyProxy(OpenAICompatibleProvider):
    name = "myproxy"

cfg = LLMConfig(provider="myproxy", model="whatever",
                base_url="https://llm.corp.internal/v1", api_key="internal-key")
```

Subclass `BaseChatProvider` (implement `_chat_once`) for exotic APIs — retry,
backoff, auth-error mapping and timeouts come free from the base class.

## API reference

```python
agent = WhatsAppAgent(llm=..., **config_overrides)
await agent.start()                    # connect (+QR); raises QRTimeoutError
await agent.run_forever()              # serve until stop(); SIGINT-safe
agent.run()                            # sync wrapper for both
await agent.stop()
async with agent: ...                  # context manager

# hooks (decorators)
@agent.on_message      async def h(msg) -> str | None   # str = reply now, skip AI
@agent.on_ready        async def h()
@agent.on_qr           async def h(qr_str) -> bool      # True = handled

# building blocks
agent.add_trigger(pattern, reply, exact=False, priority=0)
agent.add_route(name, match=None, system_prompt=..., llm=..., tools=[...], priority=0)
agent.register_tool(fn_or_tool)
agent.scheduler.every / .remind_after / .at / .cancel_all / .active
agent.send_campaign(jids, text, min_delay=, max_delay=) -> dict
agent.memory.history(jid) / .clear(jid) / .clear_all()
agent.safety.stats() / .set_blocked(jid, True|False) / .is_blocked(jid)

# outbound (usable anywhere after start)
await agent.send_text(jid, text) -> SentReceipt
await agent.send_image(jid, path_or_bytes, caption="")
await agent.send_document(jid, path_or_bytes, filename=, caption="")
await agent.send_audio(jid, path, voice_note=False)
await agent.send_typing(jid, True|False)
await agent.broadcast([jids], text)

agent.bot_jid          # your linked identity once ready
```

JID format: `<number>@s.whatsapp.net` (direct), `<id>@g.us` (group);
bare numbers are accepted wherever a JID goes through helpers like
`allowed_chats`.

## On-disk layout & debugging

| path | contents |
|---|---|
| `.wa_sessions/<name>/` | Baileys credentials (the QR scan result) — treat as secret |
| `.wa_sessions/logs/bridge-*.log` | Node bridge stdout/stderr |
| `.wa_data/safety.json` | rate-limit counters |
| `.wa_data/optouts.json` | STOP list |
| `.wa_data/notes.json` | builtin notes tool storage |

Logging: `logging.getLogger("wa_agent")` (children `.bridge`, `.safety`,
`.router`, `.scheduler`). Quick start:

```python
import logging; logging.basicConfig(level=logging.INFO)   # DEBUG for the chatty stuff
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `NodeNotFoundError` | install Node LTS, reopen terminal |
| npm install fails behind proxy | set `HTTPS_PROXY`/`HTTP_PROXY`, rerun |
| QR scanned but nothing happens | wait ~10 s (first link is slow); check bridge log |
| Replies suddenly stop | check `agent.safety.stats()` — you likely hit a limit |
| Bot ignores a user | they're probably STOP-listed (`is_blocked`) |
| Bot won't shut up in a group | `group_mention_only` is on — @mention it, or set `False` |
| Want a totally fresh start | delete `.wa_sessions/<name>/` and restart |
| `ProviderAuthError` | export the env var named in the error, or pass `api_key=` |
| Images ignored | your model isn't vision-capable — switch models |
| Bridge crash details | tail `.wa_sessions/logs/bridge-*.log` |
