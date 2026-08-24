# wa-agent-sdk — Project Documentation

Technical documentation for contributors and power users. For a usage-first
guide see [`README.md`](README.md).

- Version: 1.0.0 · Python ≥ 3.10 · Node ≥ 18 (bridge only)
- License: MIT
- Package: `wa_agent_sdk` (src layout) · CLI: `wa-agent`

---

## 1. Overview

wa-agent-sdk turns a personal WhatsApp number into an AI agent endpoint:

```
phone (any WhatsApp user)
   │  message / image / PDF / voice note
   ▼
WhatsApp multi-device servers
   │  encrypted sync
   ▼
Node bridge (Baileys)          ← bundled subprocess, owns the session
   │  localhost WebSocket, token-authenticated JSON events/RPC
   ▼
Python SDK                     ← this package
   ├─ Safety gate              anti-ban checks BEFORE any reply
   ├─ Triggers                 zero-cost keyword auto-replies
   ├─ Router (AgentRouter)     persona selection per chat/keyword
   ├─ Memory                   per-chat rolling history
   ├─ ProviderRouter           picks the LLM API per query
   ├─ Tools                    function calling + document/vision ingestion
   └─ Scheduler                reminders & recurring sends
   ▼
reply flows back through the bridge to WhatsApp
```

Design principles:

1. **One process per concern** — the bridge owns connectivity; Python owns
   intelligence. A crash in either is isolated and self-heals.
2. **Everything gated** — no outbound message bypasses the safety layer.
3. **Pluggable at every seam** — providers, tools, routes, hooks are registry/
   decorator based; no core edits needed.
4. **Local-first persistence** — JSON files under `.wa_sessions/` and
   `.wa_data/`; no external services required.

## 2. Repository layout

```
wa-agent-sdk/
├── pyproject.toml               packaging, deps, [project.scripts] wa-agent
├── README.md                    user guide
├── Doc.md                       this document
├── examples/                    six runnable bots
├── tests/                       pytest suite (57 tests)
└── src/wa_agent_sdk/
    ├── __init__.py              public exports (~40 symbols)
    ├── _version.py
    ├── cli.py                   wa-agent init | doctor
    ├── config.py                LLMConfig, AgentConfig, provider table
    ├── exceptions.py            typed error hierarchy
    ├── models.py                IncomingMessage, SentReceipt, MediaType
    ├── context.py               ContextVars (chat_jid/message_id for tools)
    ├── qr.py                    terminal QR rendering
    ├── memory.py                ConversationMemory (per-chat trimming)
    ├── safety.py                SafetyManager (anti-ban gates)
    ├── router.py                AgentRouter + TriggerBoard
    ├── routing.py               ProviderRouter + UsageTracker (multi-API)
    ├── scheduler.py             Scheduler (every/at/remind_after), Job
    ├── bridge.py                BaileysBridge supervisor (spawn/RPC/restart)
    ├── client.py                WhatsAppAgent orchestration
    ├── llm/
    │   ├── base.py              BaseChatProvider, ChatMessage/Result, retries
    │   ├── openai_compatible.py OpenAI/Groq/DeepSeek/OpenRouter/NVIDIA/xAI/…
    │   ├── anthropic.py         Claude
    │   ├── gemini.py            Google Gemini
    │   └── factory.py           kind→class registry, register_provider()
    ├── tools/
    │   ├── base.py              Tool, @tool, ToolRegistry (schema generation)
    │   ├── documents.py         PDF/DOCX/MD/CSV/HTML text extraction
    │   ├── media.py             EXIF-safe image normalisation for vision
    │   └── builtin.py           calculator, clock, persistent notes
    └── node_bridge/
        ├── package.json         baileys ^6.7 · ws · pino
        └── bridge.mjs           the entire Node side (≈300 lines)
```

## 3. Architecture components

### 3.1 Node bridge — `node_bridge/bridge.mjs`

Single responsibility: be a reliable WhatsApp Web client and expose it as a
local JSON API.

- Binds `ws://127.0.0.1:<ephemeral port>/?token=<uuid>` on startup **before**
  loading Baileys, so the parent can connect fast.
- Token check happens on every WebSocket upgrade (`4001 unauthorized`).
- Only one Python client may attach (`4002 already-connected`).
- Uses `useMultiFileAuthState` under `<sessions>/<name>/` and saves creds on
  every `creds.update`.
- Emits events (below); answers RPC commands with correlated `ref` results.

**Events (bridge → Python)**

| type | payload | meaning |
|---|---|---|
| `hello` | `{pid, bridge}` | WS attached, Baileys loading |
| `qr` | `{qr}` | fresh pairing QR string |
| `ready` | `{jid, name}` | connection open; linked identity |
| `disconnected` | `{code, logged_out, reason}` | close; bridge auto-reconnects unless logged out |
| `fatal` | `{error}` | unrecoverable (e.g. baileys import failed) |
| `message` | `{payload: IncomingMessage}` | normalised incoming message |

**RPC (Python → bridge, each answered as `{type:"result", ref, ok, …}`)**

| command | payload | result extras |
|---|---|---|
| `ping` | – | `{pong:true}` |
| `send_text` | `{to, text}` | `{id}` |
| `send_media` | `{to, media_type, data_b64, mimetype?, filename?, caption?, ptt?}` | `{id}` |
| `set_presence` | `{presence, jid?}` | – |
| `mark_read` | `{id, chat_jid, sender_jid?}` | – |
| `download_media` | `{id}` | `{data_b64, mimetype}` (LRU cache of last 200 media messages; reuploads via `sock.reuploadRequest`) |
| `logout` | – | wipes auth dir, exits after ack |

Normalisation (`normalize()`) unwraps ephemeral/view-once layers and maps
Baileys message kinds → `media_type ∈ {text,image,video,audio,sticker,
document,contact,location}`, extracting `text/caption`, `filename`,
`mimetype`, `mentioned_jids`, `quoted_participant`, `push_name`, timestamps.
Unknown message kinds return null and are dropped silently.

Reconnection model:

1. *WhatsApp-level* drops → bridge re-calls `startWhatsApp()` after 3 s
   (logged-out sessions stop instead).
2. *Process-level* crashes → Python's supervisor task respawns with
   exponential backoff (2^n up to 30 s, max `max_bridge_restarts`=5, counter
   resets after healthy uptime).

### 3.2 Bridge manager — `bridge.py`

`BaileysBridge` (async):

- `_ensure_dependencies()` runs `npm install --no-audit --no-fund` inside the
  bundled dir when `node_modules` is missing (long timeout, stderr surfaced).
- Picks a free port via ephemeral bind, spawns node with env
  (`WA_BRIDGE_PORT/TOKEN/AUTH_DIR/LOG_LEVEL`), pipes stdout+stderr to
  `.wa_sessions/logs/bridge-<port>.log`; crash reports include the log tail.
- RPC: `_rpc(payload, timeout)` assigns incrementing refs, awaits futures;
  failures raise `BridgeError`/`BridgeNotRunningError`.
- Supervisor task watches process/ws health and relaunches; emits `fatal`
  event when restart budget exhausted.

### 3.3 Agent orchestration — `client.py`

`WhatsAppAgent` owns lifecycle, hooks and the reply pipeline.

**Startup sequence**

```
start()
 ├─ create provider(s)            (default provider lazily via factory)
 ├─ BaileysBridge.start()         npm-install? spawn → ws connect → hello
 ├─ print QR (or custom on_qr)    on 'qr' event
 └─ await ready ≤ qr_timeout      else QRTimeoutError
run_forever()                      signal handlers → Ctrl-C sets stop_event
stop()                             cancel jobs/tasks → bridge.stop() → aclose providers/router
```

**Incoming-message pipeline** (per message, serialised per chat via
`_chat_locks`, global concurrency capped by semaphore(4)):

```
_process_message(msg)
 1. drop from_me · status@broadcast · groups (if ignore_groups) · not allowed_chats
 2. lock(chat) + semaphore
 3. opt-language detection        "stop/unsubscribe…"  → blocklist (+confirmation), STOP
                                  "start/subscribe…"   → unblocklist (+welcome), continue
 4. SafetyManager.gate()          trigger prefix → group mention-only → quiet hours
                                  → cooldown → hourly cap → daily caps (reserve counters)
 5. mark_read (blue ticks)
 6. TriggerBoard.match()          hit → humanized pause → send → done (no LLM)
 7. on_message hook               returns str → pause → send → done
 8. AgentRouter.resolve()         persona for this chat/keyword (may override prompt/model/tools)
 9. _build_user_message           download image → prepare_image → vision block
                                  download doc → parse_document → <document> block
                                  require_trigger prefix stripped from prompt
10. memory.append(user turn)
11. typing ON → _generate()       tool loop (see below) → typing OFF
12. humanized pause → send_text → memory.append(assistant turn)
```

Any `ProviderError` becomes a friendly one-line apology to the user; media /
document errors become their own explanatory message. Nothing ever crashes the
serve loop.

**Generation & the Provider Router**

```
_generate(chat_jid, route?)
 ├─ tools = base registry ⊕ route.tools
 ├─ if route.llm        → direct single-provider path (router bypassed)
 ├─ elif provider_router→ do_chat = router.chat(pinned=first-success-endpoint)
 └─ else                → default provider path
 loop ≤ max_tool_iterations:
     ChatResult = do_chat(conversation, schemas)
     if tool_calls → execute via registry (ContextVars set) → append role=tool
     else final text
```

### 3.4 Anti-ban — `safety.py`

`SafetyManager.gate()` evaluation order (first denial wins):
`opted_out → trigger_required → group_not_addressed → quiet_hours → cooldown →
global_hourly_limit → chat_daily_limit`. Allowed decisions *reserve* budget
immediately (counters persisted atomically). See README §8 for semantics and
tuning presets. `record_outbound()` charges manual/campaign sends into the same
budget. Humanized pauses are separate (`pause_before_send()`).

### 3.5 Personas & triggers — `router.py`

- `AgentRouter`: ordered `Route(name, match, system_prompt, llm, tools,
  priority)` list; matcher types = substring (case-insensitive) | compiled
  regex (verbatim flags) | list-of-substrings (ANY) | predicate fn | None
  (= catch-all). Highest priority match wins; fallback used otherwise.
- `TriggerBoard`: evaluated before hooks/LLM; `exact=True` compares the whole
  body; replies may be static strings or sync/async callables of the message.

### 3.6 Multi-API placement — `routing.py`

Core types: `ModelEndpoint` (tier/pricing/budget/vision/priority),
`QueryProfile` (complexity score), `UsageTracker` (persistent accounting),
`_Health` (circuit breaker), `ProviderRouter` (selection + failover).

Complexity score (clamped 0–1):

```
score = min(total_chars/6000, 1)*0.45
      + 0.25 (image)  + 0.20 (<document> present)
      + min(tools*0.02, .10)  + 0.15 (analytical keywords)
tiers: <0.34 fast · <0.67 balanced · ≥0.67 smart
```

Selection filters (all must pass): enabled → circuit-breaker OK → vision
capable → `est_input_chars ≤ max_input_chars` → `spend_today < daily_budget_usd`.
Strategies order candidates: smart (target tier first, cheapest inside tier,
auto-upgrade when empty), cheapest (estimated cost), balanced (calls today ↑),
failover (`priority` desc). Execution tries candidates sequentially; successes
reset the breaker, `ProviderError`s add a strike (threshold ⇒ cooldown),
`ProviderAuthError`s quarantine ~6× longer. Tool loops pin the winning endpoint
via `pinned=` so one reply never changes models mid-reasoning. Every success is
recorded to the tracker with real response token counts × configured prices.

### 3.7 LLM layer — `llm/`

Canonical `ChatMessage(role, content[str|blocks], tool_calls, tool_call_id,
name)`; blocks from `text_block()` / `image_block()`. `BaseChatProvider.chat()`
wraps `_chat_once()` with N-attempt exponential backoff (retryable: timeouts,
transport errors, HTTP 408/409/425/429/5xx) and maps 401/403 →
`ProviderAuthError` (never retried). Backends translate canonical messages to
native wire formats (OpenAI parts/image_url/tool_calls · Anthropic
system/tool_use/tool_result · Gemini contents/functionCall/functionResponse)
and parse native responses back, including `usage` token counts.

### 3.8 Tools — `tools/`

- Schema generation from type hints (`str/int/float/bool/list/dict`,
  `Optional`, defaults) + lightweight `Args:` docstring parsing.
- `Tool.run()` catches everything and returns error strings to the model.
- Document parsers dispatch by extension: pdf (pypdf, page-tagged, scanned-PDF
  notice), docx (python-docx paragraphs+tables), html (tag-stripping parser),
  plain-text family; truncation marker appended past `max_chars`.
- `prepare_image()`: EXIF transpose → alpha flatten → bounded thumbnail →
  PNG passthrough (small) or JPEG q88; returns base64+mime.
- Builtins: safe AST calculator (whitelisted ops/functions/constants, huge-pow
  guard), zoneinfo clock, per-chat persistent notes (JSON + asyncio lock +
  atomic replace) scoped via `context.current_chat_jid`.

### 3.9 Scheduling — `scheduler.py`

Fire-and-forget asyncio tasks tracked in a set: `every(interval, jid, text)`
(loop, exception-swallowing), `remind_after(delta|timedelta, …)`,
`at(datetime, …)`; text may be static or callable. `Job.cancel()`,
`cancel_all()` (called by `agent.stop()`). In-process only — restart clears
jobs by design.

### 3.10 Persistence map

| path | writer | contents |
|---|---|---|
| `.wa_sessions/<name>/**` | Baileys | device keys/credentials (**secret**) |
| `.wa_sessions/logs/bridge-*.log` | bridge manager | node stdout/stderr |
| `.wa_data/safety.json` | SafetyManager | day rollover, per-chat counters, hourly window |
| `.wa_data/optouts.json` | SafetyManager | sorted blocked number list |
| `.wa_data/provider_usage.json` | UsageTracker | per-day + lifetime tokens/calls/cost |
| `.wa_data/notes.json` | builtin notes | `{chat_jid: [{note, at}]}` |

All writes are tmp-file + `os.replace` (atomic on Windows/POSIX).

## 4. Extension points

| want to… | do this |
|---|---|
| add a provider backend | subclass `BaseChatProvider._chat_once`, `@register_provider("kind")` |
| route through your own logic | pass `provider_factory=` to `ProviderRouter` |
| add a tool | decorate a typed fn with `@tool(...)` then `agent.register_tool` |
| per-route toolsets | `route.tools.extend([...])` |
| custom persona matching | any `callable(IncomingMessage) -> bool` as `match=` |
| intercept messages | `@agent.on_message` (return str to short-circuit) |
| custom QR UX | `@agent.on_qr` (return True when handled) |
| richer scheduling durability | wrap `Scheduler` sender to also persist jobs |

## 5. Testing strategy

`pytest tests/` — 57 tests, no network required:

| file | coverage |
|---|---|
| `test_core.py` | provider table, calculator, schema gen, documents (pdf/docx/md), image prep, memory, agent construction, notes round-trip, dict construction |
| `test_providers.py` | wire-format conversion both ways for all 3 backends via `httpx.MockTransport` (incl. vision blocks, tool_calls, token usage), retry/backoff on 429, auth errors |
| `test_safety.py` | every gate: trigger, cooldown, daily/new-chat/hourly caps, overnight quiet hours, opt-out persistence, group mention-only, stats |
| `test_router_scheduler.py` | route priority/fallback/matchers, triggers (exact/dynamic/async), scheduler every/at/remind/cancel, campaign pacing + opt-out skip |
| `test_routing.py` | complexity profiling, all 4 strategies, vision filter, budgets, circuit breaker, pinned endpoints, usage accounting/persistence |
| `test_router_integration.py` | agent-level `_generate` through router with live failover + usage report + route-override bypass |

Plus a manual E2E harness (scripts kept out of CI): boots the real
`bridge.mjs` with installed Baileys and asserts hello/ping/error-RPC/token-auth
over WebSocket.

## 6. Security considerations

- Bridge binds **127.0.0.1** only; upgrades require a per-launch random token.
- WhatsApp credentials sit plaintext in `.wa_sessions/` (that is how Baileys
  works) — protect/backup deliberately; delete folder to unlink.
- No telemetry anywhere; usage stats stay local.
- Legal: unofficial protocol automation can breach WhatsApp ToS — dedicated
  number recommended; the safety module exists precisely to keep behaviour
  human-shaped (opt-outs honoured, paced sends, quiet hours).

## 7. Known limitations & roadmap

| limitation | note / future work |
|---|---|
| Voice notes not transcribed | placeholder text sent to model; plug Whisper behind `handle_images`-style flag later |
| Video/sticker content ignored | acknowledged placeholders only |
| Scheduler not durable | pair with DB via custom hook, or roadmap: sqlite-backed jobs |
| Single WhatsApp account per agent process | run multiple agents/processes with distinct `session_name`s today |
| Group admin/moderation features | mentions plumbing exists; policy engine is roadmap |
| Media downloads cached in RAM (200) | swap-in Redis/disk cache possible via bridge tweak |

## 8. Glossary

| term | meaning |
|---|---|
| JID | WhatsApp address: `<num>@s.whatsapp.net` (user), `<id>@g.us` (group) |
| endpoint | one configured LLM API inside a `ProviderRouter` |
| route | one persona inside an `AgentRouter` (prompt/model/toolset bundle) |
| trigger | keyword rule answered without touching any LLM |
| gate | one boolean check inside the safety pipeline |
| pinned endpoint | the router endpoint reused across a tool-call loop |
