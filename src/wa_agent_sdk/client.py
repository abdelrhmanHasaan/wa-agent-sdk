"""The high-level :class:`WhatsAppAgent` — QR linking, AI replies, tools, hooks."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import signal
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from .bridge import BaileysBridge
from .batching import MessageBatcher
from .config import AgentConfig, LLMConfig, replace_config
from .context import current_chat_jid, current_message_id
from .exceptions import (
    MediaError,
    ProviderError,
    QRTimeoutError,
    UnsupportedDocumentError,
    WaAgentError,
)
from .llm.base import BaseChatProvider, ChatMessage, image_block, text_block
from .llm.factory import create_provider
from .memory import ConversationMemory
from .models import IncomingMessage, MediaType, SentReceipt, jid_to_number
from .qr import print_pairing_qr
from .router import AgentRouter, TriggerBoard
from .safety import SafetyManager
from .scheduler import CampaignReport, Scheduler
from .tools.base import Tool, ToolRegistry, tool_from_callable
from .tools.builtin import create_builtin_tools
from .tools.documents import format_document_block, parse_document
from .tools.media import detect_image_mime, prepare_image

log = logging.getLogger("wa_agent")

# Some providers (notably NVIDIA NIM) force tool-mode server-side and answer
# ordinary chat with meta-narration instead of content. When we see that, we
# simply retry the same conversation without the tools array.
_DEGENERATE_TOOL_RE = re.compile(
    r"\bno function call\b"
    r"|\bno functions?\s+(?:is|are)\s+needed\b"
    r"|\b(?:do(?:es)?\s+not|don'?t)\s+need\s+to\s+(?:call|use)\s+(?:any\s+)?functions?\b"
    r"|\bcannot\s+call\s+any\s+functions?\b",
    re.IGNORECASE,
)

MessageHook = Callable[[IncomingMessage], Awaitable[Any]]
ReadyHook = Callable[[], Awaitable[None]]
QrHook = Callable[[str], Awaitable[bool]]


class WhatsAppAgent:
    """An AI agent reachable through your own WhatsApp number.

    Usage:
        agent = WhatsAppAgent(
            llm=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-..."),
            system_prompt="You are a helpful assistant.",
        )
        agent.run()          # shows a QR code, then serves messages forever
    """

    _LLM_FIELDS = frozenset(
        {"provider", "model", "api_key", "base_url", "temperature", "max_tokens",
         "request_timeout", "max_retries"}
    )

    def __init__(self, llm: LLMConfig | AgentConfig | dict | None = None, **overrides: Any) -> None:
        if llm is None and not overrides:
            raise WaAgentError(
                "Usage: WhatsAppAgent(llm=LLMConfig(provider=..., model=..., api_key=...), ...)"
            )
        if isinstance(llm, AgentConfig):
            config = replace_config(llm, **overrides) if overrides else llm
        elif isinstance(llm, LLMConfig):
            config = AgentConfig(llm=llm, **overrides) if overrides else AgentConfig(llm=llm)
        elif isinstance(llm, dict) or llm is None:
            merged: dict[str, Any] = {**(llm or {}), **overrides}
            llm_kwargs = {
                key: merged.pop(key) for key in list(merged) if key in self._LLM_FIELDS
            }
            if "provider" not in llm_kwargs or "model" not in llm_kwargs:
                raise WaAgentError("Dict-style construction needs at least provider= and model=")
            config = AgentConfig(llm=LLMConfig(**llm_kwargs), **merged)
        else:
            raise WaAgentError(
                "Pass llm=LLMConfig(...) (or an AgentConfig / a dict with provider & model)"
            )
        config.llm.info  # validates the provider name early
        self._setup(config)

    def _setup(self, config: AgentConfig) -> None:
        from ._console import force_utf8_stdio

        force_utf8_stdio()
        self.config = config
        self.tools = ToolRegistry()
        if config.enable_builtin_tools:
            for t in create_builtin_tools(config.resolved_data_dir()):
                self.tools.register(t)

        self.memory = ConversationMemory(
            max_messages=config.max_history_messages,
            max_chars=config.max_context_chars,
        )
        self.safety = SafetyManager(
            config.resolved_data_dir(),
            enabled=config.enable_safety,
            group_mention_only=config.group_mention_only,
            require_trigger=config.require_trigger,
            reply_cooldown=config.reply_cooldown,
            global_hourly_limit=config.global_hourly_limit,
            per_chat_daily_limit=config.per_chat_daily_limit,
            new_chat_daily_limit=config.new_chat_daily_limit,
            quiet_hours=config.quiet_hours,
            humanize_min_delay=config.humanize_min_delay,
            humanize_max_delay=config.humanize_max_delay,
        )
        self.router = AgentRouter()
        self.triggers = TriggerBoard()
        self.scheduler = Scheduler(self.send_text)
        self._provider: BaseChatProvider | None = None
        self._extra_providers: dict[str, BaseChatProvider] = {}
        self._prouter = config.provider_router
        if self._prouter is not None:
            self._prouter.attach_storage(config.resolved_data_dir())
        self._last_endpoint_name: str | None = None
        if config.human_batching:
            self._batcher = MessageBatcher(
                window=config.batch_window_seconds,
                max_wait=config.batch_max_wait_seconds,
                build=self._prepare_batched_context,
                on_flush=self._flush_batched_reply,
            )
        else:
            self._batcher = None
        self._committed_batches: set[tuple[str, tuple[str, ...]]] = set()
        self._committed_order: deque[tuple[str, tuple[str, ...]]] = deque(maxlen=400)
        self._bridge: BaileysBridge | None = None
        self._ready = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._bot_jid: str | None = None
        self._bot_name: str | None = None
        self._last_qr: str | None = None
        self._saw_qr = False
        self._qr_count = 0
        self._message_hook: MessageHook | None = None
        self._ready_hook: ReadyHook | None = None
        self._qr_hook: QrHook | None = None
        self._chat_locks: dict[str, asyncio.Lock] = {}
        self._semaphore = asyncio.Semaphore(4)
        self._background_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ hooks

    def on_message(self, fn: MessageHook) -> MessageHook:
        """Decorator: intercept every incoming message.

        Return a string from the hook to reply directly and skip the LLM,
        or return None to let the normal AI pipeline handle the message.
        """
        self._message_hook = fn
        return fn

    def on_ready(self, fn: ReadyHook) -> ReadyHook:
        """Decorator: called once WhatsApp reports the session as open."""
        self._ready_hook = fn
        return fn

    def on_qr(self, fn: QrHook) -> QrHook:
        """Decorator: custom QR handling. Return True when fully handled."""
        self._qr_hook = fn
        return fn

    def register_tool(self, item: Tool | Callable[..., Any], **kwargs: Any) -> Tool:
        """Add an extra tool the model may call. Accepts a Tool or typed function."""
        resolved = item if isinstance(item, Tool) else tool_from_callable(item, **kwargs)
        self.tools.register(resolved)
        return resolved

    def add_trigger(
        self,
        pattern: str | Any,
        reply: Any,
        *,
        exact: bool = False,
        priority: int = 0,
    ) -> None:
        """Instant rule-based reply (no LLM cost).

        ``pattern`` is a substring (case-insensitive), compiled regex, or an
        exact word when ``exact=True``. ``reply`` may be a static string or a
        (possibly async) callable receiving the message.
        """
        self.triggers.add(pattern, reply, exact=exact, priority=priority)

    def add_route(self, name: str, match: Any = None, **kwargs: Any) -> Any:
        """Register an agent persona; see :mod:`wa_agent_sdk.router`."""
        return self.router.add_route(name, match, **kwargs)

    # -------------------------------------------------------------- lifecycle

    @property
    def bot_jid(self) -> str | None:
        return self._bot_jid

    @property
    def is_connected(self) -> bool:
        return self._ready.is_set()

    async def start(self) -> None:
        """Start the bridge and wait until WhatsApp links this device."""
        if self._ready.is_set():
            return
        if self._provider is None:
            self._provider = create_provider(self.config.llm)
        if self._bridge is None:
            self._bridge = BaileysBridge(
                auth_dir=self.config.resolved_sessions_dir() / self.config.session_name,
                event_handler=self._on_bridge_event,
                log_level=self.config.log_level,
                max_restarts=self.config.max_bridge_restarts,
            )
        await self._bridge.start()
        self._saw_qr = False
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=self.config.qr_timeout)
        except asyncio.TimeoutError as exc:
            raise self._qr_failure_error() from exc

    def _qr_failure_error(self) -> QRTimeoutError:
        """Turn a linking timeout into an actionable diagnostic."""
        head = (
            f"Linking timed out after {self.config.qr_timeout:.0f}s "
            f"(attempt budget: {self.config.qr_max_attempts} via agent.run())."
        )
        session_dir = (
            self.config.resolved_sessions_dir() / self.config.session_name
        )
        if self._saw_qr:
            tip = (
                "QR codes were displayed but never scanned/confirmed in time.\n"
                f"  → Scan within ~60s of each code appearing (fresh codes print automatically).\n"
                f"  → Or raise AgentConfig(qr_timeout=..., qr_max_attempts=...)."
            )
        elif session_dir.exists() and any(session_dir.iterdir()):
            tip = (
                "No QR was ever produced and the session folder already exists — the stored\n"
                "link is stale or corrupted, so WhatsApp skips pairing entirely.\n"
                f"  → Delete it and rerun:\n"
                f"     PowerShell : Remove-Item -Recurse -Force \"{session_dir}\"\n"
                f"     cmd        : rmdir /s /q \"{session_dir}\"\n"
                f"  → or call   : await agent.unlink()"
            )
        else:
            logs_dir = self.config.resolved_sessions_dir() / "logs"
            tip = (
                "No QR was produced at all — the bridge could not reach WhatsApp.\n"
                f"  → Check bridge logs: {logs_dir}\n"
                f"  → Common causes: no internet, firewall/proxy blocking WhatsApp web,\n"
                f"     or Node < 18."
            )
        return QRTimeoutError(f"{head}\n{tip}")

    async def unlink(self) -> None:
        """Unlink this device and wipe stored credentials for a fresh QR next run."""
        session_dir = self.config.resolved_sessions_dir() / self.config.session_name
        if self._bridge is not None and self._bridge.running:
            try:
                await self._bridge.logout()
            except WaAgentError:
                pass
        import shutil

        shutil.rmtree(session_dir, ignore_errors=True)
        self._ready.clear()
        self._bot_jid = None
        self._bot_name = None
        print("🔓 Unlinked. Next start() will show a fresh pairing QR.")

    async def run_forever(self) -> None:
        """Serve incoming messages until :meth:`stop` is awaited."""
        if not self.is_connected:
            await self.start()
        self._install_signal_handlers()
        print(f"\n✅ {self._label()} connected. Waiting for messages… (Ctrl+C to quit)\n")
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._stop_event.set()
        if self._batcher is not None:
            self._batcher.cancel_all()
        cancelled_jobs = self.scheduler.cancel_all()
        if cancelled_jobs:
            log.info("Cancelled %d scheduled job(s)", cancelled_jobs)
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()
        if self._bridge is not None:
            await self._bridge.stop()
        for provider in [self._provider, *self._extra_providers.values()]:
            if provider is not None:
                await provider.aclose()
        self._extra_providers.clear()
        if self._prouter is not None:
            await self._prouter.aclose()
        self._ready.clear()

    def usage_summary(self) -> dict[str, Any] | None:
        """Token/cost report per routed endpoint (None without a router)."""
        return self._prouter.usage_summary() if self._prouter is not None else None

    async def __aenter__(self) -> "WhatsAppAgent":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    def run(self) -> None:
        """Synchronous convenience wrapper around ``start()`` + ``run_forever()``."""
        try:
            asyncio.run(self._run_managed())
        except KeyboardInterrupt:
            pass

    async def _run_managed(self) -> None:
        attempts = max(1, self.config.qr_max_attempts)
        try:
            for attempt in range(1, attempts + 1):
                try:
                    await self.start()
                    break
                except QRTimeoutError as exc:
                    if attempt == attempts:
                        raise
                    print(
                        f"\n⚠️  {exc}\n"
                        f"↻  Attempt {attempt + 1}/{attempts} — generating a fresh QR…"
                    )
                    await asyncio.sleep(1.0)
            await self.run_forever()
        except KeyboardInterrupt:
            await self.stop()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, self._request_stop)
            except NotImplementedError:  # pragma: no cover - Windows
                continue

    def _request_stop(self) -> None:
        self._stop_event.set()

    def _label(self) -> str:
        number = jid_to_number(self._bot_jid) if self._bot_jid else "?"
        name = self._bot_name or "WhatsApp"
        return f"{name} (+{number})"

    # ---------------------------------------------------------------- sending

    async def send_text(self, to: str, text: str) -> SentReceipt:
        if self._bridge is None:
            raise WaAgentError("Agent is not started")
        msg_id = await self._bridge.send_text(to, text)
        return SentReceipt(id=msg_id, to=to)

    async def send_image(
        self,
        to: str,
        source: str | Path | bytes,
        caption: str = "",
    ) -> SentReceipt:
        data_b64, mime = await self._load_image(source)
        return await self._send_media(to, "image", data_b64, mimetype=mime, caption=caption)

    async def send_document(
        self,
        to: str,
        path: str | Path | bytes,
        filename: str | None = None,
        caption: str = "",
    ) -> SentReceipt:
        data, name = self._load_bytes(path, filename)
        data_b64 = base64.b64encode(data).decode("ascii")
        mime = detect_image_mime(data) or "application/octet-stream"
        if name and name.lower().endswith(".pdf"):
            mime = "application/pdf"
        return await self._send_media(to, "document", data_b64, mimetype=mime, filename=name, caption=caption)

    async def send_audio(self, to: str, path: str | Path | bytes, voice_note: bool = False) -> SentReceipt:
        data, name = self._load_bytes(path, None)
        data_b64 = base64.b64encode(data).decode("ascii")
        mime = "audio/ogg; codecs=opus" if voice_note else "audio/mpeg"
        return await self._send_media(to, "audio", data_b64, mimetype=mime, ptt=voice_note)

    async def send_typing(self, chat_jid: str, on: bool = True) -> None:
        if self._bridge is None:
            return
        await self._bridge.set_presence("composing" if on else "paused", chat_jid)

    async def broadcast(self, jids: list[str], text: str) -> list[SentReceipt]:
        receipts: list[SentReceipt] = []
        for jid in jids:
            receipts.append(await self.send_text(jid, text))
            await asyncio.sleep(0.6)
        return receipts

    async def send_campaign(
        self,
        jids: list[str],
        text: str | Callable[[str], str],
        *,
        min_delay: float | None = None,
        max_delay: float | None = None,
        skip_opted_out: bool = True,
    ) -> dict[str, int]:
        """Human-paced bulk messaging with opt-out compliance.

        Delays are randomized between ``min_delay`` and ``max_delay`` seconds
        (defaults from AgentConfig: 6–15s). Opted-out numbers are skipped.
        """
        import random

        cfg = self.config
        lo = max(0.05, min_delay if min_delay is not None else cfg.campaign_min_delay)
        hi = max(lo, max_delay if max_delay is not None else cfg.campaign_max_delay)
        report = CampaignReport()

        total = len(jids)
        for index, jid in enumerate(jids, start=1):
            if skip_opted_out and self.safety.is_blocked(jid):
                report.skipped_opted_out += 1
                continue
            payload = text(jid) if callable(text) else text
            try:
                await self.send_text(jid, payload)
                report.sent += 1
            except WaAgentError as exc:
                log.warning("Campaign send to %s failed: %s", jid, exc)
                report.failed += 1
            self.safety.record_outbound(jid)
            log.info("Campaign progress %d/%d (sent=%d skipped=%d failed=%d)",
                     index, total, report.sent, report.skipped_opted_out, report.failed)
            if index < total:
                await asyncio.sleep(random.uniform(lo, hi))
        return report.as_dict()

    async def _send_media(self, *args: Any, **kwargs: Any) -> SentReceipt:
        if self._bridge is None:
            raise WaAgentError("Agent is not started")
        msg_id = await self._bridge.send_media(*args, **kwargs)
        return SentReceipt(id=msg_id, to=args[0])

    @staticmethod
    def _load_bytes(source: str | Path | bytes, filename: str | None) -> tuple[bytes, str | None]:
        if isinstance(source, bytes):
            return source, filename
        path = Path(source)
        return path.read_bytes(), filename or path.name

    @staticmethod
    async def _load_image(source: str | Path | bytes) -> tuple[str, str]:
        data = source if isinstance(source, bytes) else Path(source).read_bytes()
        prepared, mime = prepare_image(data)
        return prepared, mime

    # ---------------------------------------------------------------- events

    async def _on_bridge_event(self, event: str, payload: dict) -> None:
        if event == "qr":
            await self._handle_qr(payload.get("qr", ""))
        elif event == "ready":
            await self._handle_ready(payload)
        elif event == "message":
            message = IncomingMessage.model_validate(payload.get("payload") or {})
            self._spawn_background(self._process_message(message))
        elif event == "disconnected":
            reason = payload.get("reason", "")
            level = log.warning if payload.get("logged_out") else log.info
            level("WhatsApp connection closed (%s); bridge will reconnect", reason)
            if payload.get("logged_out"):
                self._ready.clear()
        elif event == "fatal":
            log.error("Bridge fatal: %s", payload.get("error"))
        elif event == "hello":
            log.debug("Bridge handshake complete")

    async def _handle_qr(self, qr_data: str) -> None:
        if not qr_data:
            return
        self._last_qr = qr_data
        self._saw_qr = True
        self._qr_count += 1
        handled = False
        if self._qr_hook is not None:
            handled = bool(await self._qr_hook(qr_data))
        if not handled:
            if self._qr_count > 1:
                print(f"🔄 Previous code expired — fresh QR #{self._qr_count}:")
            print_pairing_qr(qr_data)

    async def _handle_ready(self, payload: dict) -> None:
        first_time = not self._ready.is_set()
        self._bot_jid = payload.get("jid")
        self._bot_name = payload.get("name")
        self._ready.set()
        if self._ready_hook is not None:
            try:
                await self._ready_hook()
            except Exception:  # noqa: BLE001
                log.exception("on_ready hook raised")
        if first_time:
            log.info("Linked as %s", self._label())

    def _spawn_background(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # ------------------------------------------------------------ message flow

    async def _process_message(self, message: IncomingMessage) -> None:
        cfg = self.config
        if message.from_me:
            return
        if message.chat_jid == "status@broadcast":
            return
        if message.is_group and cfg.ignore_groups:
            return
        if not cfg.is_chat_allowed(message.chat_jid):
            return

        async with self._semaphore:
            if cfg.human_batching:
                await self._intake_message(message)
            else:
                lock = self._chat_locks.setdefault(message.chat_jid, asyncio.Lock())
                async with lock:
                    await self._reply_pipeline(message)

    async def _intake_message(self, message: IncomingMessage) -> None:
        """Immediate per-message work (opt-outs, gate, triggers, hooks) then batch."""
        bot_number = jid_to_number(self._bot_jid) if self._bot_jid else None

        opt = self.safety.detect_opt_language(message.body_text)
        if opt == "out":
            if self.safety.set_blocked(message.chat_jid, True):
                await self.send_text(
                    message.chat_jid,
                    "You've been unsubscribed. Send 'START' any time to receive "
                    "messages again.",
                )
            return
        if opt == "in" and self.safety.is_blocked(message.chat_jid):
            self.safety.set_blocked(message.chat_jid, False)
            await self.send_text(message.chat_jid, "Welcome back! You're subscribed again. ✅")
            return

        decision = self.safety.gate(message, bot_number=bot_number)
        if not decision.allowed:
            log.debug("Safety gate denied %s (%s)", message.id, decision.reason)
            return

        if self.config.mark_read and self._bridge is not None:
            try:
                await self._bridge.mark_read(message.id, message.chat_jid, message.sender_jid)
            except WaAgentError:
                pass

        trigger_reply = await self.triggers.match(message)
        if trigger_reply is not None:
            if trigger_reply.strip():
                await self.safety.pause_before_send()
                await self.send_text(message.chat_jid, trigger_reply)
            return

        if self._message_hook is not None:
            try:
                hook_reply = await self._message_hook(message)
            except Exception:  # noqa: BLE001
                log.exception("on_message hook raised; falling back to AI reply")
                hook_reply = None
            if isinstance(hook_reply, str) and hook_reply.strip():
                await self.safety.pause_before_send()
                await self.send_text(message.chat_jid, hook_reply)
                return

        await self._batcher.submit(message)

    async def _prepare_batched_context(self, message: IncomingMessage):
        use_router = self._prouter is not None
        vision_ok = (
            self._prouter.any_supports_vision() if use_router
            else getattr(self._provider_for(self.config.llm), "supports_vision", True)
        )
        return await self._build_user_message(message, vision_ok=vision_ok)

    async def _flush_batched_reply(
        self,
        merged: ChatMessage,
        last_raw: IncomingMessage,
        ids: list[str],
    ) -> None:
        chat_jid = last_raw.chat_jid
        signature = (chat_jid, tuple(ids))
        if signature in self._committed_batches:
            log.debug("Batch %s already committed; skipping", signature[1][-1])
            return
        self._committed_batches.add(signature)
        self._committed_order.append(signature)
        while len(self._committed_order) == self._committed_order.maxlen:
            old = self._committed_order.popleft()
            self._committed_batches.discard(old)

        route = self.router.resolve(last_raw)
        self.memory.append(chat_jid, merged)

        token_jid = current_chat_jid.set(chat_jid)
        token_mid = current_message_id.set(ids[-1] if ids else "")
        if self.config.typing_indicator:
            await self.send_typing(chat_jid, True)
        try:
            reply_text = await self._generate(chat_jid, route=route)
        except ProviderError as exc:
            log.error("LLM provider error: %s", exc)
            reply_text = f"⚠️ My AI backend had a problem: {str(exc)[:300]}"
        finally:
            if self.config.typing_indicator:
                await self.send_typing(chat_jid, False)
            current_chat_jid.reset(token_jid)
            current_message_id.reset(token_mid)

        reply_text = reply_text.strip()
        if reply_text:
            self.memory.append(
                chat_jid, ChatMessage(role="assistant", content=reply_text)
            )
            await self.safety.pause_before_send()
            # once we start sending, an interruption must not split the message
            await asyncio.shield(self.send_text(chat_jid, reply_text))

    async def _reply_pipeline(self, message: IncomingMessage) -> None:
        cfg = self.config
        bot_number = jid_to_number(self._bot_jid) if self._bot_jid else None

        opt = self.safety.detect_opt_language(message.body_text)
        if opt == "out":
            if self.safety.set_blocked(message.chat_jid, True):
                await self.send_text(
                    message.chat_jid,
                    "You've been unsubscribed. Send 'START' any time to receive messages again.",
                )
            return
        if opt == "in" and self.safety.is_blocked(message.chat_jid):
            self.safety.set_blocked(message.chat_jid, False)
            await self.send_text(message.chat_jid, "Welcome back! You're subscribed again. ✅")
            return

        decision = self.safety.gate(message, bot_number=bot_number)
        if not decision.allowed:
            log.debug("Safety gate denied %s (%s)", message.id, decision.reason)
            return

        if cfg.mark_read and self._bridge is not None:
            try:
                await self._bridge.mark_read(message.id, message.chat_jid, message.sender_jid)
            except WaAgentError:
                pass

        trigger_reply = await self.triggers.match(message)
        if trigger_reply is not None:
            if trigger_reply.strip():
                await self.safety.pause_before_send()
                await self.send_text(message.chat_jid, trigger_reply)
            return

        if self._message_hook is not None:
            try:
                hook_reply = await self._message_hook(message)
            except Exception:  # noqa: BLE001
                log.exception("on_message hook raised; falling back to AI reply")
                hook_reply = None
            if isinstance(hook_reply, str) and hook_reply.strip():
                await self.safety.pause_before_send()
                await self.send_text(message.chat_jid, hook_reply)
                return

        route = self.router.resolve(message)
        use_router = self._prouter is not None and not (route and route.llm)
        if use_router:
            vision_ok = self._prouter.any_supports_vision()
            provider_for_media = None
        else:
            provider_for_media = self._provider_for(
                route.llm if route and route.llm else self.config.llm
            )
            vision_ok = getattr(provider_for_media, "supports_vision", True)
        try:
            user_msg = await self._build_user_message(
                message, provider=provider_for_media, vision_ok=vision_ok
            )
        except (MediaError, UnsupportedDocumentError) as exc:
            await self.send_text(message.chat_jid, f"⚠️ Could not process your attachment: {exc}")
            return
        except Exception:  # noqa: BLE001
            log.exception("Failed to build context for %s", message.id)
            return

        if user_msg is None:
            return

        self.memory.append(message.chat_jid, user_msg)
        token_jid = current_chat_jid.set(message.chat_jid)
        token_mid = current_message_id.set(message.id)
        if cfg.typing_indicator:
            await self.send_typing(message.chat_jid, True)
        try:
            reply_text = await self._generate(message.chat_jid, route=route)
        except ProviderError as exc:
            log.error("LLM provider error: %s", exc)
            reply_text = f"⚠️ My AI backend had a problem: {str(exc)[:300]}"
        finally:
            if cfg.typing_indicator:
                await self.send_typing(message.chat_jid, False)
            current_chat_jid.reset(token_jid)
            current_message_id.reset(token_mid)

        reply_text = reply_text.strip()
        if reply_text:
            self.memory.append(message.chat_jid, ChatMessage(role="assistant", content=reply_text))
            await self.safety.pause_before_send()
            await self.send_text(message.chat_jid, reply_text)

    async def _build_user_message(
        self,
        message: IncomingMessage,
        *,
        provider: BaseChatProvider | None = None,
        vision_ok: bool | None = None,
    ) -> ChatMessage | None:
        cfg = self.config
        if vision_ok is None:
            vision_ok = getattr(provider, "supports_vision", True) if provider else True
        blocks: list[dict[str, Any]] = []
        caption = (message.caption or "").strip()
        body = (message.text or "").strip()

        if message.has_media and message.media_type == MediaType.IMAGE:
            if cfg.handle_images and self._bridge is not None and vision_ok:
                raw_b64, _mime = await self._bridge.download_media(message.id)
                img_b64, img_mime = prepare_image(base64.b64decode(raw_b64), max_side=cfg.max_image_side_px)
                blocks.append(image_block(img_mime, img_b64))
            else:
                caption = f"{caption}\n(The user sent an image; it is not shown to this model.)".strip()

        if message.has_media and message.media_type == MediaType.DOCUMENT \
                and cfg.handle_documents and self._bridge is not None:
            raw_b64, _mime = await self._bridge.download_media(message.id)
            doc = parse_document(
                message.filename or "document",
                base64.b64decode(raw_b64),
                max_chars=cfg.max_document_chars,
            )
            blocks.append(text_block(format_document_block(doc)))

        prompt_parts = [p for p in (body, caption) if p]
        trigger = self.config.require_trigger
        if body and trigger and body.lower().startswith(trigger.lower()):
            body = body[len(trigger):].lstrip()
            prompt_parts = [p for p in (body, caption) if p]
        if blocks:
            blocks.append(text_block(prompt_parts[-1] if prompt_parts else "(see attachment above)"))
            return ChatMessage(role="user", content=blocks)

        text = "\n".join(prompt_parts).strip()
        if not text:
            note = self._media_placeholder(message)
            if note is None:
                return None
            text = note
        return ChatMessage(role="user", content=text)

    @staticmethod
    def _media_placeholder(message: IncomingMessage) -> str | None:
        mapping = {
            MediaType.AUDIO: "(The user sent a voice note. Transcription is not configured — "
            "politely say you cannot listen yet.)",
            MediaType.VIDEO: "(The user sent a video. Politely acknowledge it; you can only read "
            "text captions.)",
            MediaType.STICKER: "(The user sent a sticker.)",
            MediaType.CONTACT: message.text,
            MediaType.LOCATION: message.text,
        }
        return mapping.get(message.media_type, message.text)

    def _system_prompt(self, base: str | None = None) -> str:
        today = datetime.now().strftime("%A, %d %B %Y (%H:%M local)")
        cfg = self.config

        can: list[str] = []
        if cfg.handle_documents:
            can.append(
                "- Parse and deeply analyze attached documents (PDF, DOCX, Markdown, "
                "TXT, CSV, JSON, HTML). Their extracted text is handed to you inside "
                "<document> tags — treat it as fully readable and quote from it freely."
            )
        if cfg.handle_images:
            can.append(
                "- See and reason about images, photos and screenshots people send."
            )
        if len(self.tools):
            can.append(
                "- Call these tools whenever they help: "
                + ", ".join(sorted(self.tools.names()))
                + "."
            )
        can.append("- Know the current date and time.")

        cannot: list[str] = [
            "- Listen to voice notes or watch videos. If one arrives, say politely "
            "that you cannot process audio/video yet."
        ]
        if not cfg.handle_documents:
            cannot.append("- Read attached documents (document parsing is disabled).")
        if not cfg.handle_images:
            cannot.append("- See images.")

        lines = [
            base if base is not None else cfg.system_prompt,
            "",
            f"Current date/time: {today}.",
            "You are chatting over WhatsApp: keep messages short and conversational.",
            "",
            "You ARE capable of all of the following — never deny or doubt them:",
            *can,
            "",
            "You are NOT able to:",
            *cannot,
        ]

        if cfg.handle_documents:
            lines += [
                "",
                "When a document is attached, answer like this:",
                "1. Start with a short summary (2-5 sentences): what the file is and "
                "its main takeaways.",
                "2. Then a section titled '📋 Details' with a clean structured "
                "breakdown: headings, bullets, key numbers/tables — organized, never "
                "a raw text dump.",
                "3. Close by offering to go deeper on any part.",
                "4. If the extracted text is empty or garbled (scanned file), say "
                "exactly that instead of inventing contents.",
            ]

        return "\n".join(lines)

    def _provider_for(self, llm: LLMConfig) -> BaseChatProvider:
        """Provider cache — one HTTP client per (provider, model) combo."""
        if llm is self.config.llm:
            if self._provider is None:
                self._provider = create_provider(llm)
            return self._provider
        key = f"{llm.provider}:{llm.model}:{llm.resolved_base_url}"
        if key not in self._extra_providers:
            self._extra_providers[key] = create_provider(llm)
        return self._extra_providers[key]

    def _tools_for(self, route: Any) -> tuple[ToolRegistry, list[dict[str, Any]]]:
        if not route or not route.tools:
            return self.tools, self.tools.schemas()
        merged = ToolRegistry()
        for t in self.tools.items():
            merged.register(t)
        for t in route.tools:
            merged.register(t)
        return merged, merged.schemas()

    async def _generate(self, chat_jid: str, route: Any = None) -> str:
        registry, schemas = self._tools_for(route)
        use_router = self._prouter is not None and not (route and route.llm)

        if use_router:
            state = {"endpoint": None}

            async def do_chat(messages, tools_list):
                result, endpoint = await self._prouter.chat(
                    messages, tools_list, pinned=state["endpoint"]
                )
                state["endpoint"] = endpoint.name
                self._last_endpoint_name = endpoint.name
                return result

            effective_schemas = schemas if schemas else None
        else:
            provider = self._provider_for(route.llm if route and route.llm else self.config.llm)
            supports_tools = bool(len(registry)) and getattr(provider, "supports_tools", True)
            effective_schemas = schemas if supports_tools else None

            async def do_chat(messages, tools_list):
                return await provider.chat(messages, tools=tools_list)

        conversation = [
            ChatMessage(role="system", content=self._system_prompt(
                route.system_prompt if route else None
            )),
            *self.memory.history(chat_jid),
        ]

        result_text = ""
        for _iteration in range(max(1, self.config.max_tool_iterations)):
            result = await do_chat(conversation, effective_schemas)

            if not result.has_tool_calls and effective_schemas \
                    and _DEGENERATE_TOOL_RE.search(result.text or ""):
                log.warning(
                    "Provider narrated about tools instead of answering; "
                    "retrying the same turn without tools"
                )
                result = await do_chat(conversation, None)

            if not result.has_tool_calls:
                result_text = result.text
                break

            conversation.append(
                ChatMessage(role="assistant", content=result.text or None, tool_calls=result.tool_calls)
            )
            for call in result.tool_calls:
                tool_obj = registry.get(call.name)
                if tool_obj is None:
                    output = f"Error: unknown tool '{call.name}'"
                else:
                    log.info("Tool call: %s(%s)", call.name, call.args)
                    output = await tool_obj.run(call.args)
                    log.debug("Tool %s -> %.200s", call.name, output)
                conversation.append(
                    ChatMessage(
                        role="tool",
                        content=output,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )
        else:
            result_text = ""

        return result_text


def agent_from_config(**kwargs: Any) -> WhatsAppAgent:
    """Build an :class:`WhatsAppAgent` straight from keyword configuration."""
    llm_kwargs = {
        key: kwargs.pop(key)
        for key in ("provider", "model", "api_key", "base_url", "temperature", "max_tokens")
        if key in kwargs
    }
    config = replace_config(AgentConfig(llm=LLMConfig(**llm_kwargs)), **kwargs)
    return WhatsAppAgent(config)
