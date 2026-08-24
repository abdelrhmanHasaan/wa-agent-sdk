"""Human-like message batching.

Real people don't reply to every text the instant it lands: they see three
messages arrive one after another, wait for the burst to stop, then answer
everything at once — and if a new message interrupts them mid-thought, they
fold it into the same answer.

``MessageBatcher`` gives the agent exactly that behaviour:

* **trailing-edge debounce** — every new message restarts a short window
  (default 6 s); the reply fires only when the user goes quiet.
* **max-wait cap** — even a continuous trickle can't postpone forever
  (default 30 s from the first queued message).
* **interruption with memory** — if the model is already generating and
  another message arrives, the in-flight generation is cancelled and its
  not-yet-answered messages are re-queued in front, so the eventual single
  reply accounts for the entire burst.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .llm.base import ChatMessage, text_block
from .models import IncomingMessage

log = logging.getLogger("wa_agent.batching")

Builder = Callable[[IncomingMessage], Awaitable["ChatMessage | None"]]
Flusher = Callable[["ChatMessage", IncomingMessage, "list[str]"], Awaitable[None]]


def merge_messages(items: list[ChatMessage]) -> ChatMessage:
    """Collapse a burst of user turns into ONE user turn, order preserved."""
    if len(items) == 1:
        return items[0]

    header = f"(You sent {len(items)} messages in a row — here they are in order:)"
    seq: list[dict[str, Any]] = []
    has_blocks = any(isinstance(it.content, (list, tuple)) for it in items)

    for idx, item in enumerate(items, start=1):
        if isinstance(item.content, str):
            seq.append(text_block(f"— message {idx}:\n{item.content}"))
        else:
            seq.append(text_block(f"— message {idx}:"))
            seq.extend(item.content)
            seq.append(text_block(f"— (end of message {idx})"))

    if has_blocks:
        return ChatMessage(role="user", content=[text_block(header), *seq])
    body = "\n\n".join(block["text"] for block in seq)
    return ChatMessage(role="user", content=f"{header}\n\n{body}")


@dataclass(slots=True)
class _Entry:
    items: list[ChatMessage] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    raw_last: IncomingMessage | None = None
    first_ts: float | None = None
    timer: asyncio.TimerHandle | None = None
    flush_task: asyncio.Task | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MessageBatcher:
    """Per-chat debounce + interrupt queue. One callback per settled burst."""

    def __init__(
        self,
        *,
        window: float,
        max_wait: float,
        build: Builder,
        on_flush: Flusher,
    ) -> None:
        self.window = max(0.25, window)
        # NOTE: deliberately NOT clamped to window — a short hard cap with a
        # long debounce window is a supported combination (trickle protection).
        self.max_wait = max(0.25, max_wait)
        self._build = build
        self._on_flush = on_flush
        self._entries: dict[str, _Entry] = {}

    # ------------------------------------------------------------------ stats

    def pending_chats(self) -> int:
        return sum(1 for e in self._entries.values() if e.items)

    def pending_messages(self) -> int:
        return sum(len(e.items) for e in self._entries.values())

    # ------------------------------------------------------------------ intake

    async def submit(self, message: IncomingMessage) -> None:
        """Queue one incoming message (restarting the quiet-window timer)."""
        entry = self._entries.setdefault(message.chat_jid, _Entry())
        async with entry.lock:
            if entry.flush_task is not None and not entry.flush_task.done():
                log.info("New message interrupted an in-flight reply for %s",
                         message.chat_jid)
                entry.flush_task.cancel()
                entry.flush_task = None

            built = await self._build(message)
            if built is None:
                return

            entry.items.append(built)
            entry.ids.append(message.id)
            entry.raw_last = message

            now = time.monotonic()
            if entry.first_ts is None:
                entry.first_ts = now

            if entry.timer is not None:
                entry.timer.cancel()
            delay = max(
                0.15,
                min(self.window, self.max_wait - (now - entry.first_ts)),
            )
            loop = asyncio.get_running_loop()
            entry.timer = loop.call_later(delay, self._start_flush, message.chat_jid)
            log.debug("Batched msg for %s (pending=%d, flush in %.1fs)",
                      message.chat_jid, len(entry.items), delay)

    # ------------------------------------------------------------------- flush

    def _start_flush(self, chat_jid: str) -> None:
        entry = self._entries.get(chat_jid)
        if entry is None or not entry.items:
            return
        entry.first_ts = None
        entry.flush_task = asyncio.get_running_loop().create_task(
            self._run_flush(chat_jid, entry)
        )

    async def _run_flush(self, chat_jid: str, entry: _Entry) -> None:
        k = len(entry.items)
        items = entry.items[:k]
        ids = entry.ids[:k]
        last = entry.raw_last
        # take the items out of the queue while they're being answered;
        # re-queued below if we get interrupted or the flush fails
        entry.items = entry.items[k:]
        entry.ids = entry.ids[k:]

        merged = merge_messages(items)
        try:
            await self._on_flush(merged, last, ids)
        except asyncio.CancelledError:
            entry.items[:0] = items
            entry.ids[:0] = ids
            log.info("Reply for %s cancelled — %d message(s) re-queued",
                     chat_jid, len(items))
            raise
        except Exception:  # noqa: BLE001 - retry shortly rather than lose msgs
            log.exception("Flush failed for %s; retrying in 2s", chat_jid)
            entry.items[:0] = items
            entry.ids[:0] = ids
            loop = asyncio.get_running_loop()
            entry.timer = loop.call_later(2.0, self._start_flush, chat_jid)
        else:
            if not entry.items:
                entry.raw_last = None

    # ---------------------------------------------------------------- shutdown

    def cancel_all(self) -> int:
        cancelled = 0
        for entry in self._entries.values():
            if entry.timer is not None:
                entry.timer.cancel()
                entry.timer = None
            if entry.flush_task is not None and not entry.flush_task.done():
                entry.flush_task.cancel()
                cancelled += 1
        return cancelled
