"""MessageBatcher tests: burst coalescing, debounce, interruption, max-wait."""

import asyncio
import time

import pytest

from wa_agent_sdk.batching import MessageBatcher, merge_messages
from wa_agent_sdk.llm.base import ChatMessage
from wa_agent_sdk.models import IncomingMessage

COUNTER = iter(f"mid{i:03d}" for i in range(1, 1000))


def make_msg(text: str, chat: str = "15551234567@s.whatsapp.net") -> IncomingMessage:
    return IncomingMessage(
        id=next(COUNTER), chat_jid=chat, sender_jid=chat, text=text,
        media_type="text",
    )


class Harness:
    def __init__(self, *, window=0.25, max_wait=5.0, flush_delay=0.0):
        self.flushes: list[tuple[str, list[str]]] = []  # (merged_text, ids)
        self.builds: list[str] = []
        self.cancelled_count = 0
        self.first_flush_at: float | None = None
        self._flush_delay = flush_delay
        self._first_call_started = asyncio.Event()
        self.batcher = MessageBatcher(
            window=window,
            max_wait=max_wait,
            build=self._build,
            on_flush=self._flush,
        )

    async def _build(self, message):
        self.builds.append(message.text or "")
        return ChatMessage(role="user", content=message.text or "")

    async def _flush(self, merged, last_raw, ids):
        idx = len(self.flushes)
        self._first_call_started.set()
        was_cancelled = False
        try:
            if idx == 0 and self._flush_delay:
                await asyncio.sleep(self._flush_delay)   # simulated "thinking"
        except asyncio.CancelledError:
            was_cancelled = True
            raise
        finally:
            self.cancelled_count += int(was_cancelled)
            if self.first_flush_at is None:
                self.first_flush_at = time.monotonic()
            self.flushes.append((merged.content, list(ids)))

    async def submit(self, text, chat=None):
        await self.batcher.submit(make_msg(text, chat or "c1@s.whatsapp.net"))

    @staticmethod
    async def wait_until(predicate, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.02)
        return False


@pytest.mark.asyncio
async def test_burst_coalesces_into_single_flush_in_order():
    h = Harness(window=0.2)
    await h.submit("one")
    await asyncio.sleep(0.05)
    await h.submit("two")
    await asyncio.sleep(0.05)
    await h.submit("three")

    assert await Harness.wait_until(lambda: len(h.flushes) == 1)
    merged_text, ids = h.flushes[0]
    assert merged_text.index("one") < merged_text.index("two") < merged_text.index("three")
    assert len(ids) == 3
    assert not await Harness.wait_until(lambda: len(h.flushes) > 1, timeout=0.4)


@pytest.mark.asyncio
async def test_trailing_edge_debounce_resets_window():
    h = Harness(window=0.4)
    await h.submit("first")
    await asyncio.sleep(0.25)          # inside the window…
    assert len(h.flushes) == 0          # …not flushed yet
    await h.submit("second")           # resets the timer
    assert await Harness.wait_until(lambda: len(h.flushes) == 1)
    assert "first" in h.flushes[0][0] and "second" in h.flushes[0][0]


@pytest.mark.asyncio
async def test_interrupt_cancels_generation_and_requeues_unsent():
    h = Harness(window=0.15, flush_delay=5.0)  # first flush "thinks" for 5 s
    await h.submit("message one")
    await asyncio.wait_for(h._first_call_started.wait(), 2)   # generation started
    await h.submit("message two")                              # INTERRUPT

    assert await Harness.wait_until(lambda: len(h.flushes) == 2, timeout=4)
    assert h.cancelled_count == 1
    second_merged, second_ids = h.flushes[-1]
    assert "message one" in second_merged      # re-queued in front ✓
    assert "message two" in second_merged      # new message included ✓
    assert len(second_ids) == 2


@pytest.mark.asyncio
async def test_max_wait_cap_flushes_despite_continuous_trickle():
    h = Harness(window=10.0, max_wait=0.9)     # window huge; cap forces flush
    start = time.monotonic()
    for i in range(6):
        await h.submit(f"trickle {i}")
        await asyncio.sleep(0.25)

    assert await Harness.wait_until(lambda: len(h.flushes) >= 1, timeout=2)
    assert h.first_flush_at - start < 5         # didn't wait the full 10 s window
    # let any follow-up capped batches settle, then every message must exist
    assert await Harness.wait_until(
        lambda: sum(text.count("trickle") for text, _ in h.flushes) >= 6,
        timeout=3,
    )
    all_text = " ".join(text for text, _ in h.flushes)
    assert all(f"trickle {i}" in all_text for i in range(6))


@pytest.mark.asyncio
async def test_separate_chats_batch_independently():
    h = Harness(window=0.2)
    await h.submit("a", chat="111@s.whatsapp.net")
    await h.submit("b", chat="222@s.whatsapp.net")
    assert await Harness.wait_until(lambda: len(h.flushes) == 2)
    texts = sorted(t for t, _ in h.flushes)
    assert "a" in texts[0] and "b" in texts[1]


@pytest.mark.asyncio
async def test_build_returning_none_drops_message():
    class Empty(Harness):
        async def _build(self, message):
            self.builds.append(message.text or "")
            return None if (message.text or "") == "skip" else \
                ChatMessage(role="user", content=message.text or "")

    h = Empty(window=0.15)
    await h.submit("skip")
    await h.submit("keep")
    assert await Harness.wait_until(lambda: len(h.flushes) == 1)
    assert "skip" not in h.flushes[0][0]


def test_merge_single_returns_identity():
    msg = ChatMessage(role="user", content="only")
    assert merge_messages([msg]) is msg


def test_merge_multiple_with_blocks():
    items = [
        ChatMessage(role="user", content="plain text"),
        ChatMessage(role="user", content=[{"type": "image",
                                           "mime_type": "image/jpeg",
                                           "data": "AA"}]),
    ]
    merged = merge_messages(items)
    assert isinstance(merged.content, list)
    kinds = [b["type"] for b in merged.content]
    assert "image" in kinds
