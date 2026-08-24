"""Per-chat conversation memory with automatic trimming."""

from __future__ import annotations

import json
from typing import Any

from .llm.base import ChatMessage


def _msg_chars(msg: ChatMessage) -> int:
    if msg.content is None:
        return 0
    if isinstance(msg.content, str):
        return len(msg.content)
    total = 0
    for block in msg.content:
        if block.get("type") == "text":
            total += len(block.get("text", ""))
        else:
            total += 4
    return total


class ConversationMemory:
    """Stores canonical :class:`ChatMessage` history keyed by chat JID."""

    def __init__(self, max_messages: int = 40, max_chars: int = 60_000) -> None:
        self.max_messages = max_messages
        self.max_chars = max_chars
        self._store: dict[str, list[ChatMessage]] = {}

    def append(self, chat_jid: str, message: ChatMessage) -> None:
        self._store.setdefault(chat_jid, []).append(message)
        self._trim(chat_jid)

    def extend(self, chat_jid: str, messages: list[ChatMessage]) -> None:
        for m in messages:
            self.append(chat_jid, m)

    def history(self, chat_jid: str) -> list[ChatMessage]:
        return list(self._store.get(chat_jid, ()))

    def clear(self, chat_jid: str) -> None:
        self._store.pop(chat_jid, None)

    def clear_all(self) -> None:
        self._store.clear()

    def chats(self) -> list[str]:
        return list(self._store)

    def _trim(self, chat_jid: str) -> None:
        msgs = self._store.get(chat_jid)
        if not msgs:
            return
        while len(msgs) > self.max_messages or sum(_msg_chars(m) for m in msgs) > self.max_chars:
            if len(msgs) <= 1:
                break
            msgs.pop(0)
        self._store[chat_jid] = msgs

    def dump(self, chat_jid: str) -> str:
        payload: list[dict[str, Any]] = [
            {"role": m.role, "content": m.content} for m in self.history(chat_jid)
        ]
        return json.dumps(payload, ensure_ascii=False, default=str)
