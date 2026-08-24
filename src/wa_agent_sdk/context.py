"""Process-wide context shared between the agent loop and tool handlers."""

from contextvars import ContextVar

current_chat_jid: ContextVar[str | None] = ContextVar("current_chat_jid", default=None)
"""JID of the chat whose message is currently being processed.

Built-in tools such as ``remember_note`` use this to scope data per conversation.
"""
current_message_id: ContextVar[str | None] = ContextVar("current_message_id", default=None)
