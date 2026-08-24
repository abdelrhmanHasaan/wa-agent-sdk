"""Message models exchanged between the bridge, the agent and providers."""

from __future__ import annotations

import time
from enum import Enum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


def _now() -> float:
    return time.time()


class MediaType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    STICKER = "sticker"
    DOCUMENT = "document"
    CONTACT = "contact"
    LOCATION = "location"
    OTHER = "other"


class IncomingMessage(BaseModel):
    """A normalized WhatsApp message received from the bridge."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    chat_jid: str = Field(validation_alias=AliasChoices("chat_jid", "jid"))
    sender_jid: str
    push_name: str | None = None
    from_me: bool = False
    is_group: bool = False
    media_type: MediaType = MediaType.TEXT
    text: str | None = None
    caption: str | None = None
    mimetype: str | None = None
    filename: str | None = None
    has_media: bool = False
    mentioned_jids: list[str] = Field(default_factory=list)
    quoted_participant: str | None = None
    timestamp: float = Field(default_factory=_now)

    @property
    def display_sender(self) -> str:
        if self.push_name:
            return self.push_name
        return self.sender_jid.split("@")[0]

    @property
    def body_text(self) -> str:
        return self.text or self.caption or ""


class SentReceipt(BaseModel):
    """Confirmation that an outgoing message was accepted by WhatsApp."""

    id: str | None = None
    to: str = ""
    timestamp: float = Field(default_factory=_now)


def jid_to_number(jid: str) -> str:
    """``15551234567@s.whatsapp.net:12`` -> ``15551234567``."""
    base = jid.split("@")[0] if "@" in jid else jid
    return base.split(":")[0]
