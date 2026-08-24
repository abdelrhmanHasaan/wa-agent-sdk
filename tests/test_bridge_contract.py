"""Contract test: the Python model must accept EXACTLY what bridge.mjs emits.

Mirrors normalize() from node_bridge/bridge.mjs field-for-field so a rename on
either side can never slip through again.
"""

from wa_agent_sdk.models import IncomingMessage, MediaType


def bridge_payload(**overrides):
    """Byte-for-byte shape of normalize()'s return in bridge.mjs."""
    payload = {
        "id": "AC13BCC807BDDD539",
        "chat_jid": "15551234567@s.whatsapp.net",
        "sender_jid": "15551234567@s.whatsapp.net",
        "push_name": "Test User",
        "from_me": False,
        "is_group": False,
        "media_type": "text",
        "text": "hello bot",
        "caption": None,
        "mimetype": None,
        "filename": None,
        "has_media": False,
        "mentioned_jids": [],
        "quoted_participant": None,
        "timestamp": 1787571410000,
    }
    payload.update(overrides)
    return payload


def test_text_message_from_real_bridge_shape():
    msg = IncomingMessage.model_validate(bridge_payload())
    assert msg.chat_jid == "15551234567@s.whatsapp.net"
    assert msg.media_type is MediaType.TEXT
    assert msg.body_text == "hello bot"


def test_legacy_jid_key_still_validates():
    payload = bridge_payload()
    payload["jid"] = payload.pop("chat_jid")
    msg = IncomingMessage.model_validate(payload)
    assert msg.chat_jid == "15551234567@s.whatsapp.net"


def test_group_document_with_mentions():
    msg = IncomingMessage.model_validate(
        bridge_payload(
            chat_jid="12036302@g.us",
            sender_jid="15559999999@s.whatsapp.net",
            is_group=True,
            media_type="document",
            text=None,
            caption="what does this say?",
            mimetype="application/pdf",
            filename="report.pdf",
            has_media=True,
            mentioned_jids=["15550000001@s.whatsapp.net"],
            quoted_participant="15550000001:12@s.whatsapp.net",
        )
    )
    assert msg.is_group and msg.media_type is MediaType.DOCUMENT
    assert msg.filename == "report.pdf" and msg.has_media
    assert msg.caption == "what does this say?"


def test_image_caption_only():
    msg = IncomingMessage.model_validate(
        bridge_payload(media_type="image", text=None, caption="nice view",
                       mimetype="image/jpeg", has_media=True)
    )
    assert msg.media_type is MediaType.IMAGE
    assert msg.body_text == "nice view"


def test_unknown_extra_keys_ignored():
    payload = bridge_payload(some_future_field="x", another=123)
    IncomingMessage.model_validate(payload)  # must not raise
