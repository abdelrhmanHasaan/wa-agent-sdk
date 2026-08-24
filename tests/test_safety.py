"""Unit tests for the anti-ban SafetyManager."""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from wa_agent_sdk.models import IncomingMessage
from wa_agent_sdk.safety import SafetyManager


def make_msg(text="hello", chat="15551234567@s.whatsapp.net", is_group=False,
             mentioned=None, quoted=None):
    return IncomingMessage(
        id="m1", chat_jid=chat, sender_jid=chat, text=text, is_group=is_group,
        media_type="text", mentioned_jids=mentioned or [],
        quoted_participant=quoted,
    )


def new_manager(tmp: Path, **kw) -> SafetyManager:
    defaults = dict(
        enabled=True, group_mention_only=True, require_trigger=None,
        reply_cooldown=0.0, global_hourly_limit=100, per_chat_daily_limit=100,
        new_chat_daily_limit=0, quiet_hours=None,
        humanize_min_delay=0.0, humanize_max_delay=0.0,
    )
    defaults.update(kw)
    return SafetyManager(tmp / "data", **defaults)


def test_disabled_allows_everything():
    tmp = Path(tempfile.mkdtemp())
    m = SafetyManager(tmp / "d", enabled=False)
    for _ in range(500):
        assert m.gate(make_msg()).allowed
    shutil.rmtree(tmp, ignore_errors=True)


def test_trigger_required_gate_and_stripping():
    tmp = Path(tempfile.mkdtemp())
    m = new_manager(tmp, require_trigger="!bot")
    assert not m.gate(make_msg("hi there")).allowed
    d = m.gate(make_msg("!bot hi there"))
    assert d.allowed, d.reason
    shutil.rmtree(tmp, ignore_errors=True)


def test_per_chat_cooldown():
    tmp = Path(tempfile.mkdtemp())
    m = new_manager(tmp, reply_cooldown=60)
    assert m.gate(make_msg()).allowed
    d = m.gate(make_msg(chat="15551234567@s.whatsapp.net"))
    assert not d.allowed and d.reason == "cooldown"
    # other chats unaffected
    assert m.gate(make_msg(chat="15559999999@s.whatsapp.net")).allowed
    shutil.rmtree(tmp, ignore_errors=True)


def test_daily_limits_new_vs_known_chat():
    tmp = Path(tempfile.mkdtemp())
    m = new_manager(tmp, per_chat_daily_limit=50, new_chat_daily_limit=2)
    assert m.gate(make_msg()).allowed
    assert m.gate(make_msg()).allowed
    d = m.gate(make_msg())
    assert not d.allowed and d.reason == "chat_daily_limit"
    shutil.rmtree(tmp, ignore_errors=True)


def test_global_hourly_limit():
    tmp = Path(tempfile.mkdtemp())
    m = new_manager(tmp, global_hourly_limit=3)
    for _ in range(3):
        assert m.gate(make_msg(chat=f"1555000000{_}@s.whatsapp.net")).allowed
    d = m.gate(make_msg(chat="15550000099@s.whatsapp.net"))
    assert not d.allowed and d.reason == "global_hourly_limit"
    shutil.rmtree(tmp, ignore_errors=True)


def test_quiet_hours_overnight_window():
    tmp = Path(tempfile.mkdtemp())
    m = new_manager(tmp, quiet_hours=("23:00", "07:00"))
    assert m.in_quiet_hours(datetime(2026, 1, 1, 23, 30))
    assert m.in_quiet_hours(datetime(2026, 1, 1, 3, 0))
    assert m.in_quiet_hours(datetime(2026, 1, 1, 6, 59))
    assert m.in_quiet_hours(datetime(2026, 1, 1, 12, 0)) is False
    assert m.in_quiet_hours(datetime(2026, 1, 1, 7, 0)) is False
    shutil.rmtree(tmp, ignore_errors=True)


def test_opt_out_flow_persists():
    tmp = Path(tempfile.mkdtemp())
    m = new_manager(tmp)
    assert m.detect_opt_language("STOP!!!") == "out"
    assert m.detect_opt_language("please unsubscribe me") == "out"
    assert m.detect_opt_language("let's START over") == "in"
    assert m.detect_opt_language("what time do you open?") is None

    jid = "15551234567@s.whatsapp.net"
    assert m.set_blocked(jid, True) is True
    assert m.is_blocked(jid)

    m2 = SafetyManager(tmp / "data")  # fresh instance reads persisted blocklist
    assert m2.is_blocked(jid)
    d = m2.gate(make_msg(chat=jid))
    assert not d.allowed and d.reason == "opted_out"

    assert m2.set_blocked(jid, False) is True
    assert not m2.is_blocked(jid)
    shutil.rmtree(tmp, ignore_errors=True)


def test_group_mention_only_gate():
    bot = "15550000001"
    tmp = Path(tempfile.mkdtemp())
    m = new_manager(tmp, group_mention_only=True)

    ignored = make_msg("random chatter", chat="1203@g.us", is_group=True)
    assert not m.gate(ignored, bot_number=bot).allowed

    mention = make_msg(f"@{bot} help", chat="1203@g.us", is_group=True,
                       mentioned=[f"{bot}@s.whatsapp.net"])
    assert m.gate(mention, bot_number=bot).allowed

    quoted = make_msg("replying to bot", chat="1203@g.us", is_group=True,
                      quoted=f"{bot}:12@s.whatsapp.net")
    assert m.gate(quoted, bot_number=bot).allowed
    shutil.rmtree(tmp, ignore_errors=True)


def test_stats_shape():
    tmp = Path(tempfile.mkdtemp())
    m = new_manager(tmp)
    m.gate(make_msg())
    s = m.stats()
    assert {"day", "hourly", "chats", "opted_out"} <= set(s)
    assert s["hourly"]["count"] >= 1
    shutil.rmtree(tmp, ignore_errors=True)
