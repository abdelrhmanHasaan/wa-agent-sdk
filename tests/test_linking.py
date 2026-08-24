"""QR-linking failure diagnostics and unlink behaviour."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from wa_agent_sdk import LLMConfig, WhatsAppAgent
from wa_agent_sdk.exceptions import QRTimeoutError


def make_agent(tmp: Path, session_name="default") -> WhatsAppAgent:
    return WhatsAppAgent(
        llm=LLMConfig(provider="openai", model="m", api_key="k"),
        session_name=session_name,
        sessions_dir=tmp / "wa_sessions",
        data_dir=tmp / "data",
    )


def test_timeout_message_when_qr_was_shown():
    tmp = Path(tempfile.mkdtemp())
    agent = make_agent(tmp)
    agent._saw_qr = True
    err = agent._qr_failure_error()
    assert isinstance(err, QRTimeoutError)
    assert "never scanned" in str(err)


def test_timeout_message_detects_stale_session():
    tmp = Path(tempfile.mkdtemp())
    agent = make_agent(tmp)
    session_dir = tmp / "wa_sessions" / "default"
    session_dir.mkdir(parents=True)
    (session_dir / "creds.json").write_text("{}")  # stale credentials present

    err = str(agent._qr_failure_error())
    assert "stale or corrupted" in err
    assert "agent.unlink()" in err
    assert "creds.json" not in err  # advice names the folder, not file details
    assert "wa_sessions" in err


def test_timeout_message_points_at_logs_when_no_qr_and_no_session():
    tmp = Path(tempfile.mkdtemp())
    agent = make_agent(tmp)
    err = str(agent._qr_failure_error())
    assert "No QR was produced" in err
    assert "logs" in err


def test_unlink_wipes_session_folder_without_bridge():
    tmp = Path(tempfile.mkdtemp())
    agent = make_agent(tmp)
    session_dir = tmp / "wa_sessions" / "default"
    session_dir.mkdir(parents=True)
    (session_dir / "creds.json").write_text("{}")
    agent._ready.set()
    agent._bot_jid = "1555@s.whatsapp.net"

    asyncio.run(agent.unlink())

    assert not session_dir.exists()
    assert not agent.is_connected
    assert agent.bot_jid is None


def test_start_is_idempotent_when_already_connected():
    tmp = Path(tempfile.mkdtemp())
    agent = make_agent(tmp)
    agent._ready.set()  # pretend already linked; no bridge exists

    asyncio.run(agent.start())  # must early-return, not raise about missing bridge
    assert agent.is_connected
