"""Anti-ban guardrails.

WhatsApp bans numbers that behave like bots: instant replies, unlimited
messaging to strangers, ignoring STOP requests, activity at 3 AM. This module
implements the counter-measures:

* per-chat cooldown + daily caps (with a stricter cap for brand-new chats)
* global hourly cap
* quiet hours (e.g. never reply between 23:00 and 07:00)
* opt-out / opt-in keyword compliance with a persistent blocklist
* optional trigger prefix ("!bot") so the agent only talks when addressed
* group mention-only mode
* humanized random pauses before sending

All counters are persisted under ``.wa_data/`` so limits survive restarts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .models import IncomingMessage

log = logging.getLogger("wa_agent.safety")

DEFAULT_OPT_OUT_KEYWORDS = frozenset({"stop", "unsubscribe", "remove me", "cancel"})
DEFAULT_OPT_IN_KEYWORDS = frozenset({"start", "subscribe", "unstop"})


@dataclass(slots=True, frozen=True)
class GateDecision:
    allowed: bool
    reason: str = ""


ALLOWED = GateDecision(True)


def _number(jid: str) -> str:
    return jid.split("@")[0].split(":")[0]


def _parse_hhmm(value: str) -> int:
    hours, minutes = value.strip().split(":")
    return int(hours) * 60 + int(minutes)


class SafetyManager:
    """Stateful gate evaluated before every outbound reply."""

    def __init__(
        self,
        data_dir: Path,
        *,
        enabled: bool = True,
        group_mention_only: bool = True,
        require_trigger: str | None = None,
        reply_cooldown: float = 3.0,
        global_hourly_limit: int = 80,
        per_chat_daily_limit: int = 50,
        new_chat_daily_limit: int = 12,
        quiet_hours: tuple[str, str] | None = None,
        humanize_min_delay: float = 0.8,
        humanize_max_delay: float = 3.0,
    ) -> None:
        self.enabled = enabled
        self.group_mention_only = group_mention_only
        self.require_trigger = require_trigger
        self.reply_cooldown = max(0.0, reply_cooldown)
        self.global_hourly_limit = max(1, global_hourly_limit)
        self.per_chat_daily_limit = max(1, per_chat_daily_limit)
        self.new_chat_daily_limit = new_chat_daily_limit
        self.quiet_hours = quiet_hours or None
        self.humanize_min_delay = max(0.0, humanize_min_delay)
        self.humanize_max_delay = max(self.humanize_min_delay, humanize_max_delay)

        data_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = Path(data_dir) / "safety.json"
        self._blocked_path = Path(data_dir) / "optouts.json"
        self._state: dict = {
            "day": "",
            "chats": {},
            "hourly": {"start": 0.0, "count": 0},
        }
        loaded = self._load_json(self._state_path)
        if isinstance(loaded, dict) and "chats" in loaded:
            self._state.update(loaded)
        self._blocked: set[str] = set(self._load_json(self._blocked_path) or [])
        self._last_sent: dict[str, float] = {}
        if self.quiet_hours:
            try:
                self._quiet_range = (
                    _parse_hhmm(self.quiet_hours[0]),
                    _parse_hhmm(self.quiet_hours[1]),
                )
            except Exception as exc:  # noqa: BLE001 - bad config must not crash
                log.warning("Invalid quiet_hours %r (%s); disabled", self.quiet_hours, exc)
                self._quiet_range = None
        else:
            self._quiet_range = None

    # ------------------------------------------------------------- persistence

    @staticmethod
    def _load_json(path: Path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _save_json(path: Path, data) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)

    # ---------------------------------------------------------------- opt-outs

    def is_blocked(self, chat_jid: str) -> bool:
        return _number(chat_jid) in self._blocked

    def set_blocked(self, chat_jid: str, blocked: bool) -> bool:
        """Persist an opt-out. Returns True when the state changed."""
        number = _number(chat_jid)
        changed = blocked != (number in self._blocked)
        if changed:
            if blocked:
                self._blocked.add(number)
            else:
                self._blocked.discard(number)
            self._save_json(self._blocked_path, sorted(self._blocked))
            log.info("Opt-out updated for +%s: blocked=%s", number, blocked)
        return changed

    def detect_opt_language(self, text: str | None) -> str | None:
        """Return ``"out"``, ``"in"`` or None based on keywords in *text*."""
        if not text:
            return None
        normalized = re.sub(r"[^a-z\s]", " ", text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return None
        for kw in DEFAULT_OPT_OUT_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", normalized):
                return "out"
        for kw in DEFAULT_OPT_IN_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", normalized):
                return "in"
        return None

    # ------------------------------------------------------------------- gate

    def in_quiet_hours(self, now: datetime | None = None) -> bool:
        if self._quiet_range is None:
            return False
        current = (now or datetime.now()).hour * 60 + (now or datetime.now()).minute
        start, end = self._quiet_range
        if start <= end:
            return start <= current < end
        return current >= start or current < end  # overnight window

    def _rollover_day(self, today: str) -> None:
        if self._state.get("day") == today:
            return
        previous = self._state.get("chats", {})
        self._state["chats"] = {
            jid: {"count": 0, "first_seen": info.get("first_seen", today)}
            for jid, info in previous.items()
            if isinstance(info, dict)
        }
        self._state["day"] = today

    def gate(self, msg: IncomingMessage, *, bot_number: str | None = None) -> GateDecision:
        """Decide whether the agent may reply to this message.

        When allowed, local counters are already reserved for the reply.
        """
        if not self.enabled:
            return ALLOWED

        chat = msg.chat_jid
        if self.is_blocked(chat):
            return GateDecision(False, "opted_out")

        body = (msg.body_text or "").strip()
        if self.require_trigger:
            if not body.lower().startswith(self.require_trigger.lower()):
                return GateDecision(False, "trigger_required")

        if msg.is_group and self.group_mention_only:
            mentioned = {_number(j) for j in msg.mentioned_jids}
            quoted = _number(msg.quoted_participant) if msg.quoted_participant else ""
            addressed = bot_number in mentioned or (quoted and quoted == bot_number)
            triggered = bool(self.require_trigger) and body.lower().startswith(
                self.require_trigger.lower()
            )
            if not (addressed or triggered):
                return GateDecision(False, "group_not_addressed")

        if self.in_quiet_hours():
            return GateDecision(False, "quiet_hours")

        now = datetime.now().timestamp()
        last = self._last_sent.get(chat)
        if last is not None and now - last < self.reply_cooldown:
            return GateDecision(False, "cooldown")

        today = datetime.now().date().isoformat()
        self._rollover_day(today)

        hourly = self._state["hourly"]
        if now - float(hourly.get("start", 0.0)) >= 3600:
            hourly["start"], hourly["count"] = now, 0
        if hourly["count"] >= self.global_hourly_limit:
            return GateDecision(False, "global_hourly_limit")

        chat_info = self._state["chats"].setdefault(
            chat, {"count": 0, "first_seen": today}
        )
        limit = self.per_chat_daily_limit
        if self.new_chat_daily_limit > 0 and chat_info.get("first_seen") == today:
            limit = min(limit, self.new_chat_daily_limit)
        if chat_info["count"] >= limit:
            return GateDecision(False, "chat_daily_limit")

        hourly["count"] += 1
        chat_info["count"] += 1
        self._last_sent[chat] = now
        self._save_json(self._state_path, self._state)
        return ALLOWED

    def record_outbound(self, chat_jid: str) -> None:
        """Account for manual/campaign sends so they share the same budget."""
        if not self.enabled:
            return
        today = datetime.now().date().isoformat()
        self._rollover_day(today)
        hourly = self._state["hourly"]
        now = datetime.now().timestamp()
        if now - float(hourly.get("start", 0.0)) >= 3600:
            hourly["start"], hourly["count"] = now, 0
        hourly["count"] += 1
        info = self._state["chats"].setdefault(chat_jid, {"count": 0, "first_seen": today})
        info["count"] += 1
        self._last_sent[chat_jid] = now
        self._save_json(self._state_path, self._state)

    # ------------------------------------------------------------------ pacing

    async def pause_before_send(self) -> float:
        """Humanized random pause; returns how long it slept."""
        delay = random.uniform(self.humanize_min_delay, self.humanize_max_delay)
        await asyncio.sleep(delay)
        return delay

    def stats(self) -> dict:
        today = datetime.now().date().isoformat()
        self._rollover_day(today)
        hourly = dict(self._state["hourly"])
        return {"day": today, "hourly": hourly, "chats": len(self._state["chats"]),
                "opted_out": len(self._blocked)}
