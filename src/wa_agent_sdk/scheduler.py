"""Lightweight in-process scheduling: reminders and recurring messages."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

log = logging.getLogger("wa_agent.scheduler")

Sender = Callable[[str, str], Awaitable[Any]]
TextSource = Any  # str or callable -> str (may be async)


class Job:
    """Handle for a scheduled job; call :meth:`cancel` to stop it."""

    def __init__(self, task: asyncio.Task) -> None:
        self._task = task

    def cancel(self) -> bool:
        return self._task.cancel()

    @property
    def done(self) -> bool:
        return self._task.done()


class Scheduler:
    """Sends messages later or repeatedly through a sender coroutine."""

    def __init__(self, sender: Sender) -> None:
        self._sender = sender
        self._jobs: set[asyncio.Task] = set()

    @property
    def active(self) -> int:
        return sum(1 for j in self._jobs if not j.done())

    def _spawn(self, coro) -> Job:
        task = asyncio.get_running_loop().create_task(coro)
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)
        return Job(task)

    async def _send_once(self, delay: float, chat_jid: str, text: TextSource) -> None:
        await asyncio.sleep(max(0.0, delay))
        payload = text(chat_jid) if callable(text) else text
        if inspect.isawaitable(payload):
            payload = await payload
        await self._sender(chat_jid, str(payload))

    async def _loop_every(
        self,
        interval: float,
        chat_jid: str,
        text: TextSource,
        run_immediately: bool,
    ) -> None:
        if run_immediately:
            await self._send_once(0.0, chat_jid, text)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._send_once(0.0, chat_jid, text)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - keep the recurrence alive
                log.exception("Recurring job failed for %s", chat_jid)

    def every(
        self,
        interval: float,
        chat_jid: str,
        text: TextSource,
        *,
        run_immediately: bool = False,
    ) -> Job:
        """Send *text* to *chat_jid* every *interval* seconds, forever."""
        return self._spawn(self._loop_every(interval, chat_jid, text, run_immediately))

    def remind_after(
        self,
        delay: float | timedelta,
        chat_jid: str,
        text: TextSource,
    ) -> Job:
        seconds = delay.total_seconds() if isinstance(delay, timedelta) else float(delay)
        return self._spawn(self._send_once(seconds, chat_jid, text))

    def at(self, when: datetime, chat_jid: str, text: TextSource) -> Job:
        delay = (when - datetime.now()).total_seconds()
        return self._spawn(self._send_once(delay, chat_jid, text))

    def cancel_all(self) -> int:
        cancelled = 0
        for task in list(self._jobs):
            if not task.done() and task.cancel():
                cancelled += 1
        self._jobs.clear()
        return cancelled


@dataclass(slots=True)
class CampaignReport:
    sent: int = 0
    skipped_opted_out: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "sent": self.sent,
            "skipped_opted_out": self.skipped_opted_out,
            "failed": self.failed,
        }
