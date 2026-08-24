"""Multi-agent routing and zero-cost keyword triggers.

* :class:`AgentRouter` — send different chats/keywords to different AI
  personas, each with its own system prompt, model and toolset.
* :class:`TriggerBoard` — instant rule-based auto-replies that never touch
  the LLM (FAQ answers, commands, greetings).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Pattern, Union

from .config import LLMConfig
from .models import IncomingMessage

log = logging.getLogger("wa_agent.router")

Matcher = Union[str, Pattern, Callable[[IncomingMessage], bool], None]
ReplyFactory = Union[str, Callable[[IncomingMessage], Any]]


def _matches(matcher: Matcher, message: IncomingMessage) -> bool:
    if matcher is None:
        return True
    if isinstance(matcher, re.Pattern):
        return bool(matcher.search(message.body_text or ""))
    if isinstance(matcher, str):
        return matcher.lower() in (message.body_text or "").lower()
    return bool(matcher(message))


@dataclass
class Route:
    """One agent persona inside a router."""

    name: str
    match: Matcher = None
    system_prompt: str | None = None
    llm: LLMConfig | None = None
    tools: list[Any] = field(default_factory=list)
    priority: int = 0


class AgentRouter:
    """Ordered collection of :class:`Route` entries with a fallback."""

    def __init__(self) -> None:
        self._routes: list[Route] = []

    def add_route(
        self,
        name: str,
        match: Matcher = None,
        *,
        system_prompt: str | None = None,
        llm: LLMConfig | None = None,
        tools: list[Any] | None = None,
        priority: int = 0,
    ) -> Route:
        route = Route(
            name=name,
            match=match,
            system_prompt=system_prompt,
            llm=llm,
            tools=list(tools or []),
            priority=priority,
        )
        self._routes.append(route)
        return route

    def get(self, name: str) -> Route | None:
        return next((r for r in self._routes if r.name == name), None)

    @property
    def routes(self) -> list[Route]:
        return list(self._routes)

    def resolve(self, message: IncomingMessage) -> Route | None:
        """Highest-priority matching route; falls back to the catch-all."""
        ordered = sorted(self._routes, key=lambda r: -r.priority)
        fallback: Route | None = None
        for route in ordered:
            if route.match is None:
                if fallback is None:
                    fallback = route
                continue
            try:
                if _matches(route.match, message):
                    log.debug("Route '%s' matched", route.name)
                    return route
            except Exception:  # noqa: BLE001 - user matcher must not crash the loop
                log.exception("Route '%s' matcher raised", route.name)
        return fallback


class TriggerBoard:
    """Keyword/regex auto-replies evaluated before the LLM (zero cost)."""

    def __init__(self) -> None:
        self._items: list[tuple[int, dict]] = []

    def add(
        self,
        pattern: str | Pattern,
        reply: ReplyFactory,
        *,
        exact: bool = False,
        priority: int = 0,
    ) -> None:
        """Register a trigger.

        pattern: substring (case-insensitive), compiled regex, or exact word
        reply:   static string OR callable(msg) -> str (may be async)
        """
        self._items.append((priority, {"pattern": pattern, "reply": reply, "exact": exact}))
        self._items.sort(key=lambda item: -item[0])

    async def match(self, message: IncomingMessage) -> str | None:
        body = (message.body_text or "").strip()
        lowered = body.lower()
        for _, item in self._items:
            pattern = item["pattern"]
            hit = False
            if isinstance(pattern, re.Pattern):
                hit = bool(pattern.search(body))
            elif item["exact"]:
                hit = lowered == str(pattern).lower()
            else:
                hit = str(pattern).lower() in lowered
            if not hit:
                continue
            reply = item["reply"]
            if callable(reply):
                result = reply(message)
                if inspect.isawaitable(result):
                    result = await result
                return str(result) if result is not None else None
            return str(reply)
        return None
