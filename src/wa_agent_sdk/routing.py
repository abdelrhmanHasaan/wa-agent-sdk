"""Smart multi-provider routing.

Give the agent several API keys/endpoints and it decides per-message where to
send the request:

* **smart** (default) — scores query complexity and picks a tier
  (fast / balanced / smart), e.g. "hi" → Llama on Groq, 20-page PDF analysis →
  GPT-4o/Claude.
* **cheapest** — always the lowest estimated cost that can handle the query.
* **balanced** — round-robins load across affordable endpoints.
* **failover** — strict priority order; first healthy endpoint wins.

Cross-cutting features: daily USD budgets per endpoint, token/cost accounting
persisted to ``.wa_data/provider_usage.json``, automatic failover to the next
candidate when one provider errors, and a circuit breaker that quarantines
failing endpoints for a cooldown window.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .config import LLMConfig
from .exceptions import ProviderAuthError, ProviderError, WaAgentError
from .llm.base import BaseChatProvider, ChatMessage, ChatResult
from .llm.factory import create_provider

log = logging.getLogger("wa_agent.routing")

COMPLEXITY_WORDS = re.compile(
    r"\b(why|analy[sz]e|compare|explain|refactor|debug|strategy|design|"
    r"architect|prove|derive|evaluate|implications|step[- ]by[- ]step)\b",
    re.IGNORECASE,
)
CHARS_PER_TOKEN = 4.0


class Tier(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    SMART = "smart"


class Strategy(str, Enum):
    SMART = "smart"
    CHEAPEST = "cheapest"
    BALANCED = "balanced"
    FAILOVER = "failover"


_TIER_RANK = {Tier.FAST: 0, Tier.BALANCED: 1, Tier.SMART: 2}


def _coerce_tier(value: str | Tier) -> Tier:
    try:
        return value if isinstance(value, Tier) else Tier(str(value).lower())
    except ValueError as exc:
        raise WaAgentError(f"Unknown tier {value!r} (use fast/balanced/smart)") from exc


@dataclass(slots=True)
class ModelEndpoint:
    """One routable LLM endpoint.

    name                logical label shown in logs/usage reports
    llm                 full provider configuration
    tier                fast | balanced | smart  (used by the smart strategy)
    priority            ordering hint for the failover strategy
    price_input_per_m   USD per 1M input tokens   (0 = free/unknown)
    price_output_per_m  USD per 1M output tokens
    supports_vision     None = assume yes; set False to keep images away
    daily_budget_usd    skip this endpoint once today's spend reaches it
    max_input_chars     comfort limit; bigger queries are routed elsewhere
    """

    name: str
    llm: LLMConfig
    tier: str | Tier = Tier.BALANCED
    priority: int = 0
    price_input_per_m: float = 0.0
    price_output_per_m: float = 0.0
    supports_vision: bool | None = None
    daily_budget_usd: float | None = None
    max_input_chars: int | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        self.tier = _coerce_tier(self.tier)


@dataclass(slots=True)
class QueryProfile:
    """What the router knows about the request it must place."""

    est_input_chars: int = 0
    has_image: bool = False
    has_document: bool = False
    tool_schemas: int = 0
    history_messages: int = 0
    score: float = 0.0

    @property
    def needs_vision(self) -> bool:
        return self.has_image

    @property
    def target_tier(self) -> Tier:
        if self.score >= 0.67:
            return Tier.SMART
        if self.score >= 0.34:
            return Tier.BALANCED
        return Tier.FAST


def profile_query(messages: list[ChatMessage], tools_count: int = 0,
                  *, system_chars: int = 0) -> QueryProfile:
    total_chars = system_chars
    has_image = has_document = False
    history = 0
    last_user_text = ""

    for msg in messages:
        history += 1
        if isinstance(msg.content, list):
            for block in msg.content:
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text", "")
                    total_chars += len(text)
                    if "<document" in text:
                        has_document = True
                elif btype == "image":
                    has_image = True
                    total_chars += 800  # vision payload heuristic
        elif isinstance(msg.content, str):
            total_chars += len(msg.content)
            if "<document" in msg.content:
                has_document = True
            if msg.role == "user":
                last_user_text = msg.content

    score = min(total_chars / 6000.0, 1.0) * 0.45
    if has_image:
        score += 0.25
    if has_document:
        score += 0.20
    if tools_count:
        score += min(tools_count * 0.02, 0.10)
    if last_user_text and COMPLEXITY_WORDS.search(last_user_text):
        score += 0.15

    return QueryProfile(
        est_input_chars=total_chars,
        has_image=has_image,
        has_document=has_document,
        tool_schemas=tools_count,
        history_messages=history,
        score=min(score, 1.0),
    )


class UsageTracker:
    """Persistent per-day token/cost accounting per endpoint."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {"days": {}, "lifetime": {}}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._data.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass

    def _bucket(self, section: str, name: str) -> dict[str, float]:
        return self._data[section].setdefault(
            name, {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cost_usd": 0.0}
        )

    def record(self, endpoint: str, input_tokens: int, output_tokens: int,
               cost_usd: float) -> None:
        day = datetime.now().date().isoformat()
        day_bucket = self._data["days"].setdefault(day, {}).setdefault(endpoint, {})
        lifetime_bucket = self._bucket("lifetime", endpoint)
        for bucket in (day_bucket, lifetime_bucket):
            bucket["input_tokens"] = bucket.get("input_tokens", 0) + input_tokens
            bucket["output_tokens"] = bucket.get("output_tokens", 0) + output_tokens
            bucket["calls"] = bucket.get("calls", 0) + 1
            bucket["cost_usd"] = round(bucket.get("cost_usd", 0.0) + cost_usd, 6)
        self._save()

    def spend_today(self, endpoint: str) -> float:
        day = datetime.now().date().isoformat()
        info = self._data["days"].get(day, {}).get(endpoint)
        return float(info.get("cost_usd", 0.0)) if info else 0.0

    def calls_today(self, endpoint: str) -> int:
        day = datetime.now().date().isoformat()
        info = self._data["days"].get(day, {}).get(endpoint)
        return int(info.get("calls", 0)) if info else 0

    def summary(self) -> dict[str, Any]:
        day = datetime.now().date().isoformat()
        return {
            "today": {
                ep: dict(stats) for ep, stats in self._data["days"].get(day, {}).items()
            },
            "lifetime": {
                ep: dict(stats) for ep, stats in self._data["lifetime"].items()
            },
        }

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:  # noqa: BLE001 - accounting must never break replies
            log.warning("Could not persist usage stats to %s", self.path)


class _Health:
    """Tiny circuit breaker: N consecutive failures => cooldown."""

    def __init__(self, threshold: int = 3, cooldown_seconds: float = 300.0) -> None:
        self.threshold = threshold
        self.cooldown = cooldown_seconds
        self._consecutive: dict[str, int] = {}
        self._blocked_until: dict[str, float] = {}

    def is_ok(self, name: str) -> bool:
        return time.time() >= self._blocked_until.get(name, 0.0)

    def record_success(self, name: str) -> None:
        self._consecutive[name] = 0
        self._blocked_until.pop(name, None)

    def record_failure(self, name: str) -> None:
        strikes = self._consecutive.get(name, 0) + 1
        self._consecutive[name] = strikes
        if strikes >= self.threshold:
            self._blocked_until[name] = time.time() + self.cooldown
            log.warning("Endpoint '%s' quarantined for %.0fs after %d failures",
                        name, self.cooldown, strikes)

    def quarantine(self, name: str, multiplier: float = 6.0) -> None:
        self._blocked_until[name] = time.time() + self.cooldown * multiplier
        log.warning("Endpoint '%s' quarantined long-term (auth/config error)", name)


class ProviderRouter:
    """Routes each reply to the best available endpoint."""

    def __init__(
        self,
        endpoints: list[ModelEndpoint] | None = None,
        *,
        strategy: str | Strategy = Strategy.SMART,
        usage_tracker: UsageTracker | None = None,
        provider_factory: Callable[[LLMConfig], BaseChatProvider] = create_provider,
        failure_threshold: int = 3,
        cooldown_seconds: float = 300.0,
        est_output_tokens: int = 500,
    ) -> None:
        try:
            self.strategy = strategy if isinstance(strategy, Strategy) else Strategy(str(strategy).lower())
        except ValueError as exc:
            raise WaAgentError(f"Unknown routing strategy {strategy!r}") from exc
        self.usage = usage_tracker
        self._factory = provider_factory
        self.health = _Health(failure_threshold, cooldown_seconds)
        self.est_output_tokens = est_output_tokens
        self._endpoints: dict[str, ModelEndpoint] = {}
        self._providers: dict[str, BaseChatProvider] = {}
        self._rr = 0
        for ep in endpoints or []:
            self.add(ep)

    # ------------------------------------------------------------- registry

    def add(self, endpoint: ModelEndpoint) -> ModelEndpoint:
        if endpoint.name in self._endpoints:
            raise WaAgentError(f"Duplicate endpoint name '{endpoint.name}'")
        self._endpoints[endpoint.name] = endpoint
        return endpoint

    def remove(self, name: str) -> None:
        self._endpoints.pop(name, None)
        self._providers.pop(name, None)

    def get(self, name: str) -> ModelEndpoint | None:
        return self._endpoints.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._endpoints)

    def attach_storage(self, data_dir: Path) -> UsageTracker:
        if self.usage is None:
            self.usage = UsageTracker(Path(data_dir) / "provider_usage.json")
        return self.usage

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
        self._providers.clear()

    # ------------------------------------------------------------ internals

    def _provider_for(self, endpoint: ModelEndpoint) -> BaseChatProvider:
        if endpoint.name not in self._providers:
            self._providers[endpoint.name] = self._factory(endpoint.llm)
        return self._providers[endpoint.name]

    def _est_cost(self, endpoint: ModelEndpoint, profile: QueryProfile) -> float:
        in_tokens = profile.est_input_chars / CHARS_PER_TOKEN
        return (in_tokens * endpoint.price_input_per_m +
                self.est_output_tokens * endpoint.price_output_per_m) / 1_000_000

    def _actual_cost(self, endpoint: ModelEndpoint, result: ChatResult) -> float:
        return (result.input_tokens * endpoint.price_input_per_m +
                result.output_tokens * endpoint.price_output_per_m) / 1_000_000

    def _can_handle(self, endpoint: ModelEndpoint, profile: QueryProfile) -> bool:
        if not endpoint.enabled:
            return False
        if not self.health.is_ok(endpoint.name):
            return False
        if profile.has_image and endpoint.supports_vision is False:
            return False
        if endpoint.max_input_chars and profile.est_input_chars > endpoint.max_input_chars:
            return False
        if endpoint.daily_budget_usd is not None and self.usage is not None:
            if self.usage.spend_today(endpoint.name) >= endpoint.daily_budget_usd:
                return False
        return True

    def select(self, profile: QueryProfile) -> list[ModelEndpoint]:
        """Ordered candidate list according to the active strategy."""
        capable = [ep for ep in self._endpoints.values() if self._can_handle(ep, profile)]
        if not capable:
            raise ProviderError(
                "ProviderRouter: no healthy endpoint available "
                "(all disabled, budget-capped, or in failure cooldown)"
            )

        if self.strategy is Strategy.FAILOVER:
            return sorted(capable, key=lambda ep: (-ep.priority, ep.name))

        costs = {ep.name: self._est_cost(ep, profile) for ep in capable}
        if self.strategy is Strategy.CHEAPEST:
            return sorted(capable, key=lambda ep: (costs[ep.name], ep.name))

        if self.strategy is Strategy.BALANCED:
            loads = {ep.name: (self.usage.calls_today(ep.name) if self.usage else 0)
                     for ep in capable}
            self._rr += 1
            return sorted(capable, key=lambda ep: (loads[ep.name], costs[ep.name],
                                                   (ep.name.__hash__() ^ self._rr) % 7))

        # SMART: pick the target tier, upgrade if empty, then order by cost
        want = _TIER_RANK[profile.target_tier]
        exact = [ep for ep in capable if _TIER_RANK[ep.tier] == want]
        pool = exact
        if not pool:
            higher = sorted((ep for ep in capable if _TIER_RANK[ep.tier] > want),
                            key=lambda ep: _TIER_RANK[ep.tier])
            pool = list(higher[:1]) if higher else capable
        if want == _TIER_RANK[Tier.SMART]:
            pool = sorted(pool, key=lambda ep: (-_TIER_RANK[ep.tier], costs[ep.name]))
        else:
            pool = sorted(pool, key=lambda ep: (costs[ep.name], ep.name))
        return pool

    # ------------------------------------------------------------------ chat

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        *,
        pinned: str | None = None,
        profile: QueryProfile | None = None,
    ) -> tuple[ChatResult, ModelEndpoint]:
        """Run one completion through the best endpoint; returns (result, endpoint).

        ``pinned`` keeps a multi-step tool conversation on the same endpoint
        after the first successful call.
        """
        q = profile or profile_query(messages, len(tools or []))
        candidates = self.select(q)
        if pinned and pinned in self._endpoints:
            preferred = self._endpoints[pinned]
            candidates = [preferred] + [c for c in candidates if c.name != pinned]

        last_exc: Exception | None = None
        for endpoint in candidates:
            try:
                result = await self._provider_for(endpoint).chat(messages, tools)
                self.health.record_success(endpoint.name)
                if self.usage is not None:
                    self.usage.record(
                        endpoint.name,
                        result.input_tokens,
                        result.output_tokens,
                        self._actual_cost(endpoint, result),
                    )
                log.info("Routed to '%s' (%s/%s, score=%.2f)",
                         endpoint.name, endpoint.llm.provider, endpoint.llm.model, q.score)
                return result, endpoint
            except ProviderAuthError as exc:
                last_exc = exc
                self.health.quarantine(endpoint.name)
                log.error("Endpoint '%s' auth failed: %s", endpoint.name, exc)
            except ProviderError as exc:
                last_exc = exc
                self.health.record_failure(endpoint.name)
                log.warning("Endpoint '%s' failed (%s); trying next candidate",
                            endpoint.name, exc)
            except Exception as exc:  # noqa: BLE001 - unknown errors still fail over
                last_exc = exc
                self.health.record_failure(endpoint.name)
                log.exception("Unexpected error on '%s'", endpoint.name)

        raise ProviderError(
            f"All {len(candidates)} routed endpoints failed. Last error: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------- reporting

    def any_supports_vision(self) -> bool:
        return any(
            ep.enabled and ep.supports_vision is not False
            for ep in self._endpoints.values()
        )

    def usage_summary(self) -> dict[str, Any]:
        if self.usage is None:
            return {"today": {}, "lifetime": {}, "note": "no usage recorded yet"}
        report = self.usage.summary()
        for section in ("today", "lifetime"):
            for name, stats in report[section].items():
                ep = self._endpoints.get(name)
                if ep is not None:
                    stats["budget_remaining"] = (
                        None if ep.daily_budget_usd is None
                        else round(max(0.0, ep.daily_budget_usd - stats.get("cost_usd", 0.0)), 4)
                    )
        return report
