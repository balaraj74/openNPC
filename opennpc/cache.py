"""Decision cache with TTL for reducing redundant computation.

When an agent's state hasn't changed meaningfully within a TTL window, the
cache returns the previous decision instead of re-running the full pipeline.

This is especially useful in high-agent-count scenes where many NPCs are idle
or moving slowly — the cache avoids re-scoring goals, re-querying memory, and
re-running the policy for identical states.

Usage::

    from opennpc.cache import DecisionCache

    cache = DecisionCache(ttl_seconds=0.5, tolerance=0.05)

    # First call computes normally
    hit = cache.get(config, state)  # None — cache miss

    # Store the decision
    cache.put(config, state, decision)

    # Identical state within TTL — cache hit
    hit = cache.get(config, state)  # returns the stored decision
"""

from __future__ import annotations

import hashlib
import json
from time import monotonic
from typing import Any

from opennpc.types import ActionDecision, AgentConfig, GameState


def _quantize(value: float, tolerance: float) -> float:
    """Round a float to the nearest tolerance step.

    This lets us treat states that differ by less than ``tolerance`` as
    equivalent for caching purposes.
    """
    if tolerance <= 0:
        return value
    return round(value / tolerance) * tolerance


def _state_fingerprint(state: GameState, tolerance: float) -> str:
    """Produce a stable hash from the gameplay-relevant parts of a state.

    Fields that change every frame but don't affect decisions (like exact
    timestamps or render data) are excluded.  Numeric fields are quantized
    so that micro-jitter doesn't bust the cache.
    """
    key_parts = {
        "health": _quantize(state.health, tolerance),
        "stamina": _quantize(state.stamina, tolerance),
        "threat": _quantize(state.threat_level, tolerance),
        "distance": _quantize(state.distance_to_target or 0.0, tolerance),
        "target_health": _quantize(state.target_health or 0.0, tolerance),
        "cover": state.cover_available,
        "nearby": sorted(state.nearby_entities),
    }
    raw = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


class CacheEntry:
    """A single cached decision with timestamp."""

    __slots__ = ("decision", "fingerprint", "created_at")

    def __init__(self, decision: ActionDecision, fingerprint: str) -> None:
        self.decision = decision
        self.fingerprint = fingerprint
        self.created_at = monotonic()


class DecisionCache:
    """TTL-based cache for agent decisions.

    Parameters:
        ttl_seconds: How long a cached entry is valid (default 0.5s).
        tolerance: Numeric quantization step for state comparison. States
            differing by less than this in any single field are treated as
            identical (default 0.05).
        max_entries: Maximum number of agents to cache. When exceeded the
            oldest entries are evicted (default 10000).
    """

    def __init__(
        self,
        ttl_seconds: float = 0.5,
        tolerance: float = 0.05,
        max_entries: int = 10_000,
    ) -> None:
        self.ttl_seconds = max(0.0, ttl_seconds)
        self.tolerance = max(0.0, tolerance)
        self.max_entries = max(1, max_entries)
        self._entries: dict[str, CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get(self, config: AgentConfig, state: GameState) -> ActionDecision | None:
        """Return a cached decision if the state fingerprint matches and TTL hasn't expired."""
        entry = self._entries.get(config.agent_id)
        if entry is None:
            self._misses += 1
            return None

        age = monotonic() - entry.created_at
        if age > self.ttl_seconds:
            self._misses += 1
            del self._entries[config.agent_id]
            return None

        fingerprint = _state_fingerprint(state, self.tolerance)
        if fingerprint != entry.fingerprint:
            self._misses += 1
            return None

        self._hits += 1
        return ActionDecision(
            agent_id=config.agent_id,
            action=entry.decision.action,
            confidence=max(0.01, entry.decision.confidence - 0.02),
            reason=f"Cached decision (age={age:.3f}s). {entry.decision.reason}",
            memory_update=None,
            trace=entry.decision.trace,
        )

    def put(self, config: AgentConfig, state: GameState, decision: ActionDecision) -> None:
        """Store a decision in the cache."""
        if len(self._entries) >= self.max_entries:
            self._evict_oldest()
        fingerprint = _state_fingerprint(state, self.tolerance)
        self._entries[config.agent_id] = CacheEntry(decision, fingerprint)

    def invalidate(self, agent_id: str) -> None:
        """Remove a specific agent's cached decision."""
        self._entries.pop(agent_id, None)

    def clear(self) -> None:
        """Clear all cached decisions."""
        self._entries.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache performance statistics."""
        total = self._hits + self._misses
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            "ttl_seconds": self.ttl_seconds,
            "tolerance": self.tolerance,
        }

    def _evict_oldest(self) -> None:
        """Remove the oldest cache entry."""
        if not self._entries:
            return
        oldest_key = min(self._entries, key=lambda k: self._entries[k].created_at)
        del self._entries[oldest_key]
