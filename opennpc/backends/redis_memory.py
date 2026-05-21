"""Redis-backed memory store for OpenNPC.

Stores agent memory events in Redis sorted sets, enabling shared memory
across multiple game-server processes or distributed deployments.

Requirements
------------
    pip install 'opennpc[backends]'
    # or
    pip install redis>=4

Usage
-----
    from opennpc.backends.redis_memory import RedisMemoryStore
    from opennpc.decision import DecisionEngine

    store = RedisMemoryStore(host="localhost", port=6379, db=0)
    engine = DecisionEngine(memory_store=store)

Redis Key Schema
----------------
  ``opennpc:mem:{agent_id}``  — Sorted set, score = timestamp, value = JSON event.
  ``opennpc:important:{agent_id}`` — Sorted set, score = importance, value = JSON event.

Both sets are capped to ``max_events`` members to bound memory usage.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from opennpc.types import MemoryEvent

try:
    import redis as redis_lib  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "RedisMemoryStore requires 'redis-py'. Install it with: "
        "pip install 'opennpc[backends]' or pip install redis>=4"
    ) from exc

if TYPE_CHECKING:
    from redis import Redis


class RedisMemoryStore:
    """Agent memory store backed by Redis.

    Parameters
    ----------
    host:
        Redis server hostname. Defaults to ``localhost``.
    port:
        Redis server port. Defaults to ``6379``.
    db:
        Redis database index. Defaults to ``0``.
    max_events:
        Maximum number of events retained per agent per sorted set.
        Older events are evicted once this limit is reached.
    key_prefix:
        Namespace prefix for all Redis keys. Defaults to ``opennpc:mem``.
    client:
        Optionally inject a pre-configured ``redis.Redis`` client directly
        (useful for testing with ``fakeredis``).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        max_events: int = 512,
        key_prefix: str = "opennpc:mem",
        client: "Redis | None" = None,
    ) -> None:
        self._r: Redis = client or redis_lib.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
        )
        self.max_events = max_events
        self._prefix = key_prefix

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _time_key(self, agent_id: str) -> str:
        return f"{self._prefix}:time:{agent_id}"

    def _importance_key(self, agent_id: str) -> str:
        return f"{self._prefix}:imp:{agent_id}"

    # ------------------------------------------------------------------
    # MemoryStore interface
    # ------------------------------------------------------------------

    def add(self, event: MemoryEvent) -> None:
        """Persist a memory event for ``event.agent_id``."""
        payload = json.dumps({
            "agent_id": event.agent_id,
            "text": event.text,
            "importance": event.importance,
            "tags": event.tags,
            "timestamp": event.timestamp,
            "metadata": event.metadata,
        })
        pipe = self._r.pipeline(transaction=True)
        time_key = self._time_key(event.agent_id)
        imp_key = self._importance_key(event.agent_id)

        # Store in time-ordered set (score = timestamp).
        pipe.zadd(time_key, {payload: event.timestamp})
        # Store in importance-ordered set (score = importance).
        pipe.zadd(imp_key, {payload: event.importance})

        # Trim both sets to max_events (keep most recent / most important).
        pipe.zremrangebyrank(time_key, 0, -(self.max_events + 1))
        pipe.zremrangebyrank(imp_key, 0, -(self.max_events + 1))

        pipe.execute()

    def recent(self, agent_id: str, limit: int = 8) -> list[MemoryEvent]:
        """Return up to ``limit`` most-recent events (newest first)."""
        raw = self._r.zrevrange(self._time_key(agent_id), 0, limit - 1)
        return [self._deserialize(r) for r in raw]

    def important(
        self,
        agent_id: str,
        minimum_importance: float = 0.7,
        limit: int = 8,
    ) -> list[MemoryEvent]:
        """Return up to ``limit`` events with importance ≥ ``minimum_importance``."""
        raw = self._r.zrevrangebyscore(
            self._importance_key(agent_id),
            "+inf",
            minimum_importance,
            start=0,
            num=limit,
        )
        return [self._deserialize(r) for r in raw]

    def summarize(self, agent_id: str, limit: int = 8) -> str:
        """Produce a concise text summary blending important and recent events."""
        recent = self.recent(agent_id, limit=limit)
        important = self.important(agent_id, limit=limit)
        seen: set[str] = set()
        fragments: list[str] = []
        for event in important + recent:
            if event.text in seen:
                continue
            seen.add(event.text)
            label = "important" if event.importance >= 0.7 else "recent"
            fragments.append(f"{label}: {event.text}")
        return " | ".join(fragments)

    def clear(self, agent_id: str | None = None) -> None:
        """Delete memory for a specific agent (or ALL agents if ``None``)."""
        if agent_id is None:
            pattern = f"{self._prefix}:*"
            cursor = 0
            while True:
                cursor, keys = self._r.scan(cursor, match=pattern, count=200)
                if keys:
                    self._r.delete(*keys)
                if cursor == 0:
                    break
        else:
            self._r.delete(
                self._time_key(agent_id),
                self._importance_key(agent_id),
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def agent_count(self) -> int:
        """Return number of distinct agents with stored memory."""
        cursor = 0
        agents: set[str] = set()
        prefix = f"{self._prefix}:time:"
        while True:
            cursor, keys = self._r.scan(cursor, match=f"{prefix}*", count=200)
            for key in keys:
                agent_id = key[len(prefix):]
                agents.add(agent_id)
            if cursor == 0:
                break
        return len(agents)

    def ping(self) -> bool:
        """Return ``True`` if Redis is reachable."""
        try:
            return bool(self._r.ping())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _deserialize(raw: str) -> MemoryEvent:
        data: dict[str, Any] = json.loads(raw)
        return MemoryEvent(
            agent_id=data["agent_id"],
            text=data["text"],
            importance=float(data["importance"]),
            tags=data.get("tags", []),
            timestamp=float(data.get("timestamp", time.time())),
            metadata=data.get("metadata", {}),
        )
