"""Tests for RedisMemoryStore — uses fakeredis to avoid a live Redis dependency."""

from __future__ import annotations

import time

import pytest

from opennpc.types import MemoryEvent

fakeredis = pytest.importorskip("fakeredis", reason="fakeredis not installed")


def _make_store():
    from opennpc.backends.redis_memory import RedisMemoryStore
    import fakeredis

    client = fakeredis.FakeRedis(decode_responses=True)
    return RedisMemoryStore(client=client)


def _event(agent_id: str, text: str, importance: float = 0.5) -> MemoryEvent:
    return MemoryEvent(
        agent_id=agent_id,
        text=text,
        importance=importance,
        tags=[],
        timestamp=time.time(),
        metadata={},
    )


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

class TestRedisMemoryStoreBasic:
    def test_add_and_recent(self):
        store = _make_store()
        store.add(_event("agent1", "patrol started"))
        store.add(_event("agent1", "enemy spotted"))

        recent = store.recent("agent1", limit=5)
        assert len(recent) == 2
        texts = [e.text for e in recent]
        assert "patrol started" in texts
        assert "enemy spotted" in texts

    def test_recent_limit_respected(self):
        store = _make_store()
        for i in range(10):
            store.add(_event("a", f"event {i}"))

        recent = store.recent("a", limit=3)
        assert len(recent) == 3

    def test_important_filter(self):
        store = _make_store()
        store.add(_event("a", "low", importance=0.2))
        store.add(_event("a", "critical", importance=0.9))

        important = store.important("a", minimum_importance=0.7)
        assert len(important) == 1
        assert important[0].text == "critical"

    def test_clear_agent(self):
        store = _make_store()
        store.add(_event("a", "foo"))
        store.add(_event("b", "bar"))
        store.clear("a")

        assert store.recent("a") == []
        assert len(store.recent("b")) == 1

    def test_clear_all(self):
        store = _make_store()
        store.add(_event("x", "event x"))
        store.add(_event("y", "event y"))
        store.clear()

        assert store.recent("x") == []
        assert store.recent("y") == []

    def test_summarize_non_empty(self):
        store = _make_store()
        store.add(_event("s", "ran away", importance=0.3))
        store.add(_event("s", "saw player", importance=0.8))
        summary = store.summarize("s")
        assert "saw player" in summary
        assert "ran away" in summary

    def test_summarize_empty(self):
        store = _make_store()
        assert store.summarize("nobody") == ""

    def test_max_events_cap(self):
        """Ensure the sorted sets are capped at max_events."""
        import fakeredis

        client = fakeredis.FakeRedis(decode_responses=True)
        from opennpc.backends.redis_memory import RedisMemoryStore

        store = RedisMemoryStore(client=client, max_events=5)
        for i in range(10):
            store.add(_event("a", f"ev {i}"))

        assert len(store.recent("a", limit=100)) <= 5

    def test_ping_with_fakeredis(self):
        store = _make_store()
        assert store.ping() is True

    def test_agent_count(self):
        store = _make_store()
        store.add(_event("p1", "e"))
        store.add(_event("p2", "e"))
        store.add(_event("p1", "e2"))
        assert store.agent_count() == 2

    def test_separate_agents(self):
        """Events must be isolated per agent."""
        store = _make_store()
        store.add(_event("guard", "raised alarm"))
        store.add(_event("merchant", "sold item"))

        guard_events = store.recent("guard")
        merchant_events = store.recent("merchant")
        assert all(e.agent_id == "guard" for e in guard_events)
        assert all(e.agent_id == "merchant" for e in merchant_events)
