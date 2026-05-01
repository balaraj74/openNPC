"""Tests for the decision cache."""

from __future__ import annotations

import time

import pytest

from opennpc.cache import DecisionCache, _quantize, _state_fingerprint
from opennpc.types import ActionDecision, AgentConfig, AgentType, DecisionTrace, GameState, Goal, Personality


def _config(agent_id: str = "npc_01") -> AgentConfig:
    return AgentConfig(
        agent_id=agent_id,
        agent_type=AgentType.ENEMY,
        personality=Personality(aggression=0.7),
        goals=[Goal("attack_target", 0.8)],
        allowed_actions=["attack", "defend", "flee"],
    )


def _state(agent_id: str = "npc_01", health: int = 80, distance: float = 3.0) -> GameState:
    return GameState(
        agent_id=agent_id,
        health=health,
        threat_level=0.5,
        distance_to_target=distance,
        target_health=60,
    )


def _decision(agent_id: str = "npc_01") -> ActionDecision:
    return ActionDecision(
        agent_id=agent_id,
        action="attack",
        confidence=0.9,
        reason="Test decision",
        trace=DecisionTrace(
            policy_name="heuristic",
            policy_action="attack",
            final_action="attack",
            confidence=0.9,
            reason="Test",
            fallback_used=False,
            valid_actions=["attack", "defend"],
            goal_scores={},
            personality_influence={},
            memory_summary="",
            state_snapshot={},
        ),
    )


class TestQuantize:
    def test_basic_quantize(self) -> None:
        assert _quantize(0.73, 0.05) == pytest.approx(0.75)
        assert _quantize(0.72, 0.05) == pytest.approx(0.7)
        assert _quantize(0.0, 0.05) == pytest.approx(0.0)

    def test_zero_tolerance(self) -> None:
        assert _quantize(0.73, 0.0) == 0.73


class TestStateFingerprint:
    def test_same_state_same_fingerprint(self) -> None:
        s1 = _state(health=80, distance=3.0)
        s2 = _state(health=80, distance=3.0)
        assert _state_fingerprint(s1, 0.05) == _state_fingerprint(s2, 0.05)

    def test_different_state_different_fingerprint(self) -> None:
        s1 = _state(health=80, distance=3.0)
        s2 = _state(health=40, distance=3.0)
        assert _state_fingerprint(s1, 0.05) != _state_fingerprint(s2, 0.05)

    def test_within_tolerance_same_fingerprint(self) -> None:
        s1 = _state(health=80, distance=3.0)
        s2 = _state(health=80, distance=3.04)
        assert _state_fingerprint(s1, 0.1) == _state_fingerprint(s2, 0.1)


class TestDecisionCache:
    def test_miss_on_empty_cache(self) -> None:
        cache = DecisionCache(ttl_seconds=1.0)
        result = cache.get(_config(), _state())
        assert result is None

    def test_hit_after_put(self) -> None:
        cache = DecisionCache(ttl_seconds=1.0)
        config = _config()
        state = _state()
        decision = _decision()

        cache.put(config, state, decision)
        result = cache.get(config, state)
        assert result is not None
        assert result.action == "attack"
        assert "Cached decision" in result.reason

    def test_miss_after_ttl_expires(self) -> None:
        cache = DecisionCache(ttl_seconds=0.05)
        config = _config()
        state = _state()
        decision = _decision()

        cache.put(config, state, decision)
        time.sleep(0.08)
        result = cache.get(config, state)
        assert result is None

    def test_miss_on_state_change(self) -> None:
        cache = DecisionCache(ttl_seconds=1.0, tolerance=0.05)
        config = _config()
        decision = _decision()

        cache.put(config, _state(health=80), decision)
        result = cache.get(config, _state(health=30))
        assert result is None

    def test_invalidate(self) -> None:
        cache = DecisionCache(ttl_seconds=1.0)
        config = _config()
        state = _state()
        decision = _decision()

        cache.put(config, state, decision)
        cache.invalidate("npc_01")
        assert cache.get(config, state) is None

    def test_clear(self) -> None:
        cache = DecisionCache(ttl_seconds=1.0)
        for i in range(5):
            config = _config(f"npc_{i}")
            cache.put(config, _state(f"npc_{i}"), _decision(f"npc_{i}"))
        cache.clear()
        assert cache.stats()["entries"] == 0

    def test_stats(self) -> None:
        cache = DecisionCache(ttl_seconds=1.0)
        config = _config()
        state = _state()
        decision = _decision()

        cache.get(config, state)
        cache.put(config, state, decision)
        cache.get(config, state)
        cache.get(config, state)

        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] > 0.5

    def test_eviction_on_max_entries(self) -> None:
        cache = DecisionCache(ttl_seconds=1.0, max_entries=3)
        for i in range(5):
            config = _config(f"npc_{i}")
            cache.put(config, _state(f"npc_{i}"), _decision(f"npc_{i}"))
        assert cache.stats()["entries"] == 3

    def test_confidence_decay_on_cache_hit(self) -> None:
        cache = DecisionCache(ttl_seconds=1.0)
        config = _config()
        state = _state()
        decision = _decision()

        cache.put(config, state, decision)
        cached = cache.get(config, state)
        assert cached is not None
        assert cached.confidence < decision.confidence
