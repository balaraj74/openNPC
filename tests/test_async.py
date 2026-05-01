"""Tests for the async decision engine wrapper."""

import asyncio

from opennpc.async_engine import AsyncDecisionEngine
from opennpc.types import AgentConfig, GameState, Goal, Personality


def test_async_decide_returns_decision() -> None:
    engine = AsyncDecisionEngine(max_workers=2)
    config = AgentConfig(
        agent_id="npc_01",
        personality=Personality(aggression=0.8),
        goals=[Goal("attack_target", 1.0)],
        allowed_actions=["idle", "move", "attack", "defend"],
        seed=1,
    )
    state = GameState(
        agent_id="npc_01",
        health=90,
        threat_level=0.2,
        nearby_entities=["player"],
        target_health=50,
        distance_to_target=1.0,
    )

    decision = asyncio.run(engine.decide(config, state))
    assert decision.action in config.allowed_actions
    assert decision.agent_id == "npc_01"
    engine.shutdown()


def test_async_batch_decide() -> None:
    engine = AsyncDecisionEngine(max_workers=2)
    configs_states = [
        (
            AgentConfig(
                agent_id=f"npc_{i}",
                goals=[Goal("survive", 1.0)],
                allowed_actions=["idle", "move", "flee", "defend"],
                seed=i,
            ),
            GameState(agent_id=f"npc_{i}", health=50, threat_level=0.5),
        )
        for i in range(4)
    ]

    decisions = asyncio.run(engine.decide_batch(configs_states))
    assert len(decisions) == 4
    for decision in decisions:
        assert decision.action in {"idle", "move", "flee", "defend"}
    engine.shutdown()


def test_async_decide_scheduled() -> None:
    engine = AsyncDecisionEngine(max_workers=1)
    config = AgentConfig(
        agent_id="npc_02",
        goals=[Goal("survive", 1.0)],
        allowed_actions=["idle", "defend"],
        decision_interval_ms=50000,
        seed=5,
    )
    state = GameState(agent_id="npc_02", health=80)

    d1 = asyncio.run(engine.decide_scheduled(config, state))
    d2 = asyncio.run(engine.decide_scheduled(config, state))
    # Second call should reuse the first (interval not elapsed)
    assert d2.action == d1.action
    engine.shutdown()
