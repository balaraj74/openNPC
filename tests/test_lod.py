"""Tests for the AI Level-of-Detail engine."""

from opennpc.lod import LODConfig, LODEngine, LODProfile, LODTier
from opennpc.types import AgentConfig, Personality


def test_close_agent_gets_full_tier() -> None:
    engine = LODEngine()
    engine.register("npc_01", distance_to_camera=5.0, is_visible=True)
    tier = engine.update("npc_01", distance_to_camera=5.0)
    assert tier == LODTier.FULL


def test_far_agent_gets_dormant_tier() -> None:
    engine = LODEngine()
    engine.register("npc_02", distance_to_camera=100.0, is_visible=False)
    tier = engine.update("npc_02", distance_to_camera=100.0, is_visible=False, narrative_importance=0.1)
    assert tier == LODTier.DORMANT


def test_combat_agent_always_full() -> None:
    engine = LODEngine()
    engine.register("npc_03", distance_to_camera=80.0, in_combat=True)
    tier = engine.update("npc_03", distance_to_camera=80.0, in_combat=True)
    assert tier == LODTier.FULL


def test_narrative_importance_boosts_tier() -> None:
    engine = LODEngine(config=LODConfig(full_distance=10.0, reduced_distance=30.0))
    # distance=14, importance=0.9 → effective = 14 - (0.9 * 0.5 * 10) = 14 - 4.5 = 9.5 → FULL
    engine.register("npc_04", distance_to_camera=14.0, narrative_importance=0.9)
    tier = engine.update("npc_04", distance_to_camera=14.0, narrative_importance=0.9)
    assert tier == LODTier.FULL


def test_evaluate_all_returns_dict() -> None:
    engine = LODEngine()
    engine.register("a", distance_to_camera=5.0)
    engine.register("b", distance_to_camera=50.0)
    result = engine.evaluate_all()
    assert len(result) == 2
    assert "a" in result
    assert "b" in result


def test_apply_to_config_overrides_policy() -> None:
    engine = LODEngine()
    engine.register("npc_05", distance_to_camera=40.0, is_visible=True, narrative_importance=0.2)
    engine.update("npc_05", distance_to_camera=40.0)
    config = AgentConfig(agent_id="npc_05", rl_policy="heuristic")
    modified = engine.apply_to_config("npc_05", config)
    profile = engine.profile_for("npc_05")
    if profile.policy_override:
        assert modified.rl_policy == profile.policy_override


def test_summary_counts_correct() -> None:
    engine = LODEngine()
    engine.register("a", distance_to_camera=5.0)
    engine.register("b", distance_to_camera=5.0)
    engine.register("c", distance_to_camera=100.0, is_visible=False, narrative_importance=0.1)
    engine.evaluate_all()
    summary = engine.summary()
    assert summary["total_agents"] == 3
    assert sum(summary["distribution"].values()) == 3
