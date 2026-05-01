#!/usr/bin/env python3
"""Demo: LOD engine scaling with many agents.

Shows how the LOD system assigns tiers to agents at different distances and
adjusts their compute budget. Simulates 50 agents spread across a scene.

Usage:
    python examples/lod_demo.py
"""

import random

from opennpc.lod import LODEngine, LODTier
from opennpc.types import AgentConfig, Goal


def main() -> None:
    rng = random.Random(42)
    engine = LODEngine()

    print("=" * 70)
    print("  OpenNPC — AI Level-of-Detail Demo")
    print("=" * 70)

    # Register 50 agents at various distances
    agents = []
    for i in range(50):
        agent_id = f"npc_{i:03d}"
        dist = rng.uniform(2.0, 120.0)
        visible = rng.random() > 0.2
        importance = rng.uniform(0.0, 1.0) if i < 5 else rng.uniform(0.0, 0.4)
        in_combat = rng.random() < 0.1
        engine.register(agent_id, distance_to_camera=dist, is_visible=visible,
                        narrative_importance=importance, in_combat=in_combat)
        agents.append((agent_id, dist, visible, importance, in_combat))

    tiers = engine.evaluate_all()
    summary = engine.summary()

    print(f"\n  Total agents: {summary['total_agents']}")
    print(f"  Distribution: {summary['distribution']}")
    print()

    # Show first 15 agents with their assignments
    print(f"  {'Agent':<12} {'Distance':>8} {'Visible':>8} {'Import':>7} {'Combat':>7} {'Tier':>10}")
    print("  " + "-" * 58)
    for agent_id, dist, vis, imp, combat in agents[:15]:
        tier = tiers[agent_id]
        profile = engine.profile_for(agent_id)
        print(
            f"  {agent_id:<12} {dist:8.1f} {'yes' if vis else 'no':>8} "
            f"{imp:7.2f} {'yes' if combat else 'no':>7} {tier.name:>10}"
        )

    print()

    # Demonstrate config modification
    config = AgentConfig(
        agent_id="npc_000",
        goals=[Goal("survive", 1.0)],
        allowed_actions=["idle", "move", "defend"],
        rl_policy="heuristic",
        decision_interval_ms=250,
    )
    modified = engine.apply_to_config("npc_000", config)
    tier = engine.tier_for("npc_000")
    print(f"  Config modification example for npc_000 (tier={tier.name}):")
    print(f"    Original policy:   {config.rl_policy} | interval: {config.decision_interval_ms}ms")
    print(f"    Modified policy:   {modified.rl_policy} | interval: {modified.decision_interval_ms}ms")
    print(f"    Memory enabled:    {modified.memory_enabled}")
    print(f"    LOD level:         {modified.lod_level}")
    print("=" * 70)


if __name__ == "__main__":
    main()
