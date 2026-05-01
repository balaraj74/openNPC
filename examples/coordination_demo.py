"""Multi-agent coordination demo.

Run with:
    python examples/coordination_demo.py
"""

from __future__ import annotations

from opennpc import AgentConfig, GameState, Goal, MultiAgentCoordinator, Personality


def make_enemy(agent_id: str, aggression: float, distance: float) -> tuple[AgentConfig, GameState]:
    config = AgentConfig(
        agent_id=agent_id,
        personality=Personality(aggression=aggression, caution=0.25, risk_tolerance=0.8),
        goals=[Goal("attack_target", 1.0)],
        allowed_actions=["idle", "move", "attack", "defend", "flank", "set_trap"],
        seed=11,
    )
    state = GameState(
        agent_id=agent_id,
        health=85,
        threat_level=0.25,
        nearby_entities=["player"],
        target_health=28,
        distance_to_target=distance,
        tick=3,
    )
    return config, state


def main() -> None:
    squad = [
        make_enemy("raider_01", aggression=0.95, distance=1.0),
        make_enemy("raider_02", aggression=0.8, distance=1.2),
        make_enemy("raider_03", aggression=0.7, distance=1.4),
    ]
    configs = [item[0] for item in squad]
    states = [item[1] for item in squad]

    coordinator = MultiAgentCoordinator(max_attackers_per_target=1)
    plan = coordinator.coordinate(configs, states)

    print("Coordinated squad plan:")
    for agent_id, decision in plan.decisions.items():
        assignment = plan.assignments[agent_id]
        print(f"- {agent_id}: {decision.action:9s} role={assignment.role:11s} reason={decision.reason}")
    print(f"\nShared context: {plan.shared_context}")


if __name__ == "__main__":
    main()
