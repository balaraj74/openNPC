#!/usr/bin/env python3
"""Demo: Villain vs Player — adaptive enemy intelligence showcase.

Runs a 30-tick combat loop where a villain agent adapts its strategy based
on observed player behavior. Demonstrates the full pipeline:
  PlayerPatternTracker → VillainPlanner → DecisionEngine → Action

Usage:
    python examples/villain_demo.py
"""

from opennpc.decision import DecisionEngine
from opennpc.strategy import PlayerPatternTracker
from opennpc.types import AgentConfig, AgentType, GameState, Goal, Personality
from opennpc.villain import VillainPlanner

PLAYER_PATTERN = [
    "attack", "attack", "attack", "move", "attack",
    "flank", "attack", "attack", "move", "attack",
    "attack", "charge", "attack", "attack", "move",
    "attack", "attack", "flank", "attack", "attack",
    "move", "attack", "attack", "attack", "attack",
    "charge", "attack", "attack", "move", "attack",
]


def main() -> None:
    tracker = PlayerPatternTracker(window_size=15)
    planner = VillainPlanner(pattern_tracker=tracker)
    engine = DecisionEngine()

    config = AgentConfig(
        agent_id="boss_01",
        agent_type=AgentType.VILLAIN,
        personality=Personality(aggression=0.65, caution=0.7, patience=0.85, risk_tolerance=0.45),
        goals=[
            Goal("weaken_player", 0.9),
            Goal("control_area", 0.75),
            Goal("trigger_strategic_traps", 0.7),
            Goal("survive", 0.55),
        ],
        allowed_actions=["idle", "move", "attack", "defend", "flee", "seek_cover", "flank", "set_trap", "patrol", "hide"],
        rl_policy="heuristic",
        seed=42,
    )

    villain_hp = 100.0
    player_hp = 100.0
    distance = 5.0

    print("=" * 70)
    print("  OpenNPC — Villain Adaptive Intelligence Demo")
    print("=" * 70)

    for tick in range(min(30, len(PLAYER_PATTERN))):
        player_action = PLAYER_PATTERN[tick]
        tracker.record(player_action)

        state = GameState(
            agent_id="boss_01",
            agent_type=AgentType.VILLAIN,
            health=villain_hp,
            target_health=player_hp,
            distance_to_target=distance,
            threat_level=min(1.0, tracker.aggression_estimate()),
            nearby_entities=["player"],
            cover_available=True,
            tick=tick,
        )

        # Generate strategic plan every 5 ticks
        if tick % 5 == 0:
            plan = planner.plan(config, state)
            print(f"\n  [STRATEGY tick={tick}] {plan.long_term_strategy}")
            print(f"    → Predicted player response: {plan.predicted_player_response}")
            print(f"    → Adaptation: {plan.adaptation_note}")
            print(f"    → Confidence: {plan.confidence:.2f}")

        decision = engine.decide(config, state)

        # Simulate simple combat effects
        if decision.action == "attack" and distance <= 2.0:
            player_hp = max(0, player_hp - 12)
        if player_action == "attack" and distance <= 2.0:
            villain_hp = max(0, villain_hp - 8)
        if decision.action == "move":
            distance = max(1.0, distance - 1.5)
        if decision.action == "flee":
            distance = min(10.0, distance + 2.0)
        if player_action in {"move", "charge"}:
            distance = max(1.0, distance - 1.0)

        print(
            f"  [tick={tick:2d}] Player: {player_action:8s} → "
            f"Villain: {decision.action:10s} (conf={decision.confidence:.2f}) "
            f"| HP: V={villain_hp:.0f} P={player_hp:.0f} | dist={distance:.1f}"
        )

        if villain_hp <= 0 or player_hp <= 0:
            break

    print("\n" + "=" * 70)
    print("  Pattern Analysis Summary:")
    print(f"    {tracker.summary()}")
    print(f"    Counter-recommendation: {tracker.counter_recommendation()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
