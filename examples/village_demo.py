#!/usr/bin/env python3
"""Demo: Village social simulation with civilian NPCs.

Runs a social simulation where a civilian NPC navigates a village, trades,
interacts with other NPCs, and responds to dynamic events. Demonstrates
non-combat agent behavior driven by the decision engine.

Usage:
    python examples/village_demo.py
"""

from opennpc.decision import DecisionEngine
from opennpc.types import AgentConfig, AgentType, GameState, Goal, Personality
from opennpc.simulation.village import VillageEnv


def main() -> None:
    engine = DecisionEngine()
    env = VillageEnv(max_steps=25, seed=42)

    config = AgentConfig(
        agent_id="civilian_01",
        agent_type=AgentType.CIVILIAN,
        personality=Personality(
            aggression=0.1, curiosity=0.7, sociability=0.85,
            caution=0.6, patience=0.7, loyalty=0.5, risk_tolerance=0.3,
        ),
        goals=[
            Goal("trade", 0.8),
            Goal("interact", 0.7),
            Goal("explore", 0.6),
            Goal("work", 0.5),
            Goal("survive", 0.4),
        ],
        constraints=["pacifist"],
        allowed_actions=["idle", "move", "talk", "trade", "follow", "flee", "hide", "patrol", "use_item"],
        rl_policy="heuristic",
        seed=42,
    )

    state = env.reset()

    print("=" * 70)
    print("  OpenNPC — Village Social Simulation Demo")
    print("=" * 70)
    print(f"  Starting gold: {env.villager.gold} | Reputation: {env.villager.reputation:.2f}")
    print()

    total_reward = 0.0
    for tick in range(25):
        decision = engine.decide(config, state, available_actions=config.allowed_actions)
        result = env.step(decision.action)
        total_reward += result.reward

        event_text = ""
        if result.info.get("danger"):
            event_text = " ⚠ DANGER!"

        print(
            f"  [tick={tick:2d}] Location: {result.info.get('location', '?'):14s} "
            f"→ {decision.action:8s} (conf={decision.confidence:.2f}) "
            f"| Gold: {env.villager.gold:3d} | Rep: {env.villager.reputation:.2f} "
            f"| Reward: {result.reward:+.2f}{event_text}"
        )

        state = result.state
        if result.done:
            print(f"\n  Episode ended at tick {tick}.")
            break

    print("\n" + "=" * 70)
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final gold: {env.villager.gold} | Final reputation: {env.villager.reputation:.2f}")
    memory_summary = engine.memory_store.summarize("civilian_01")
    if memory_summary:
        print(f"  Memory summary: {memory_summary[:200]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
