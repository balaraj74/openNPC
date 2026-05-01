from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opennpc import AgentConfig, DecisionEngine, Goal, Personality
from opennpc.simulation.environment import COMBAT_ACTIONS, GridCombatEnv


def main() -> None:
    env = GridCombatEnv(seed=5)
    state = env.reset()
    engine = DecisionEngine()
    config = AgentConfig(
        agent_id="enemy_01",
        personality=Personality(aggression=0.75, caution=0.45, risk_tolerance=0.65),
        goals=[Goal("attack_target", 0.8), Goal("survive", 0.65), Goal("weaken_player", 0.4)],
        allowed_actions=list(COMBAT_ACTIONS),
    )
    total = 0.0
    for _ in range(30):
        decision = engine.decide(config, state)
        result = env.step(decision.action)
        total += result.reward
        print(
            f"{env.render_text()} action={decision.action:<10} reward={result.reward:>5.2f} "
            f"enemy={env.enemy_health:>5.1f} player={env.player_health:>5.1f}"
        )
        state = result.state
        if result.done:
            break
    print(f"total_reward={total:.2f}")


if __name__ == "__main__":
    main()
