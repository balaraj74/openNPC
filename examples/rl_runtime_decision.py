"""Load a trained PPO checkpoint into the runtime DecisionEngine.

Run after training:
    python examples/train_enemy.py
    python examples/rl_runtime_decision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opennpc import AgentConfig, DecisionEngine, GameState, Goal, Personality
from opennpc.simulation.environment import COMBAT_ACTIONS


def main() -> None:
    model_path = Path("artifacts/ppo_enemy.pt")
    if not model_path.exists():
        print("No PPO checkpoint found. Run: python examples/train_enemy.py")
        return

    engine = DecisionEngine()
    config = AgentConfig(
        agent_id="enemy_01",
        rl_policy="ppo",
        metadata={"model_path": str(model_path)},
        personality=Personality(aggression=0.75, caution=0.45, risk_tolerance=0.65),
        goals=[Goal("attack_target", 0.8), Goal("survive", 0.6)],
        allowed_actions=list(COMBAT_ACTIONS),
    )
    state = GameState(
        agent_id="enemy_01",
        health=88,
        threat_level=0.25,
        nearby_entities=["player"],
        target_health=32,
        distance_to_target=1.0,
        tick=5,
    )

    decision = engine.decide(config, state)
    policy_name = decision.trace.policy_name if decision.trace else "unknown"
    fallback = decision.trace.fallback_used if decision.trace else False
    print(f"policy={policy_name} action={decision.action} confidence={decision.confidence:.2f} fallback={fallback}")
    print(decision.reason)


if __name__ == "__main__":
    main()
