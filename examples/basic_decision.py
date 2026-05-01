from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opennpc import AgentConfig, DecisionEngine, GameState, Goal, Personality


def main() -> None:
    engine = DecisionEngine()
    config = AgentConfig(
        agent_id="enemy_01",
        personality=Personality(aggression=0.85, caution=0.35, risk_tolerance=0.7),
        goals=[Goal("attack_target", 0.85), Goal("survive", 0.55)],
        allowed_actions=["idle", "move", "attack", "defend", "flee", "seek_cover", "flank", "set_trap"],
    )
    state = GameState(
        agent_id="enemy_01",
        health=72,
        threat_level=0.55,
        nearby_entities=["player"],
        target_health=34,
        distance_to_target=1.0,
        cover_available=True,
        tick=3,
    )
    decision = engine.decide(config, state, event="Player rushed aggressively twice in recent combat.")
    print(json.dumps(decision.to_dict(), indent=2))


if __name__ == "__main__":
    main()
