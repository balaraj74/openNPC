"""Baseline agents for comparing OpenNPC policies."""

from __future__ import annotations

from opennpc.simulation.environment import COMBAT_ACTIONS
from opennpc.types import AgentConfig, GameState


class ScriptedCombatAgent:
    def __init__(self, agent_id: str = "enemy_01") -> None:
        self.config = AgentConfig(
            agent_id=agent_id,
            rl_policy="scripted_combat",
            allowed_actions=list(COMBAT_ACTIONS),
            memory_enabled=False,
        )

    def act(self, state: GameState) -> str:
        distance = state.distance_to_target if state.distance_to_target is not None else 99
        if state.health <= 30:
            return "flee"
        if state.cover_available and state.threat_level > 0.6:
            return "seek_cover"
        if distance <= 1.5:
            return "attack"
        return "move"
