"""Goal scoring for OpenNPC decisions."""

from __future__ import annotations

from dataclasses import dataclass

from opennpc.types import AgentConfig, GameState


GOAL_ACTION_WEIGHTS: dict[str, dict[str, float]] = {
    "survive": {"flee": 0.95, "seek_cover": 0.85, "defend": 0.75, "hide": 0.55},
    "attack": {"attack": 0.9, "flank": 0.55, "move": 0.25},
    "attack_target": {"attack": 1.0, "flank": 0.7, "move": 0.3, "set_trap": 0.2},
    "defend": {"defend": 0.9, "patrol": 0.35, "seek_cover": 0.55},
    "ambush": {"hide": 0.75, "flank": 0.7, "set_trap": 0.9, "attack": 0.35},
    "weaken_player": {"attack": 0.65, "set_trap": 0.9, "flank": 0.75},
    "control_area": {"patrol": 0.8, "set_trap": 0.7, "defend": 0.6},
    "trigger_strategic_traps": {"set_trap": 1.0, "hide": 0.45, "patrol": 0.25},
    "work": {"move": 0.5, "idle": 0.3, "trade": 0.5},
    "trade": {"trade": 1.0, "talk": 0.55, "move": 0.2},
    "explore": {"move": 0.65, "patrol": 0.55, "talk": 0.25},
    "interact": {"talk": 0.85, "trade": 0.45, "follow": 0.25},
    "protect": {"follow": 0.65, "defend": 0.75, "attack": 0.35},
    "assist_player": {"follow": 0.75, "defend": 0.55, "use_item": 0.4},
}


@dataclass(slots=True)
class GoalEvaluation:
    action_scores: dict[str, float]
    active_goals: list[str]


class GoalManager:
    def evaluate(self, config: AgentConfig, state: GameState, memory_summary: str = "") -> GoalEvaluation:
        scores = {action: 0.0 for action in config.allowed_actions}
        active_goals: list[str] = []
        for goal in config.goals:
            if not goal.active:
                continue
            active_goals.append(goal.name)
            urgency = self._urgency(goal.name, goal.priority, state, memory_summary)
            for action, weight in GOAL_ACTION_WEIGHTS.get(goal.name, {}).items():
                if action in scores:
                    scores[action] += urgency * weight
        if not active_goals and state.current_goal:
            active_goals.append(state.current_goal)
            for action, weight in GOAL_ACTION_WEIGHTS.get(state.current_goal, {}).items():
                if action in scores:
                    scores[action] += weight
        return GoalEvaluation(action_scores=scores, active_goals=active_goals)

    def _urgency(self, goal_name: str, priority: float, state: GameState, memory_summary: str) -> float:
        priority = max(0.0, float(priority))
        health_ratio = max(0.0, min(1.0, state.health / 100.0))
        threat = state.threat_level
        distance = state.distance_to_target if state.distance_to_target is not None else 3.0
        target_health = state.target_health if state.target_health is not None else 100.0

        if goal_name == "survive":
            return priority * (0.4 + threat + (1.0 - health_ratio))
        if goal_name in {"attack", "attack_target", "weaken_player"}:
            close_bonus = 0.4 if distance <= 1.5 else 0.0
            weak_target_bonus = 0.35 if target_health <= 35 else 0.0
            return priority * (0.55 + close_bonus + weak_target_bonus)
        if goal_name in {"ambush", "trigger_strategic_traps"}:
            pattern_bonus = 0.3 if "rushed" in memory_summary or "aggressive" in memory_summary else 0.0
            return priority * (0.6 + pattern_bonus)
        if goal_name in {"defend", "control_area", "protect"}:
            return priority * (0.55 + threat * 0.5)
        return priority
