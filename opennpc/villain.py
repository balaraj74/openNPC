"""Villain multi-step planner.

Provides a planning layer for villain agents that considers long-term strategy,
territory control, trap placement, and player pattern adaptation. This operates
on top of the decision engine — it does not replace it. Instead, it adjusts
goal priorities and injects strategic memory events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from opennpc.strategy import PlayerPatternTracker
from opennpc.types import AgentConfig, GameState, Goal


@dataclass(slots=True)
class VillainPlan:
    """A structured villain plan output."""

    next_action: str
    long_term_strategy: str
    predicted_player_response: str
    adaptation_note: str
    adjusted_goals: list[Goal]
    confidence: float


class VillainPlanner:
    """Generates multi-step strategic plans for villain-type agents.

    The planner uses the player pattern tracker to adjust villain goals and
    recommend tactical adaptations. It is designed to be called periodically
    (every N ticks, not every frame) per the project prompt guidelines.
    """

    def __init__(self, pattern_tracker: PlayerPatternTracker | None = None) -> None:
        self.tracker = pattern_tracker or PlayerPatternTracker()
        self._plan_count: int = 0

    def plan(self, config: AgentConfig, state: GameState) -> VillainPlan:
        """Generate a villain strategy plan based on current state and patterns."""
        self._plan_count += 1
        agg = self.tracker.aggression_estimate()
        pred = self.tracker.predictability()
        app = self.tracker.approach_tendency()
        health_ratio = max(0.0, min(1.0, state.health / 100.0))
        target_health = state.target_health if state.target_health is not None else 100.0
        distance = state.distance_to_target if state.distance_to_target is not None else 5.0

        # Determine long-term strategy
        if health_ratio < 0.3:
            strategy = "retreat_and_rebuild"
            next_action = "flee" if "flee" in config.allowed_actions else "hide"
            predicted = "Player will likely advance to finish the villain."
            adaptation = "Set traps along retreat path and prepare ambush at fallback position."
        elif pred > 0.55 and agg > 0.5:
            strategy = "exploit_predictability"
            next_action = "set_trap" if "set_trap" in config.allowed_actions else "hide"
            predicted = f"Player will likely repeat {self.tracker.dominant_action()} pattern."
            adaptation = "Place traps where the player's pattern leads them."
        elif agg > 0.6:
            strategy = "defensive_attrition"
            next_action = "defend" if "defend" in config.allowed_actions else "seek_cover"
            predicted = "Aggressive player will overextend. Wait for opening."
            adaptation = "Let the player exhaust themselves, then counter-attack."
        elif app > 0.6 and distance <= 3:
            strategy = "flanking_ambush"
            next_action = "flank" if "flank" in config.allowed_actions else "hide"
            predicted = "Player charges in. Flank to gain positional advantage."
            adaptation = "Use terrain and traps to control engagement distance."
        elif target_health < 35:
            strategy = "press_advantage"
            next_action = "attack" if "attack" in config.allowed_actions else "flank"
            predicted = "Weakened player will try to retreat or heal."
            adaptation = "Maintain pressure. Don't let them recover."
        else:
            strategy = "control_and_observe"
            next_action = "patrol" if "patrol" in config.allowed_actions else "move"
            predicted = "Player is cautious. Maintain territory control."
            adaptation = "Expand trapped area and wait for the player to commit."

        # Adjust goals based on strategy
        adjusted = self._adjust_goals(config.goals, strategy, agg, health_ratio)

        confidence = 0.5 + pred * 0.2 + (0.15 if health_ratio > 0.5 else -0.1)
        confidence = max(0.1, min(0.95, confidence))

        return VillainPlan(
            next_action=next_action,
            long_term_strategy=strategy,
            predicted_player_response=predicted,
            adaptation_note=adaptation,
            adjusted_goals=adjusted,
            confidence=round(confidence, 3),
        )

    def _adjust_goals(
        self,
        goals: list[Goal],
        strategy: str,
        player_aggression: float,
        health_ratio: float,
    ) -> list[Goal]:
        """Return adjusted goal list based on the current strategy."""
        adjusted: list[Goal] = []
        for goal in goals:
            new_priority = goal.priority
            if strategy == "retreat_and_rebuild" and goal.name == "survive":
                new_priority = min(1.0, goal.priority + 0.3)
            elif strategy == "exploit_predictability" and goal.name in {"trigger_strategic_traps", "ambush"}:
                new_priority = min(1.0, goal.priority + 0.25)
            elif strategy == "defensive_attrition" and goal.name in {"control_area", "defend"}:
                new_priority = min(1.0, goal.priority + 0.2)
            elif strategy == "press_advantage" and goal.name in {"attack_target", "weaken_player"}:
                new_priority = min(1.0, goal.priority + 0.3)

            if health_ratio < 0.3 and goal.name in {"attack_target", "weaken_player"}:
                new_priority = max(0.0, new_priority - 0.25)
            if player_aggression > 0.7 and goal.name == "survive":
                new_priority = min(1.0, new_priority + 0.15)

            adjusted.append(Goal(name=goal.name, priority=round(new_priority, 3), active=goal.active))
        return adjusted

    @property
    def plans_generated(self) -> int:
        return self._plan_count

    def summary(self) -> str:
        """Short text summary for logging or debug endpoints."""
        return (
            f"VillainPlanner: {self._plan_count} plans generated. "
            f"Pattern tracker has {self.tracker.total_observations} observations."
        )
