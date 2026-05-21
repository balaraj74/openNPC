"""Villain multi-step planner.

Provides a planning layer for villain agents that considers long-term strategy,
territory control, trap placement, and player pattern adaptation. This operates
on top of the decision engine — it does not replace it. Instead, it adjusts
goal priorities and injects strategic memory events.

When an LLM engine is provided, the planner enriches its heuristic plans
with natural-language strategy descriptions, predictions, and adaptations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opennpc.strategy import PlayerPatternTracker
from opennpc.types import AgentConfig, GameState, Goal

if TYPE_CHECKING:
    from opennpc.llm import LLMEngine

logger = logging.getLogger(__name__)

STRATEGY_SYSTEM_PROMPT = """You are a strategic villain AI planner in a game.
You think like a cunning boss enemy — calculating, adaptive, and ruthless.
Analyze the player's patterns and generate a tactical plan.
Be specific and actionable. Keep each field under 2 sentences.
Return ONLY valid JSON with these exact keys:
{
  "strategy": "your long-term strategy name",
  "next_move": "immediate tactical recommendation",
  "prediction": "what the player will likely do next",
  "adaptation": "how to counter their expected move",
  "taunt": "a short in-character villain line (max 15 words)"
}"""


@dataclass(slots=True)
class VillainPlan:
    """A structured villain plan output."""

    next_action: str
    long_term_strategy: str
    predicted_player_response: str
    adaptation_note: str
    adjusted_goals: list[Goal]
    confidence: float
    taunt: str = ""
    llm_enhanced: bool = False


class VillainPlanner:
    """Generates multi-step strategic plans for villain-type agents.

    The planner uses the player pattern tracker to adjust villain goals and
    recommend tactical adaptations. When an LLM is provided, it enriches
    plans with natural-language reasoning.

    Parameters
    ----------
    pattern_tracker : PlayerPatternTracker, optional
        Tracker for player behavior analysis.
    llm : LLMEngine, optional
        Optional LLM for enhanced strategic reasoning.
    """

    def __init__(
        self,
        pattern_tracker: PlayerPatternTracker | None = None,
        llm: LLMEngine | None = None,
    ) -> None:
        self.tracker = pattern_tracker or PlayerPatternTracker()
        self.llm = llm
        self._plan_count: int = 0

    def plan(self, config: AgentConfig, state: GameState) -> VillainPlan:
        """Generate a villain strategy plan based on current state and patterns."""
        self._plan_count += 1
        agg = self.tracker.aggression_estimate()
        pred = self.tracker.predictability()
        app = self.tracker.approach_tendency()
        max_health = float(state.value("max_health", 100.0)) if hasattr(state, "value") else 100.0
        health_ratio = max(0.0, min(1.0, state.health / max_health))
        target_health = state.target_health if state.target_health is not None else max_health
        distance = state.distance_to_target if state.distance_to_target is not None else 5.0

        # ── Step 1: Heuristic plan (always runs, <1ms) ──
        heuristic_plan = self._heuristic_plan(
            config, agg, pred, app, health_ratio, target_health, distance,
        )

        # ── Step 2: LLM enrichment (optional, ~100-300ms) ──
        if self.llm is not None and self.llm.is_loaded:
            try:
                enriched = self._llm_enrich(
                    config, state, heuristic_plan, agg, pred, app,
                    health_ratio, target_health, distance,
                )
                return enriched
            except Exception as e:
                logger.warning("LLM enrichment failed, using heuristic plan: %s", e)

        return heuristic_plan

    def _heuristic_plan(
        self,
        config: AgentConfig,
        agg: float,
        pred: float,
        app: float,
        health_ratio: float,
        target_health: float,
        distance: float,
    ) -> VillainPlan:
        """Fast rule-based strategic planning."""
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
            taunt="",
            llm_enhanced=False,
        )

    def _llm_enrich(
        self,
        config: AgentConfig,
        state: GameState,
        base_plan: VillainPlan,
        agg: float, pred: float, app: float,
        health_ratio: float, target_health: float, distance: float,
    ) -> VillainPlan:
        """Enrich a heuristic plan with LLM reasoning."""
        assert self.llm is not None

        prompt = (
            f"SITUATION ANALYSIS:\n"
            f"- Villain HP: {state.health:.0f}/{health_ratio * 100:.0f}%\n"
            f"- Player HP: {target_health:.0f}\n"
            f"- Distance to player: {distance:.1f} blocks\n"
            f"- Player aggression level: {agg:.1%}\n"
            f"- Player predictability: {pred:.1%}\n"
            f"- Player approach tendency: {app:.1%}\n"
            f"- Player's dominant move: {self.tracker.dominant_action()}\n"
            f"- Pattern observations: {self.tracker.total_observations}\n"
            f"- Heuristic recommends: {base_plan.long_term_strategy}\n\n"
            f"Generate a strategic plan to defeat this player."
        )

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=STRATEGY_SYSTEM_PROMPT,
            max_new_tokens=150,
            temperature=0.6,
        )

        # Try to parse LLM JSON output
        try:
            llm_plan = json.loads(response.text)
        except json.JSONDecodeError:
            # If LLM doesn't return valid JSON, use its text as enrichment
            logger.debug("LLM returned non-JSON, using as enrichment text")
            enriched = VillainPlan(
                next_action=base_plan.next_action,
                long_term_strategy=base_plan.long_term_strategy,
                predicted_player_response=base_plan.predicted_player_response,
                adaptation_note=response.text[:200],
                adjusted_goals=base_plan.adjusted_goals,
                confidence=base_plan.confidence,
                taunt="",
                llm_enhanced=True,
            )
            return enriched

        # Merge LLM output with heuristic plan
        return VillainPlan(
            next_action=base_plan.next_action,  # Keep heuristic action (validated)
            long_term_strategy=llm_plan.get("strategy", base_plan.long_term_strategy),
            predicted_player_response=llm_plan.get("prediction", base_plan.predicted_player_response),
            adaptation_note=llm_plan.get("adaptation", base_plan.adaptation_note),
            adjusted_goals=base_plan.adjusted_goals,
            confidence=min(0.95, base_plan.confidence + 0.05),
            taunt=llm_plan.get("taunt", ""),
            llm_enhanced=True,
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
        llm_status = "LLM: active" if (self.llm and self.llm.is_loaded) else "LLM: not loaded"
        return (
            f"VillainPlanner: {self._plan_count} plans generated. "
            f"Pattern tracker has {self.tracker.total_observations} observations. "
            f"{llm_status}"
        )

