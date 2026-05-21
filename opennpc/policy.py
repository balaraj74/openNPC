"""Policies used by the OpenNPC decision engine."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from opennpc.types import AgentConfig, GameState


@dataclass(slots=True)
class PolicyDecision:
    action: str
    confidence: float
    reason: str
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class Policy(Protocol):
    name: str

    def select_action(
        self,
        config: AgentConfig,
        state: GameState,
        valid_actions: list[str],
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        ...


class HeuristicPolicy:
    """Fast local policy for runtime decisions and deterministic fallback."""

    name = "heuristic"

    def select_action(
        self,
        config: AgentConfig,
        state: GameState,
        valid_actions: list[str],
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        context = context or {}
        if not valid_actions:
            return PolicyDecision("idle", 1.0, "No valid actions were available.", {"idle": 1.0})

        scores = {action: 0.01 for action in valid_actions}
        for action, score in context.get("goal_scores", {}).items():
            if action in scores:
                scores[action] += float(score)

        personality_scores = self._personality_scores(config, valid_actions)
        for action, score in personality_scores.items():
            scores[action] += score

        self._apply_state_scores(scores, state)
        self._apply_memory_scores(scores, context.get("memory_summary", ""))

        if state.last_action in scores:
            repeat_penalty = 0.12 + config.personality.patience * 0.08
            scores[state.last_action] -= repeat_penalty

        if "idle" in scores and max(scores.values()) < 0.2:
            scores["idle"] += 0.1

        seed = config.seed if config.seed is not None else self._stable_seed(config.agent_id, state.tick)
        rng = random.Random(seed + int(state.tick))
        for action in scores:
            scores[action] += rng.uniform(0.0, 0.015)

        best_action = max(scores, key=scores.get)
        ordered = sorted(scores.values(), reverse=True)
        margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
        confidence = max(0.05, min(0.98, 0.45 + margin))
        reason = self._reason(best_action, state, context.get("active_goals", []))
        return PolicyDecision(best_action, confidence, reason, scores, {"personality_scores": personality_scores})

    def _personality_scores(self, config: AgentConfig, valid_actions: list[str]) -> dict[str, float]:
        p = config.personality
        candidates = {action: 0.0 for action in valid_actions}
        boosts = {
            "attack": p.aggression * 0.45 + p.risk_tolerance * 0.2,
            "flank": p.aggression * 0.25 + p.curiosity * 0.15 + p.risk_tolerance * 0.15,
            "set_trap": p.patience * 0.25 + p.caution * 0.15,
            "seek_cover": p.caution * 0.45 + (1.0 - p.risk_tolerance) * 0.2,
            "defend": p.caution * 0.35 + p.patience * 0.2,
            "flee": p.caution * 0.25 + (1.0 - p.loyalty) * 0.12,
            "talk": p.sociability * 0.45,
            "trade": p.sociability * 0.25 + p.patience * 0.15,
            "patrol": p.curiosity * 0.25 + p.patience * 0.12,
            "move": p.curiosity * 0.2,
            "follow": p.loyalty * 0.35 + p.sociability * 0.12,
            "hide": p.caution * 0.25 + p.patience * 0.15,
        }
        for action, score in boosts.items():
            if action in candidates:
                candidates[action] += score
        return candidates

    def _apply_state_scores(self, scores: dict[str, float], state: GameState) -> None:
        max_health = float(state.value("max_health", 20.0))
        health_ratio = max(0.0, min(1.0, state.health / max_health))
        distance = state.distance_to_target if state.distance_to_target is not None else 99.0
        target_health = state.target_health if state.target_health is not None else 20.0
        target_max_health = float(state.value("target_max_health", 20.0))
        target_health_ratio = max(0.0, min(1.0, target_health / target_max_health))
        is_enemy = state.agent_type in ("enemy", "villain") or state.agent_type.value in ("enemy", "villain")
        has_target = distance < 90 or "player" in state.nearby_entities

        # --- Enemy-type mob scoring (zombies, skeletons, etc.) ---
        if is_enemy and has_target:
            # Enemies are inherently aggressive toward players
            self._add(scores, "attack", 0.55)
            self._add(scores, "move", 0.25)

            if distance <= 2.5:
                # In melee range — attack is overwhelmingly favored
                self._add(scores, "attack", 0.6)
                self._add(scores, "defend", 0.05)
            elif distance <= 6.0:
                # Close enough to chase — attack + flank
                self._add(scores, "attack", 0.35)
                self._add(scores, "flank", 0.2)
                self._add(scores, "move", 0.15)
            elif distance <= 16.0:
                # Medium range — pursue
                self._add(scores, "move", 0.4)
                self._add(scores, "attack", 0.15)
                self._add(scores, "flank", 0.1)
            else:
                # Far away — patrol or slowly approach
                self._add(scores, "patrol", 0.2)
                self._add(scores, "move", 0.3)
                self._add(scores, "attack", -0.3)

            # Only flee when genuinely near death (< 20% HP)
            if health_ratio <= 0.2:
                self._add(scores, "flee", 0.5)
                self._add(scores, "attack", -0.15)
            elif health_ratio <= 0.4:
                # Wounded but still fight — slightly more defensive
                self._add(scores, "defend", 0.15)

            # Smell blood — target is weak, go for the kill
            if target_health_ratio <= 0.3:
                self._add(scores, "attack", 0.5)
                self._add(scores, "flank", 0.2)
            elif target_health_ratio <= 0.5:
                self._add(scores, "attack", 0.25)

            if "player" in state.nearby_entities:
                self._add(scores, "attack", 0.15)

            return  # Enemy scoring is complete

        # --- Non-enemy / no-target scoring (civilians, companions) ---
        if state.threat_level >= 0.65:
            self._add(scores, "defend", 0.35)
            self._add(scores, "seek_cover", 0.35 if state.cover_available else 0.0)
            self._add(scores, "flee", 0.2)
        if health_ratio <= 0.35:
            self._add(scores, "flee", 0.65)
            self._add(scores, "seek_cover", 0.45 if state.cover_available else 0.0)
            self._add(scores, "attack", -0.3)
        if distance <= 1.5:
            self._add(scores, "attack", 0.45)
            self._add(scores, "defend", 0.1)
        elif distance <= 4.0:
            self._add(scores, "flank", 0.2)
            self._add(scores, "move", 0.2)
        else:
            self._add(scores, "move", 0.35)
            self._add(scores, "patrol", 0.1)
            self._add(scores, "attack", -0.4)
        if target_health_ratio <= 0.3:
            self._add(scores, "attack", 0.35)
            self._add(scores, "flank", 0.15)
        if "player" in state.nearby_entities:
            self._add(scores, "talk", 0.1)
            self._add(scores, "attack", 0.1)
        if state.recent_rewards and sum(state.recent_rewards[-3:]) < 0:
            self._add(scores, "set_trap", 0.25)
            self._add(scores, "flank", 0.15)

    def _apply_memory_scores(self, scores: dict[str, float], memory_summary: str) -> None:
        memory = memory_summary.lower()
        if "rushed" in memory or "aggressive" in memory:
            self._add(scores, "set_trap", 0.35)
            self._add(scores, "seek_cover", 0.15)
        if "failed attack" in memory or "attack failed" in memory:
            self._add(scores, "attack", -0.8)
            self._add(scores, "flank", 0.45)
            self._add(scores, "set_trap", 0.4)
        if "route is dangerous" in memory:
            self._add(scores, "patrol", -0.2)
            self._add(scores, "seek_cover", 0.2)
        if "player is hostile" in memory:
            self._add(scores, "defend", 0.2)
            self._add(scores, "attack", 0.15)

    def _add(self, scores: dict[str, float], action: str, amount: float) -> None:
        if action in scores:
            scores[action] += amount

    def _reason(self, action: str, state: GameState, active_goals: list[str]) -> str:
        goal_text = f" while pursuing {', '.join(active_goals)}" if active_goals else ""
        if action in {"flee", "seek_cover", "defend"}:
            return f"Selected {action} because health/threat conditions favor survival{goal_text}."
        if action in {"attack", "flank", "set_trap"}:
            return f"Selected {action} because combat goals and target context favor pressure{goal_text}."
        if action in {"talk", "trade", "follow"}:
            return f"Selected {action} because social or companion goals are currently useful{goal_text}."
        return f"Selected {action} as the highest-scoring valid action{goal_text}."

    def _stable_seed(self, agent_id: str, tick: int) -> int:
        return sum((idx + 1) * ord(char) for idx, char in enumerate(agent_id)) + tick * 997


class ScriptedCombatPolicy:
    """Simple baseline enemy behavior for simulation comparisons."""

    name = "scripted_combat"

    def select_action(
        self,
        config: AgentConfig,
        state: GameState,
        valid_actions: list[str],
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        if state.health <= 30 and "flee" in valid_actions:
            return PolicyDecision("flee", 0.85, "Scripted baseline flees at low health.")
        if state.cover_available and state.threat_level > 0.6 and "seek_cover" in valid_actions:
            return PolicyDecision("seek_cover", 0.8, "Scripted baseline seeks cover under high threat.")
        if (state.distance_to_target or 99) <= 1.5 and "attack" in valid_actions:
            return PolicyDecision("attack", 0.8, "Scripted baseline attacks in range.")
        if "move" in valid_actions:
            return PolicyDecision("move", 0.65, "Scripted baseline moves toward the target.")
        return PolicyDecision(valid_actions[0] if valid_actions else "idle", 0.4, "Scripted baseline fallback.")


class RandomPolicy:
    name = "random"

    def select_action(
        self,
        config: AgentConfig,
        state: GameState,
        valid_actions: list[str],
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        if not valid_actions:
            return PolicyDecision("idle", 1.0, "No valid actions were available.")
        seed = config.seed if config.seed is not None else sum(ord(char) for char in config.agent_id)
        rng = random.Random(seed + state.tick)
        action = rng.choice(valid_actions)
        return PolicyDecision(action, 0.25, "Random policy sampled a valid action.", {action: 1.0})
