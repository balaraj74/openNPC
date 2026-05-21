"""Reward shaping strategies for runtime feedback and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from opennpc.types import AgentConfig, AgentType, GameState


@dataclass(slots=True)
class RewardBreakdown:
    """A shaped reward with named component scores."""

    total: float
    components: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "components": {key: round(value, 4) for key, value in self.components.items()},
            "reason": self.reason,
        }


class RewardStrategy(Protocol):
    """Interface for agent-type-specific reward shaping."""

    name: str

    def compute(self, config: AgentConfig, state: GameState, action: str, next_state: GameState) -> RewardBreakdown:
        ...


class EnemyRewardStrategy:
    """Combat reward shaping for normal enemies."""

    name = "enemy"

    def compute(self, config: AgentConfig, state: GameState, action: str, next_state: GameState) -> RewardBreakdown:
        target_delta = _target_health(state) - _target_health(next_state)
        self_damage = state.health - next_state.health
        distance_delta = _distance(state) - _distance(next_state)
        components = {
            "damage_dealt": target_delta / 25.0,
            "damage_taken": -self_damage / 30.0,
            "positioning": distance_delta * 0.04,
            "survival": 0.1 if next_state.health > 0 else -2.0,
        }
        if action == "attack" and target_delta <= 0:
            components["failed_attack"] = -0.12
        if action in {"seek_cover", "defend", "flee"} and next_state.threat_level < state.threat_level:
            components["threat_reduced"] = 0.2
        return _breakdown(components, "Enemy reward balances pressure, survival, and positioning.")


class VillainRewardStrategy:
    """Strategic reward shaping for boss or villain agents."""

    name = "villain"

    def compute(self, config: AgentConfig, state: GameState, action: str, next_state: GameState) -> RewardBreakdown:
        target_delta = _target_health(state) - _target_health(next_state)
        self_damage = state.health - next_state.health
        trap_delta = float(next_state.value("trap_count", 0)) - float(state.value("trap_count", 0))
        components = {
            "pressure": target_delta / 28.0,
            "survival": -self_damage / 35.0,
            "control": 0.18 if action in {"set_trap", "patrol", "defend"} else 0.0,
            "trap_setup": max(0.0, trap_delta) * 0.2,
        }
        if next_state.health < 30 and action not in {"seek_cover", "flee", "defend"}:
            components["reckless_low_health"] = -0.25
        return _breakdown(components, "Villain reward favors long-term pressure and area control.")


class CivilianRewardStrategy:
    """Social and survival reward shaping for civilian NPCs."""

    name = "civilian"

    def compute(self, config: AgentConfig, state: GameState, action: str, next_state: GameState) -> RewardBreakdown:
        reputation_delta = float(next_state.value("reputation", 0.5)) - float(state.value("reputation", 0.5))
        gold_delta = float(next_state.value("gold", 0)) - float(state.value("gold", 0))
        danger_was_active = bool(state.value("danger_active", False))
        components = {
            "reputation": reputation_delta * 2.0,
            "economic_change": gold_delta / 100.0,
            "productive_social": 0.08 if action in {"talk", "trade", "work"} else 0.0,
            "danger_response": 0.25 if danger_was_active and action in {"flee", "hide"} else 0.0,
        }
        if danger_was_active and action in {"idle", "trade", "talk"}:
            components["ignored_danger"] = -0.25
        return _breakdown(components, "Civilian reward favors believable routine, social value, and safety.")


class CompanionRewardStrategy:
    """Support reward shaping for companions."""

    name = "companion"

    def compute(self, config: AgentConfig, state: GameState, action: str, next_state: GameState) -> RewardBreakdown:
        ally_health = float(next_state.value("ally_health", state.value("ally_health", 100.0)))
        previous_ally_health = float(state.value("ally_health", ally_health))
        ally_delta = ally_health - previous_ally_health
        components = {
            "ally_protection": ally_delta / 25.0,
            "stayed_close": 0.1 if action == "follow" else 0.0,
            "support_action": 0.12 if action in {"defend", "use_item", "follow"} else 0.0,
            "self_survival": 0.08 if next_state.health > 0 else -1.0,
        }
        return _breakdown(components, "Companion reward favors player support and survivability.")


def reward_strategy_for(agent_type: AgentType | str) -> RewardStrategy:
    agent_type = agent_type if isinstance(agent_type, AgentType) else AgentType(str(agent_type))
    return {
        AgentType.CIVILIAN: CivilianRewardStrategy(),
        AgentType.ENEMY: EnemyRewardStrategy(),
        AgentType.VILLAIN: VillainRewardStrategy(),
        AgentType.COMPANION: CompanionRewardStrategy(),
    }[agent_type]


def _breakdown(components: dict[str, float], reason: str) -> RewardBreakdown:
    return RewardBreakdown(total=sum(components.values()), components=components, reason=reason)


def _target_health(state: GameState) -> float:
    val = state.target_health if state.target_health is not None else state.value("target_health", 100.0)
    return float(val) if val is not None else 100.0


def _distance(state: GameState) -> float:
    val = state.distance_to_target if state.distance_to_target is not None else state.value("distance_to_target", 0.0)
    return float(val) if val is not None else 0.0
