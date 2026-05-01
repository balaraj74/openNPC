"""Multi-agent coordination for squads and encounter groups."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from opennpc.decision import DecisionEngine
from opennpc.types import ActionDecision, AgentConfig, DecisionTrace, GameState


@dataclass(slots=True)
class CoordinationAssignment:
    """A lightweight tactical role assigned during coordination."""

    agent_id: str
    role: str
    target_id: str | None
    action: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CoordinationPlan:
    """Coordinated decisions and the shared context used to produce them."""

    decisions: dict[str, ActionDecision]
    assignments: dict[str, CoordinationAssignment]
    shared_context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": {agent_id: decision.to_dict() for agent_id, decision in self.decisions.items()},
            "assignments": {
                agent_id: assignment.to_dict() for agent_id, assignment in self.assignments.items()
            },
            "shared_context": dict(self.shared_context),
        }


class MultiAgentCoordinator:
    """Coordinates independent decisions into a cleaner group plan.

    The coordinator deliberately stays small: it reuses ``DecisionEngine`` for
    per-agent intelligence, then applies deterministic encounter rules so a
    squad does not all choose the same tactical job at once.
    """

    def __init__(
        self,
        decision_engine: DecisionEngine | None = None,
        max_attackers_per_target: int = 2,
    ) -> None:
        self.decision_engine = decision_engine or DecisionEngine()
        self.max_attackers_per_target = max(1, int(max_attackers_per_target))

    def coordinate(
        self,
        configs: Sequence[AgentConfig],
        states: Sequence[GameState],
        available_actions: Mapping[str, list[str]] | None = None,
    ) -> CoordinationPlan:
        """Return coordinated decisions for a group of agents."""

        state_by_id = {state.agent_id: state for state in states}
        decisions: dict[str, ActionDecision] = {}
        assignments: dict[str, CoordinationAssignment] = {}

        for config in configs:
            state = state_by_id.get(config.agent_id)
            if state is None:
                continue
            decision = self.decision_engine.decide(
                config,
                state,
                available_actions=(available_actions or {}).get(config.agent_id),
                record_memory=False,
            )
            decisions[config.agent_id] = decision
            assignments[config.agent_id] = self._assignment_for(config, state, decision)

        self._balance_attack_pressure(configs, state_by_id, decisions, assignments, available_actions or {})
        self._remember_final_decisions(configs, state_by_id, decisions, assignments)
        shared_context = self._shared_context(decisions, assignments)
        return CoordinationPlan(decisions=decisions, assignments=assignments, shared_context=shared_context)

    def _balance_attack_pressure(
        self,
        configs: Sequence[AgentConfig],
        state_by_id: dict[str, GameState],
        decisions: dict[str, ActionDecision],
        assignments: dict[str, CoordinationAssignment],
        available_actions: Mapping[str, list[str]],
    ) -> None:
        config_by_id = {config.agent_id: config for config in configs}
        attackers_by_target: dict[str, list[str]] = defaultdict(list)
        for agent_id, decision in decisions.items():
            if decision.action == "attack":
                attackers_by_target[self._target_for(state_by_id[agent_id])].append(agent_id)

        for target_id, attackers in attackers_by_target.items():
            ranked = sorted(
                attackers,
                key=lambda agent_id: self._attack_rank(config_by_id[agent_id], state_by_id[agent_id], decisions[agent_id]),
                reverse=True,
            )
            for agent_id in ranked[self.max_attackers_per_target :]:
                config = config_by_id[agent_id]
                state = state_by_id[agent_id]
                valid_actions = self.decision_engine.valid_actions(
                    config,
                    state,
                    available_actions=available_actions.get(agent_id),
                )
                replacement = self._support_action(valid_actions, state)
                decisions[agent_id] = self._replace_action(
                    decisions[agent_id],
                    replacement,
                    f"Coordinated away from direct attack on {target_id}; squad pressure is already covered.",
                )
                assignments[agent_id] = self._assignment_for(config, state, decisions[agent_id])

    def _assignment_for(
        self,
        config: AgentConfig,
        state: GameState,
        decision: ActionDecision,
    ) -> CoordinationAssignment:
        target_id = self._target_for(state)
        role_by_action = {
            "attack": "assault",
            "flank": "flanker",
            "set_trap": "area_denial",
            "seek_cover": "support",
            "defend": "guard",
            "flee": "withdraw",
            "follow": "escort",
            "talk": "social",
            "trade": "support",
            "patrol": "screen",
            "move": "reposition",
        }
        role = role_by_action.get(decision.action, "hold")
        return CoordinationAssignment(
            agent_id=config.agent_id,
            role=role,
            target_id=target_id,
            action=decision.action,
            reason=decision.reason,
        )

    def _support_action(self, valid_actions: list[str], state: GameState) -> str:
        preferences = ["flank", "set_trap"]
        if state.cover_available:
            preferences.append("seek_cover")
        preferences.extend(["defend", "move", "patrol", "idle"])
        for action in preferences:
            if action in valid_actions:
                return action
        return valid_actions[0] if valid_actions else "idle"

    def _replace_action(self, decision: ActionDecision, action: str, reason: str) -> ActionDecision:
        trace = self._replace_trace(decision.trace, action, reason)
        return ActionDecision(
            agent_id=decision.agent_id,
            action=action,
            confidence=max(0.05, decision.confidence - 0.08),
            reason=reason,
            memory_update=decision.memory_update,
            trace=trace,
        )

    def _replace_trace(self, trace: DecisionTrace | None, action: str, reason: str) -> DecisionTrace | None:
        if trace is None:
            return None
        return replace(trace, final_action=action, reason=reason)

    def _remember_final_decisions(
        self,
        configs: Sequence[AgentConfig],
        state_by_id: dict[str, GameState],
        decisions: dict[str, ActionDecision],
        assignments: dict[str, CoordinationAssignment],
    ) -> None:
        config_by_id = {config.agent_id: config for config in configs}
        for agent_id, decision in decisions.items():
            config = config_by_id[agent_id]
            state = state_by_id[agent_id]
            update = self.decision_engine.remember_decision(
                config,
                state,
                decision.action,
                policy_name=decision.trace.policy_name if decision.trace else config.rl_policy,
            )
            decision.memory_update = update
            assignments[agent_id].reason = decision.reason

    def _attack_rank(self, config: AgentConfig, state: GameState, decision: ActionDecision) -> float:
        distance = state.distance_to_target if state.distance_to_target is not None else 99.0
        closeness = max(0.0, 1.0 - min(distance, 10.0) / 10.0)
        health = max(0.0, min(1.0, state.health / 100.0))
        return decision.confidence + config.personality.aggression * 0.3 + closeness * 0.25 + health * 0.1

    def _target_for(self, state: GameState) -> str:
        target_id = state.value("target_id")
        if target_id:
            return str(target_id)
        if "player" in state.nearby_entities:
            return "player"
        return "unknown"

    def _shared_context(
        self,
        decisions: dict[str, ActionDecision],
        assignments: dict[str, CoordinationAssignment],
    ) -> dict[str, Any]:
        action_counts = Counter(decision.action for decision in decisions.values())
        role_counts = Counter(assignment.role for assignment in assignments.values())
        target_pressure: Counter[str] = Counter()
        for assignment in assignments.values():
            if assignment.target_id:
                target_pressure[assignment.target_id] += 1
        return {
            "agent_count": len(decisions),
            "action_counts": dict(action_counts),
            "role_counts": dict(role_counts),
            "target_pressure": dict(target_pressure),
            "max_attackers_per_target": self.max_attackers_per_target,
        }
