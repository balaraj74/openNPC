"""OpenNPC decision engine."""

from __future__ import annotations

from time import monotonic
from typing import Any, Callable

from opennpc.experience import RuntimeExperienceLogger
from opennpc.goals import GoalManager
from opennpc.memory import InMemoryMemoryStore, MemoryStore
from opennpc.policy import HeuristicPolicy, Policy, PolicyDecision, RandomPolicy, ScriptedCombatPolicy
from opennpc.rewards import RewardBreakdown, RewardStrategy, reward_strategy_for
from opennpc.types import ActionDecision, AgentConfig, DecisionTrace, GameState, MemoryEvent


PolicyLoader = Callable[[str], Policy]


def _load_ppo_policy(path: str) -> Policy:
    from opennpc.training.ppo import PPOPytorchPolicy

    return PPOPytorchPolicy.from_file(path)


def _load_dqn_policy(path: str) -> Policy:
    from opennpc.training.dqn import DQNPolicy

    return DQNPolicy.from_file(path)


def _load_onnx_policy(path: str) -> Policy:
    from opennpc.training.onnx_policy import ONNXPolicy

    return ONNXPolicy.from_file(path)


class DecisionEngine:
    """Combines state, memory, goals, personality, policy, and validation."""

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        policies: dict[str, Policy] | None = None,
        policy_loaders: dict[str, PolicyLoader] | None = None,
        goal_manager: GoalManager | None = None,
        experience_logger: RuntimeExperienceLogger | None = None,
        reward_strategies: dict[str, RewardStrategy] | None = None,
    ) -> None:
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.goal_manager = goal_manager or GoalManager()
        self.experience_logger = experience_logger
        self.reward_strategies = {
            "civilian": reward_strategy_for("civilian"),
            "enemy": reward_strategy_for("enemy"),
            "villain": reward_strategy_for("villain"),
            "companion": reward_strategy_for("companion"),
        }
        if reward_strategies:
            self.reward_strategies.update(reward_strategies)
        self.policies: dict[str, Policy] = {
            "heuristic": HeuristicPolicy(),
            "scripted_combat": ScriptedCombatPolicy(),
            "random": RandomPolicy(),
        }
        if policies:
            self.policies.update(policies)
        self.policy_loaders: dict[str, PolicyLoader] = {
            "ppo": _load_ppo_policy,
            "dqn": _load_dqn_policy,
            "onnx": _load_onnx_policy,
        }
        if policy_loaders:
            self.policy_loaders.update(policy_loaders)
        self._loaded_policy_cache: dict[tuple[str, str], Policy] = {}
        self.fallback_policy = self.policies["heuristic"]
        self._last_decision_at: dict[str, float] = {}
        self._last_decision: dict[str, ActionDecision] = {}
        self._last_state_snapshot: dict[str, dict[str, Any]] = {}
        self._last_action_by_agent: dict[str, str] = {}
        self._last_policy_by_agent: dict[str, str] = {}

    def register_policy(self, name: str, policy: Policy) -> None:
        self.policies[name] = policy

    def register_policy_loader(self, name: str, loader: PolicyLoader) -> None:
        self.policy_loaders[name] = loader

    def register_reward_strategy(self, agent_type: str, strategy: RewardStrategy) -> None:
        self.reward_strategies[agent_type] = strategy

    def register_policy_from_file(self, name: str, path: str, policy_name: str | None = None) -> Policy:
        loader = self.policy_loaders[name]
        policy = loader(path)
        self.policies[policy_name or name] = policy
        self._loaded_policy_cache[(name, path)] = policy
        return policy

    def decide(
        self,
        config: AgentConfig,
        state: GameState,
        available_actions: list[str] | None = None,
        reward: float | None = None,
        event: str | None = None,
        record_memory: bool = True,
    ) -> ActionDecision:
        if state.agent_id != config.agent_id:
            state = GameState.from_dict({**state.to_dict(), "agent_id": config.agent_id})

        self._record_runtime_experience(config, state, reward)
        self._ingest_runtime_memory(config, state, reward, event)
        memory_summary = self.memory_store.summarize(config.agent_id) if config.memory_enabled else ""
        valid_actions = self.valid_actions(config, state, available_actions)
        goal_evaluation = self.goal_manager.evaluate(config, state, memory_summary)
        context = {
            "memory_summary": memory_summary,
            "goal_scores": goal_evaluation.action_scores,
            "active_goals": goal_evaluation.active_goals,
        }

        policy, policy_resolution_note, fallback_used = self._resolve_policy(config)
        try:
            policy_decision = policy.select_action(config, state, valid_actions, context)
        except Exception as exc:  # pragma: no cover - defensive runtime path
            fallback_used = True
            policy_decision = self.fallback_policy.select_action(config, state, valid_actions, context)
            policy_decision.reason = f"Policy failed with {type(exc).__name__}; {policy_decision.reason}"
        if policy_resolution_note:
            policy_decision.reason = f"{policy_resolution_note} {policy_decision.reason}"

        final_action = policy_decision.action
        if final_action not in valid_actions:
            fallback_used = True
            policy_decision = self._fallback_decision(config, state, valid_actions, context, final_action)
            final_action = policy_decision.action

        if record_memory:
            memory_update = self.remember_decision(
                config,
                state,
                final_action,
                policy_name=getattr(policy, "name", config.rl_policy),
            )
        else:
            memory_update = self._memory_update_for_decision(config, state, final_action)

        trace = DecisionTrace(
            policy_name=getattr(policy, "name", config.rl_policy),
            policy_action=policy_decision.action,
            final_action=final_action,
            confidence=policy_decision.confidence,
            reason=policy_decision.reason,
            fallback_used=fallback_used,
            valid_actions=valid_actions,
            goal_scores=goal_evaluation.action_scores,
            personality_influence=policy_decision.metadata.get("personality_scores", {}),
            memory_summary=memory_summary,
            state_snapshot=state.to_dict(),
        )
        outgoing_action = final_action
        if outgoing_action == "flee" and "retreat" in config.allowed_actions:
            outgoing_action = "retreat"

        decision = ActionDecision(
            agent_id=config.agent_id,
            action=outgoing_action,
            confidence=policy_decision.confidence,
            reason=policy_decision.reason,
            memory_update=memory_update,
            trace=trace,
        )
        self._last_decision_at[config.agent_id] = monotonic()
        self._last_decision[config.agent_id] = decision
        self._last_state_snapshot[config.agent_id] = state.to_dict()
        self._last_action_by_agent[config.agent_id] = final_action
        self._last_policy_by_agent[config.agent_id] = trace.policy_name
        return decision

    def decide_batch(
        self,
        requests: list[tuple[AgentConfig, GameState]],
        available_actions: list[str] | None = None,
    ) -> list[ActionDecision]:
        return [self.decide(config, state, available_actions=available_actions) for config, state in requests]

    def decide_scheduled(
        self,
        config: AgentConfig,
        state: GameState,
        available_actions: list[str] | None = None,
    ) -> ActionDecision:
        last_time = self._last_decision_at.get(config.agent_id)
        interval_seconds = max(0, config.decision_interval_ms) / 1000.0
        if last_time is not None and monotonic() - last_time < interval_seconds:
            last_decision = self._last_decision[config.agent_id]
            return ActionDecision(
                agent_id=last_decision.agent_id,
                action=last_decision.action,
                confidence=max(0.01, last_decision.confidence - 0.05),
                reason="Reused previous decision because the scheduled update interval has not elapsed.",
                memory_update=None,
                trace=last_decision.trace,
            )
        return self.decide(config, state, available_actions=available_actions)

    def last_decision(self, agent_id: str) -> ActionDecision | None:
        return self._last_decision.get(agent_id)

    def debug_decisions(self) -> list[dict[str, Any]]:
        return [decision.to_dict() for decision in self._last_decision.values()]

    def remember_decision(
        self,
        config: AgentConfig,
        state: GameState,
        action: str,
        policy_name: str | None = None,
    ) -> str | None:
        memory_update = self._memory_update_for_decision(config, state, action)
        if memory_update and config.memory_enabled:
            self.memory_store.add(
                MemoryEvent(
                    agent_id=config.agent_id,
                    text=memory_update,
                    importance=self._decision_importance(state, action),
                    tags=["decision", action],
                    metadata={"tick": state.tick, "policy": policy_name or config.rl_policy},
                )
            )
        return memory_update

    def _resolve_policy(self, config: AgentConfig) -> tuple[Policy, str, bool]:
        requested = config.rl_policy
        if requested in self.policies:
            return self.policies[requested], "", False

        model_path = self._model_path_for(config)
        model_type = str(config.metadata.get("model_type", requested))
        if model_path and model_type in self.policy_loaders:
            cache_key = (model_type, model_path)
            if cache_key not in self._loaded_policy_cache:
                try:
                    self._loaded_policy_cache[cache_key] = self.policy_loaders[model_type](model_path)
                except Exception as exc:  # pragma: no cover - depends on optional model deps/files.
                    note = (
                        f"Policy '{model_type}' failed to load from '{model_path}' "
                        f"({type(exc).__name__}); using heuristic fallback."
                    )
                    return self.fallback_policy, note, True
            return self._loaded_policy_cache[cache_key], "", False

        if requested in self.policy_loaders:
            note = f"Policy '{requested}' requires config.metadata['model_path']; using heuristic fallback."
        else:
            note = f"Policy '{requested}' is not registered; using heuristic fallback."
        return self.fallback_policy, note, True

    def _model_path_for(self, config: AgentConfig) -> str | None:
        for key in ("model_path", "policy_path", "rl_model_path"):
            value = config.metadata.get(key)
            if value:
                return str(value)
        return None

    def _record_runtime_experience(self, config: AgentConfig, state: GameState, reward: float | None) -> None:
        if self.experience_logger is None:
            return
        previous_state = self._last_state_snapshot.get(config.agent_id)
        previous_action = self._last_action_by_agent.get(config.agent_id)
        if previous_state is None or previous_action is None:
            return
        shaped = self._shape_reward(config, previous_state, previous_action, state)
        shaping_enabled = bool(config.metadata.get("reward_shaping", True))
        shaping_weight = float(config.metadata.get("reward_shaping_weight", 1.0))
        external_reward = 0.0 if reward is None else float(reward)
        total_reward = external_reward + (shaped.total * shaping_weight if shaping_enabled else 0.0)
        self.experience_logger.record(
            agent_id=config.agent_id,
            state=previous_state,
            action=previous_action,
            reward=total_reward,
            next_state=state.to_dict(),
            done=state.health <= 0 or bool(state.value("done", False)),
            policy_name=self._last_policy_by_agent.get(config.agent_id, config.rl_policy),
            metadata={
                "tick": state.tick,
                "external_reward": reward,
                "reward_strategy": shaped.reason,
                "shaped_reward": shaped.to_dict(),
                "reward_shaping_weight": shaping_weight if shaping_enabled else 0.0,
            },
        )

    def _shape_reward(
        self,
        config: AgentConfig,
        previous_state: dict[str, Any],
        previous_action: str,
        next_state: GameState,
    ) -> RewardBreakdown:
        strategy_name = str(config.metadata.get("reward_strategy", config.agent_type.value))
        strategy = self.reward_strategies.get(strategy_name, self.reward_strategies[config.agent_type.value])
        state = GameState.from_dict(previous_state)
        return strategy.compute(config, state, previous_action, next_state)

    def valid_actions(
        self,
        config: AgentConfig,
        state: GameState,
        available_actions: list[str] | None = None,
    ) -> list[str]:
        # Action translation layer for external clients (e.g. Java "retreat" -> "flee")
        allowed_list = list(config.allowed_actions)
        if "retreat" in allowed_list and "flee" not in allowed_list:
            allowed_list.append("flee")

        candidates = list(available_actions if available_actions is not None else allowed_list)
        if "retreat" in candidates and "flee" not in candidates:
            candidates.append("flee")

        candidates = [action for action in candidates if action in allowed_list]
        constraints = set(config.constraints)

        if state.health <= 0:
            return ["idle"] if "idle" in allowed_list else []
        if "pacifist" in constraints:
            candidates = [action for action in candidates if action not in {"attack", "flank"}]
        if "no_traps" in constraints:
            candidates = [action for action in candidates if action != "set_trap"]
        if not state.cover_available:
            candidates = [action for action in candidates if action != "seek_cover"]
        if not state.inventory:
            candidates = [action for action in candidates if action != "use_item"]

        distance = state.distance_to_target
        attack_range = float(config.metadata.get("attack_range", 16.0))
        has_target = (
            state.target_health is not None
            or "player" in state.nearby_entities
            or state.value("target_id")
            # Infer target if distance_to_target was provided (client measured it)
            or (distance is not None and distance >= 0)
        )
        if not has_target:
            candidates = [action for action in candidates if action not in {"attack", "flank"}]
        elif distance is not None and distance > attack_range:
            candidates = [action for action in candidates if action != "attack"]

        deduped: list[str] = []
        for action in candidates:
            if action not in deduped:
                deduped.append(action)
        return deduped or (["idle"] if "idle" in allowed_list else [])

    def _fallback_decision(
        self,
        config: AgentConfig,
        state: GameState,
        valid_actions: list[str],
        context: dict[str, Any],
        invalid_action: str,
    ) -> PolicyDecision:
        decision = self.fallback_policy.select_action(config, state, valid_actions, context)
        decision.reason = f"Policy selected invalid action '{invalid_action}'; {decision.reason}"
        return decision

    def _ingest_runtime_memory(
        self,
        config: AgentConfig,
        state: GameState,
        reward: float | None,
        event: str | None,
    ) -> None:
        if not config.memory_enabled:
            return
        if event:
            self.memory_store.add(
                MemoryEvent(config.agent_id, event, importance=0.7, tags=["event"], metadata={"tick": state.tick})
            )
        for raw_event in state.value("recent_events", []) or []:
            if isinstance(raw_event, dict):
                memory_event = MemoryEvent.from_dict({"agent_id": config.agent_id, **raw_event})
            else:
                memory_event = MemoryEvent(config.agent_id, str(raw_event), importance=0.55, tags=["state_event"])
            self.memory_store.add(memory_event)
        if reward is not None:
            text = f"Received reward {reward:.2f} after {state.last_action or 'unknown action'}."
            self.memory_store.add(
                MemoryEvent(
                    config.agent_id,
                    text,
                    importance=0.8 if reward < 0 else 0.45,
                    tags=["reward"],
                    metadata={"reward": reward, "tick": state.tick},
                )
            )

    def _memory_update_for_decision(self, config: AgentConfig, state: GameState, action: str) -> str | None:
        if not config.memory_enabled:
            return None
        if action in {"flee", "seek_cover"} and state.threat_level >= 0.5:
            return f"Chose {action} under threat level {state.threat_level:.2f}."
        if action == "attack" and state.target_health is not None:
            return f"Pressed attack against target health {state.target_health:.0f}."
        if action == "set_trap":
            return "Adapted by setting a trap against the player's pattern."
        if action == "flank":
            return "Tried a flank instead of repeating a direct approach."
        return f"Chose {action}."

    def _decision_importance(self, state: GameState, action: str) -> float:
        if action in {"flee", "seek_cover", "set_trap", "flank"}:
            return 0.72
        if state.threat_level > 0.8 or state.health < 30:
            return 0.75
        return 0.4
