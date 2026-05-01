"""Policy evaluation helpers."""

from __future__ import annotations

from statistics import mean
from typing import Protocol

from opennpc import AgentConfig, DecisionEngine, Goal, Personality
from opennpc.simulation.baseline import ScriptedCombatAgent
from opennpc.simulation.environment import COMBAT_ACTIONS, GridCombatEnv


class AgentLike(Protocol):
    def act(self, state) -> str:
        ...


def evaluate_policy(agent: AgentLike, episodes: int = 10, seed: int = 11) -> dict[str, float]:
    rewards: list[float] = []
    lengths: list[int] = []
    actions: list[str] = []
    hurt_opportunities = 0
    adaptive_responses = 0
    wins = 0
    for idx in range(episodes):
        env = GridCombatEnv(seed=seed + idx)
        state = env.reset(seed=seed + idx)
        total = 0.0
        done = False
        steps = 0
        while not done:
            action = agent.act(state)
            if state.health < 35:
                hurt_opportunities += 1
                adaptive_responses += int(action in {"flee", "seek_cover", "defend"})
            actions.append(action)
            result = env.step(action)
            total += result.reward
            state = result.state
            done = result.done
            steps += 1
        rewards.append(total)
        lengths.append(steps)
        wins += int(env.player_health <= 0 and env.enemy_health > 0)
    return {
        "episodes": episodes,
        "average_reward": round(mean(rewards), 4),
        "win_rate": round(wins / episodes, 4),
        "avg_steps": round(mean(lengths), 2),
        "action_diversity": round(len(set(actions)) / len(COMBAT_ACTIONS), 4) if actions else 0.0,
        "adaptability": round(adaptive_responses / hurt_opportunities, 4) if hurt_opportunities else 0.0,
    }


class DecisionEngineCombatAgent:
    def __init__(self, engine: DecisionEngine | None = None, config: AgentConfig | None = None) -> None:
        self.engine = engine or DecisionEngine()
        self.config = config or AgentConfig(
            agent_id="enemy_01",
            personality=Personality(aggression=0.75, caution=0.45, risk_tolerance=0.65),
            goals=[Goal("attack_target", 0.8), Goal("survive", 0.65), Goal("weaken_player", 0.4)],
            allowed_actions=list(COMBAT_ACTIONS),
        )

    def act(self, state) -> str:
        return self.engine.decide(self.config, state).action


def compare_baseline(episodes: int = 10, seed: int = 11) -> dict[str, dict[str, float]]:
    return {
        "scripted": evaluate_policy(ScriptedCombatAgent(), episodes=episodes, seed=seed),
        "opennpc_heuristic": evaluate_policy(DecisionEngineCombatAgent(), episodes=episodes, seed=seed),
    }


def benchmark_combat(
    episodes: int = 20,
    seed: int = 11,
    ppo_model_path: str | None = None,
    dqn_model_path: str | None = None,
) -> dict[str, dict[str, float]]:
    results = compare_baseline(episodes=episodes, seed=seed)
    if ppo_model_path:
        ppo_config = AgentConfig(
            agent_id="enemy_01",
            rl_policy="ppo",
            metadata={"model_path": ppo_model_path},
            goals=[Goal("attack_target", 0.8), Goal("survive", 0.65), Goal("weaken_player", 0.4)],
            allowed_actions=list(COMBAT_ACTIONS),
        )
        results["ppo"] = evaluate_policy(DecisionEngineCombatAgent(config=ppo_config), episodes=episodes, seed=seed)
    if dqn_model_path:
        dqn_config = AgentConfig(
            agent_id="enemy_01",
            rl_policy="dqn",
            metadata={"model_path": dqn_model_path},
            goals=[Goal("attack_target", 0.8), Goal("survive", 0.65), Goal("weaken_player", 0.4)],
            allowed_actions=list(COMBAT_ACTIONS),
        )
        results["dqn"] = evaluate_policy(DecisionEngineCombatAgent(config=dqn_config), episodes=episodes, seed=seed)
    return results
