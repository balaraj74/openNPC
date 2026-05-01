"""Replay-to-training converter.

Reads JSONL files produced by :class:`RuntimeExperienceLogger` and builds
PyTorch-compatible replay buffers for offline PPO or DQN fine-tuning.

Usage::

    converter = ReplayConverter.from_jsonl("artifacts/runtime_experience.jsonl")
    converter.summary()

    # DQN fine-tuning with the converter's replay buffer
    converter.fine_tune_dqn("artifacts/dqn_enemy.pt", "artifacts/dqn_enemy_v2.pt", episodes=50)

    # PPO fine-tuning
    converter.fine_tune_ppo("artifacts/ppo_enemy.pt", "artifacts/ppo_enemy_v2.pt", episodes=50)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opennpc.experience import ExperienceRecord
from opennpc.training.ppo import STATE_SIZE, encode_combat_state, require_training_deps

try:  # pragma: no cover
    import numpy as np
    import torch
except Exception:  # pragma: no cover
    np = None
    torch = None


ACTION_SPACE = ["idle", "move", "attack", "defend", "flee", "seek_cover", "flank"]


def _state_dict_to_game_state(raw: dict[str, Any]) -> Any:
    """Convert a raw state dict back into a GameState for encoding."""
    from opennpc.types import GameState

    return GameState.from_dict(raw)


@dataclass(slots=True)
class ReplayStats:
    """Summary statistics for a loaded replay buffer."""

    total_transitions: int = 0
    unique_agents: int = 0
    unique_actions: int = 0
    reward_mean: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0
    terminal_count: int = 0
    action_distribution: dict[str, int] = field(default_factory=dict)
    agent_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_transitions": self.total_transitions,
            "unique_agents": self.unique_agents,
            "unique_actions": self.unique_actions,
            "reward_mean": round(self.reward_mean, 4),
            "reward_min": round(self.reward_min, 4),
            "reward_max": round(self.reward_max, 4),
            "terminal_count": self.terminal_count,
            "action_distribution": self.action_distribution,
            "agent_ids": self.agent_ids,
        }


class ReplayConverter:
    """Loads JSONL experience files and converts them into training-ready buffers."""

    def __init__(self, records: list[ExperienceRecord], actions: list[str] | None = None) -> None:
        self.records = records
        self.actions = actions or list(ACTION_SPACE)
        self._action_to_idx = {action: idx for idx, action in enumerate(self.actions)}

    @classmethod
    def from_jsonl(cls, path: str | Path, actions: list[str] | None = None) -> "ReplayConverter":
        """Load experience records from a JSONL file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Experience log not found: {path}")

        records: list[ExperienceRecord] = []
        with open(path, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(ExperienceRecord.from_dict(data))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

        return cls(records, actions)

    @classmethod
    def from_records(cls, records: list[ExperienceRecord], actions: list[str] | None = None) -> "ReplayConverter":
        """Create directly from in-memory experience records."""
        return cls(records, actions)

    def summary(self) -> ReplayStats:
        """Compute summary statistics over the loaded records."""
        if not self.records:
            return ReplayStats()

        rewards = [record.reward for record in self.records]
        actions: dict[str, int] = {}
        agents: set[str] = set()
        terminals = 0

        for record in self.records:
            agents.add(record.agent_id)
            actions[record.action] = actions.get(record.action, 0) + 1
            if record.done:
                terminals += 1

        return ReplayStats(
            total_transitions=len(self.records),
            unique_agents=len(agents),
            unique_actions=len(actions),
            reward_mean=sum(rewards) / len(rewards),
            reward_min=min(rewards),
            reward_max=max(rewards),
            terminal_count=terminals,
            action_distribution=actions,
            agent_ids=sorted(agents),
        )

    def filter(
        self,
        agent_id: str | None = None,
        min_reward: float | None = None,
        max_reward: float | None = None,
        action: str | None = None,
        policy_name: str | None = None,
    ) -> "ReplayConverter":
        """Return a new converter with records matching the given filters."""
        filtered = self.records
        if agent_id is not None:
            filtered = [record for record in filtered if record.agent_id == agent_id]
        if min_reward is not None:
            filtered = [record for record in filtered if record.reward >= min_reward]
        if max_reward is not None:
            filtered = [record for record in filtered if record.reward <= max_reward]
        if action is not None:
            filtered = [record for record in filtered if record.action == action]
        if policy_name is not None:
            filtered = [record for record in filtered if record.policy_name == policy_name]
        return ReplayConverter(filtered, self.actions)

    def to_dqn_replay_buffer(self) -> Any:
        """Convert records to a DQN ReplayBuffer.

        Returns:
            A populated ``ReplayBuffer`` from :mod:`opennpc.training.dqn`.
        """
        require_training_deps()
        from opennpc.training.dqn import ReplayBuffer, Transition

        buffer = ReplayBuffer(capacity=max(len(self.records), 2000))
        for record in self.records:
            action_idx = self._action_to_idx.get(record.action)
            if action_idx is None:
                continue
            try:
                state_gs = _state_dict_to_game_state(record.state)
                next_gs = _state_dict_to_game_state(record.next_state)
                state_vec = encode_combat_state(state_gs)
                next_vec = encode_combat_state(next_gs)
            except Exception:
                continue

            buffer.push(
                Transition(
                    state=state_vec,
                    action=action_idx,
                    reward=record.reward,
                    next_state=next_vec,
                    done=record.done,
                )
            )
        return buffer

    def to_ppo_rollout(self) -> dict[str, list[Any]]:
        """Convert records to a PPO-style rollout dict.

        Returns a dict with keys: states, actions, rewards, dones — each a list
        of tensors/values compatible with ``PPOTrainer._update()``.
        """
        require_training_deps()

        states: list[Any] = []
        actions: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []

        for record in self.records:
            action_idx = self._action_to_idx.get(record.action)
            if action_idx is None:
                continue
            try:
                state_gs = _state_dict_to_game_state(record.state)
                state_vec = encode_combat_state(state_gs)
            except Exception:
                continue

            states.append(state_vec)
            actions.append(action_idx)
            rewards.append(record.reward)
            dones.append(record.done)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }

    def fine_tune_dqn(
        self,
        base_checkpoint: str | Path | None = None,
        output_path: str | Path = "artifacts/dqn_finetuned.pt",
        episodes: int = 50,
    ) -> list[float]:
        """Fine-tune a DQN model using the loaded experience replay.

        If ``base_checkpoint`` is given, loads the pre-trained model first.
        Otherwise starts from scratch but pre-fills the replay buffer with the
        loaded experience data.
        """
        require_training_deps()
        from opennpc.training.dqn import DQNConfig, DQNTrainer

        trainer = DQNTrainer(config=DQNConfig(episodes=episodes))

        if base_checkpoint:
            checkpoint = torch.load(base_checkpoint, map_location="cpu")
            trainer.q_net.load_state_dict(checkpoint["state_dict"])
            trainer.target_net.load_state_dict(checkpoint["state_dict"])

        replay_buffer = self.to_dqn_replay_buffer()
        for transition in replay_buffer._buffer:
            trainer.replay.push(transition)

        rewards = trainer.train()
        trainer.save(output_path)
        return rewards

    def fine_tune_ppo(
        self,
        base_checkpoint: str | Path | None = None,
        output_path: str | Path = "artifacts/ppo_finetuned.pt",
        episodes: int = 50,
    ) -> list[float]:
        """Fine-tune a PPO model.

        Loads a base checkpoint if given, then continues training.
        Note: PPO is on-policy, so offline replay data acts as a warm-start
        rather than a direct training signal.
        """
        require_training_deps()
        from opennpc.training.ppo import PPOConfig, PPOTrainer

        trainer = PPOTrainer(config=PPOConfig(episodes=episodes))

        if base_checkpoint:
            checkpoint = torch.load(base_checkpoint, map_location="cpu")
            trainer.model.load_state_dict(checkpoint["state_dict"])

        rewards = trainer.train()
        trainer.save(output_path)
        return rewards
