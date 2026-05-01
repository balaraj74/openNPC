"""DQN training implementation for the reference combat environment.

Provides a secondary RL algorithm alongside PPO. DQN is simpler and useful
for comparison and for scenarios where PPO's on-policy nature is not ideal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opennpc.model_registry import build_metadata, embed_metadata, validate_checkpoint
from opennpc.policy import PolicyDecision
from opennpc.simulation.environment import COMBAT_ACTIONS, GridCombatEnv
from opennpc.training.ppo import STATE_SIZE, encode_combat_state, require_training_deps
from opennpc.types import AgentConfig, GameState

try:  # pragma: no cover
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover
    np = None
    torch = None
    nn = None
    optim = None


class QNetwork(nn.Module if nn else object):
    """Simple feedforward Q-network."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64) -> None:
        require_training_deps()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x)


@dataclass(slots=True)
class DQNConfig:
    episodes: int = 200
    gamma: float = 0.95
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 32
    replay_capacity: int = 2000
    target_update_freq: int = 10
    seed: int = 7


@dataclass(slots=True)
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 2000) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        import random as rng

        return rng.sample(list(self._buffer), min(batch_size, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)


class DQNTrainer:
    """DQN trainer for the GridCombatEnv."""

    def __init__(
        self,
        env: GridCombatEnv | None = None,
        actions: list[str] | None = None,
        config: DQNConfig | None = None,
    ) -> None:
        require_training_deps()
        self.env = env or GridCombatEnv(seed=7)
        self.actions = actions or list(COMBAT_ACTIONS)
        self.config = config or DQNConfig()
        torch.manual_seed(self.config.seed)

        action_size = len(self.actions)
        self.q_net = QNetwork(STATE_SIZE, action_size)
        self.target_net = QNetwork(STATE_SIZE, action_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_capacity)
        self.epsilon = self.config.epsilon_start

    def train(self) -> list[float]:
        """Run DQN training loop. Returns per-episode total rewards."""
        import random as rng

        returns_history: list[float] = []
        for episode in range(self.config.episodes):
            state = self.env.reset(seed=self.config.seed + episode)
            total_reward = 0.0
            done = False

            while not done:
                state_vec = encode_combat_state(state)
                if rng.random() < self.epsilon:
                    action_idx = rng.randrange(len(self.actions))
                else:
                    with torch.no_grad():
                        q_vals = self.q_net(torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0))
                        action_idx = int(q_vals.argmax(dim=1).item())

                result = self.env.step(self.actions[action_idx])
                next_state_vec = encode_combat_state(result.state)
                self.replay.push(Transition(state_vec, action_idx, result.reward, next_state_vec, result.done))
                total_reward += result.reward
                state = result.state
                done = result.done

                if len(self.replay) >= self.config.batch_size:
                    self._optimize()

            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
            if (episode + 1) % self.config.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
            returns_history.append(total_reward)

        return returns_history

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"state_dict": self.q_net.state_dict(), "actions": self.actions}
        metadata = build_metadata("dqn", STATE_SIZE, self.actions)
        embed_metadata(checkpoint, metadata)
        torch.save(checkpoint, path)

    def _optimize(self) -> None:
        batch = self.replay.sample(self.config.batch_size)
        states = torch.tensor([t.state for t in batch], dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.tensor([t.next_state for t in batch], dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)

        q_values = self.q_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.config.gamma * max_next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DQNPolicy:
    """Runtime policy backed by a trained DQN model."""

    name = "dqn"

    def __init__(self, model: QNetwork, actions: list[str]) -> None:
        require_training_deps()
        self.model = model
        self.model.eval()
        self.actions = actions

    @classmethod
    def from_file(cls, path: str | Path) -> "DQNPolicy":
        require_training_deps()
        checkpoint = torch.load(path, map_location="cpu")
        validate_checkpoint(
            checkpoint,
            expected_model_type="dqn",
            expected_state_size=STATE_SIZE,
        )
        actions = list(checkpoint["actions"])
        model = QNetwork(STATE_SIZE, len(actions))
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, actions)

    def select_action(
        self,
        config: AgentConfig,
        state: GameState,
        valid_actions: list[str],
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        if not valid_actions:
            return PolicyDecision("idle", 1.0, "No valid actions were available.")
        if not any(action in valid_actions for action in self.actions):
            return PolicyDecision(
                valid_actions[0],
                0.2,
                "DQN policy action head had no overlap with valid actions; selected first valid fallback.",
            )
        with torch.no_grad():
            tensor = torch.tensor(encode_combat_state(state), dtype=torch.float32).unsqueeze(0)
            q_values = self.model(tensor).squeeze(0)
            mask = torch.tensor([a in valid_actions for a in self.actions], dtype=torch.bool)
            masked_q = torch.where(mask, q_values, torch.tensor(-1e9))
            action_idx = int(masked_q.argmax().item())
            action = self.actions[action_idx]
            confidence = float(torch.softmax(masked_q, dim=0)[action_idx].item())
        return PolicyDecision(action, confidence, "DQN policy selected the highest Q-value valid action.")


def train_dqn_and_save(path: str | Path, episodes: int = 200) -> list[float]:
    """Convenience function to train DQN and save the checkpoint."""
    trainer = DQNTrainer(config=DQNConfig(episodes=episodes))
    rewards = trainer.train()
    trainer.save(path)
    return rewards
