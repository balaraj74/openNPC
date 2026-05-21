"""Small PPO implementation for the reference combat environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opennpc.model_registry import build_metadata, embed_metadata, validate_checkpoint
from opennpc.policy import PolicyDecision
from opennpc.simulation.environment import COMBAT_ACTIONS, GridCombatEnv
from opennpc.types import AgentConfig, GameState

try:  # pragma: no cover - import behavior depends on optional dependency.
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
except Exception as exc:  # pragma: no cover
    np = None
    torch = None
    nn = None
    optim = None
    Categorical = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


STATE_SIZE = 9


def require_training_deps() -> None:
    if torch is None or np is None:
        raise RuntimeError("PPO training requires the optional training dependencies: numpy and torch.") from _IMPORT_ERROR


def encode_combat_state(state: GameState) -> list[float]:
    distance = float(state.distance_to_target if state.distance_to_target is not None else 8.0)
    target_health = float(state.target_health if state.target_health is not None else 100.0)
    recent_reward = sum(state.recent_rewards[-3:]) / 3.0 if state.recent_rewards else 0.0
    return [
        state.health / 100.0,
        state.stamina / 100.0,
        target_health / 100.0,
        min(1.0, distance / 8.0),
        state.threat_level,
        1.0 if state.cover_available else 0.0,
        float(state.value("enemy_pos", 0)) / 8.0,
        float(state.value("player_pos", 7)) / 8.0,
        max(-1.0, min(1.0, recent_reward)),
    ]


class ActorCritic(nn.Module if nn else object):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64) -> None:
        require_training_deps()
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, states: Any) -> tuple[Any, Any]:
        features = self.shared(states)
        return self.actor(features), self.critic(features).squeeze(-1)


@dataclass(slots=True)
class PPOConfig:
    episodes: int = 120
    gamma: float = 0.96
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    update_epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    seed: int = 7


class PPOTrainer:
    def __init__(
        self,
        env: GridCombatEnv | None = None,
        actions: list[str] | None = None,
        config: PPOConfig | None = None,
    ) -> None:
        require_training_deps()
        self.env = env or GridCombatEnv(seed=7)
        self.actions = actions or list(COMBAT_ACTIONS)
        self.config = config or PPOConfig()
        torch.manual_seed(self.config.seed)
        self.model = ActorCritic(STATE_SIZE, len(self.actions))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def train(self) -> list[float]:
        returns_history: list[float] = []
        for episode in range(self.config.episodes):
            rollout = self._collect_episode(seed=self.config.seed + episode)
            returns_history.append(sum(rollout["rewards"]))
            self._update(rollout)
        return returns_history

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"state_dict": self.model.state_dict(), "actions": self.actions}
        metadata = build_metadata("ppo", STATE_SIZE, self.actions)
        embed_metadata(checkpoint, metadata)
        torch.save(checkpoint, path)

    def _collect_episode(self, seed: int) -> dict[str, list[Any]]:
        state = self.env.reset(seed=seed)
        rollout: dict[str, list[Any]] = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "dones": [],
        }
        done = False
        while not done:
            tensor = torch.tensor(encode_combat_state(state), dtype=torch.float32).unsqueeze(0)
            logits, value = self.model(tensor)
            dist = Categorical(logits=logits)
            action_idx = dist.sample()
            result = self.env.step(self.actions[int(action_idx.item())])
            rollout["states"].append(tensor.squeeze(0))
            rollout["actions"].append(action_idx.squeeze(0))
            rollout["log_probs"].append(dist.log_prob(action_idx).squeeze(0).detach())
            rollout["values"].append(value.squeeze(0).detach())
            rollout["rewards"].append(result.reward)
            rollout["dones"].append(result.done)
            state = result.state
            done = result.done
        return rollout

    def _update(self, rollout: dict[str, list[Any]]) -> None:
        states = torch.stack(rollout["states"])
        actions = torch.stack(rollout["actions"])
        old_log_probs = torch.stack(rollout["log_probs"])
        values = torch.stack(rollout["values"])
        returns = self._discounted_returns(rollout["rewards"], rollout["dones"])
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.config.update_epochs):
            logits, new_values = self.model(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            ratios = torch.exp(new_log_probs - old_log_probs)
            clipped = torch.clamp(ratios, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
            policy_loss = -torch.min(ratios * advantages, clipped * advantages).mean()
            value_loss = (returns - new_values).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _discounted_returns(self, rewards: list[float], dones: list[bool]) -> Any:
        returns: list[float] = []
        running = 0.0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            running = reward + self.config.gamma * running * (0.0 if done else 1.0)
            returns.append(running)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)


class PPOPytorchPolicy:
    name = "ppo"

    def __init__(self, model: ActorCritic, actions: list[str]) -> None:
        require_training_deps()
        self.model = model
        self.model.eval()
        self.actions = actions

    @classmethod
    def from_file(cls, path: str | Path) -> "PPOPytorchPolicy":
        require_training_deps()
        checkpoint = torch.load(path, map_location="cpu")
        validate_checkpoint(
            checkpoint,
            expected_model_type="ppo",
            expected_state_size=STATE_SIZE,
        )
        actions = list(checkpoint["actions"])
        model = ActorCritic(STATE_SIZE, len(actions))
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, actions)

    def export_onnx(self, path: str | Path) -> Path:
        """Export this PPO policy to ONNX format for dependency-free inference.

        Produces two files:
        - ``<path>``: the ONNX model (actor head only; outputs action logits).
        - ``<path>.meta.json``: ``action_space`` and ``state_size`` metadata
          consumed by :class:`~opennpc.training.onnx_policy.ONNXPolicy`.

        Parameters
        ----------
        path:
            Destination ``.onnx`` file path.

        Returns
        -------
        Path
            The resolved path to the written ONNX file.

        Example
        -------
        ::

            policy = PPOPytorchPolicy.from_file("enemy.pt")
            policy.export_onnx("enemy.onnx")

            # Load at runtime (no PyTorch needed):
            from opennpc.training.onnx_policy import ONNXPolicy
            onnx_policy = ONNXPolicy.from_file("enemy.onnx")
        """
        require_training_deps()
        try:
            import json as _json
            import torch.onnx as _onnx  # noqa: F401 — verify torch.onnx is present
        except ImportError as exc:
            raise ImportError("torch is required to export ONNX models.") from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Wrap to export only the actor head (logits only, no value head).
        class _ActorOnly(torch.nn.Module):
            def __init__(self, actor_critic: ActorCritic) -> None:
                super().__init__()
                self._m = actor_critic

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                logits, _ = self._m(x)
                return logits

        actor_only = _ActorOnly(self.model)
        actor_only.eval()
        dummy_input = torch.zeros(1, STATE_SIZE, dtype=torch.float32)

        torch.onnx.export(
            actor_only,
            dummy_input,
            str(path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["state"],
            output_names=["logits"],
            dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}},
        )

        # Write companion metadata file.
        meta_path = path.with_suffix(".onnx.meta.json")
        meta_path.write_text(_json.dumps({
            "action_space": self.actions,
            "state_size": STATE_SIZE,
            "model_type": "ppo",
            "sdk_version": "0.1.0",
        }, indent=2))

        return path

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
                "PPO policy action head had no overlap with valid actions; selected first valid fallback.",
            )
        with torch.no_grad():
            tensor = torch.tensor(encode_combat_state(state), dtype=torch.float32).unsqueeze(0)
            logits, _ = self.model(tensor)
            logits = logits.squeeze(0)
            mask = torch.tensor([action in valid_actions for action in self.actions], dtype=torch.bool)
            masked_logits = torch.where(mask, logits, torch.tensor(-1e9))
            probabilities = torch.softmax(masked_logits, dim=0)
            action_idx = int(torch.argmax(probabilities).item())
            action = self.actions[action_idx]
            confidence = float(probabilities[action_idx].item())
        return PolicyDecision(action, confidence, "PPO policy selected the highest-probability valid action.")


def train_and_save(path: str | Path, episodes: int = 120) -> list[float]:
    trainer = PPOTrainer(config=PPOConfig(episodes=episodes))
    rewards = trainer.train()
    trainer.save(path)
    return rewards
