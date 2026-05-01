"""Training logger for recording rewards, states, and action history.

Provides structured episode logging for both PPO and DQN training runs.
Logs are stored as JSON-lines files for easy post-hoc analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import time
from typing import Any


@dataclass(slots=True)
class StepLog:
    """One step within a training episode."""

    tick: int
    action: str
    reward: float
    health: float
    target_health: float
    distance: float
    threat_level: float
    done: bool


@dataclass(slots=True)
class EpisodeLog:
    """Full record of one training episode."""

    episode: int
    total_reward: float
    steps: int
    win: bool
    timestamp: float = field(default_factory=time)
    step_logs: list[StepLog] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingLogger:
    """Logs training episodes to a JSON-lines file and accumulates metrics."""

    def __init__(self, log_path: str | Path | None = None) -> None:
        self.log_path = Path(log_path) if log_path else None
        self._episodes: list[EpisodeLog] = []
        self._current_steps: list[StepLog] = []
        self._current_episode: int = 0
        self._current_reward: float = 0.0
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def begin_episode(self, episode: int) -> None:
        """Start recording a new episode."""
        self._current_episode = episode
        self._current_steps = []
        self._current_reward = 0.0

    def log_step(
        self,
        tick: int,
        action: str,
        reward: float,
        health: float = 100.0,
        target_health: float = 100.0,
        distance: float = 0.0,
        threat_level: float = 0.0,
        done: bool = False,
    ) -> None:
        """Record a single step within the current episode."""
        self._current_reward += reward
        self._current_steps.append(
            StepLog(
                tick=tick,
                action=action,
                reward=round(reward, 4),
                health=round(health, 2),
                target_health=round(target_health, 2),
                distance=round(distance, 2),
                threat_level=round(threat_level, 3),
                done=done,
            )
        )

    def end_episode(self, win: bool = False, metadata: dict[str, Any] | None = None) -> EpisodeLog:
        """Finalize and store the current episode log."""
        episode = EpisodeLog(
            episode=self._current_episode,
            total_reward=round(self._current_reward, 4),
            steps=len(self._current_steps),
            win=win,
            step_logs=list(self._current_steps),
            metadata=metadata or {},
        )
        self._episodes.append(episode)
        if self.log_path:
            self._write_episode(episode)
        return episode

    def summary(self) -> dict[str, Any]:
        """Return aggregate training metrics."""
        if not self._episodes:
            return {"episodes": 0}
        rewards = [ep.total_reward for ep in self._episodes]
        wins = sum(1 for ep in self._episodes if ep.win)
        return {
            "episodes": len(self._episodes),
            "total_reward_mean": round(sum(rewards) / len(rewards), 4),
            "total_reward_min": round(min(rewards), 4),
            "total_reward_max": round(max(rewards), 4),
            "win_rate": round(wins / len(self._episodes), 4),
            "avg_steps": round(sum(ep.steps for ep in self._episodes) / len(self._episodes), 2),
        }

    def action_distribution(self) -> dict[str, int]:
        """Return action frequencies across all logged episodes."""
        counts: dict[str, int] = {}
        for ep in self._episodes:
            for step in ep.step_logs:
                counts[step.action] = counts.get(step.action, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    def reward_curve(self) -> list[float]:
        """Return a list of total rewards per episode — useful for plotting."""
        return [ep.total_reward for ep in self._episodes]

    def _write_episode(self, episode: EpisodeLog) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            record = {
                "episode": episode.episode,
                "total_reward": episode.total_reward,
                "steps": episode.steps,
                "win": episode.win,
                "timestamp": episode.timestamp,
                "metadata": episode.metadata,
                "actions": [step.action for step in episode.step_logs],
            }
            f.write(json.dumps(record) + "\n")
