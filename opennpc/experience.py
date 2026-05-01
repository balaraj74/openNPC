"""Runtime experience logging for online reward feedback."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import time
from typing import Any, Mapping


@dataclass(slots=True)
class ExperienceRecord:
    """One gameplay transition suitable for replay buffers or offline retraining."""

    agent_id: str
    state: dict[str, Any]
    action: str
    reward: float
    next_state: dict[str, Any]
    done: bool = False
    policy_name: str = "unknown"
    timestamp: float = field(default_factory=time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ExperienceRecord":
        return cls(
            agent_id=str(values["agent_id"]),
            state=dict(values["state"]),
            action=str(values["action"]),
            reward=float(values["reward"]),
            next_state=dict(values["next_state"]),
            done=bool(values.get("done", False)),
            policy_name=str(values.get("policy_name", "unknown")),
            timestamp=float(values.get("timestamp", time())),
            metadata=dict(values.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeExperienceLogger:
    """Records gameplay rewards as transition data.

    The logger is deliberately framework-neutral: it keeps recent records in
    memory and can optionally append each transition to JSONL for later PPO/DQN
    fine-tuning or imitation-style analysis.
    """

    def __init__(self, log_path: str | Path | None = None, max_records: int = 10_000) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.max_records = max(1, int(max_records))
        self._records: list[ExperienceRecord] = []
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        agent_id: str,
        state: Mapping[str, Any],
        action: str,
        reward: float,
        next_state: Mapping[str, Any],
        done: bool = False,
        policy_name: str = "unknown",
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperienceRecord:
        experience = ExperienceRecord(
            agent_id=agent_id,
            state=dict(state),
            action=action,
            reward=float(reward),
            next_state=dict(next_state),
            done=done,
            policy_name=policy_name,
            metadata=dict(metadata or {}),
        )
        self._records.append(experience)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]
        if self.log_path:
            self._append_jsonl(experience)
        return experience

    def recent(self, agent_id: str | None = None, limit: int = 20) -> list[ExperienceRecord]:
        records = self._records if agent_id is None else [item for item in self._records if item.agent_id == agent_id]
        return list(reversed(records[-limit:]))

    def summary(self) -> dict[str, Any]:
        if not self._records:
            return {"transitions": 0}
        rewards = [item.reward for item in self._records]
        by_agent: dict[str, int] = {}
        for item in self._records:
            by_agent[item.agent_id] = by_agent.get(item.agent_id, 0) + 1
        return {
            "transitions": len(self._records),
            "reward_mean": round(sum(rewards) / len(rewards), 4),
            "reward_min": round(min(rewards), 4),
            "reward_max": round(max(rewards), 4),
            "terminal_transitions": sum(1 for item in self._records if item.done),
            "by_agent": by_agent,
        }

    def clear(self, agent_id: str | None = None) -> None:
        if agent_id is None:
            self._records.clear()
        else:
            self._records = [item for item in self._records if item.agent_id != agent_id]

    def _append_jsonl(self, experience: ExperienceRecord) -> None:
        assert self.log_path is not None
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(experience.to_dict()) + "\n")
