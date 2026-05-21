"""Shared OpenNPC data models.

The core models are dataclasses instead of framework-specific schemas so the SDK
can run embedded in a game loop, inside a web service, or in tests without extra
runtime dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from enum import Enum
from time import time
from typing import Any, Mapping, Sequence


class AgentType(str, Enum):
    CIVILIAN = "civilian"
    ENEMY = "enemy"
    VILLAIN = "villain"
    COMPANION = "companion"


DEFAULT_ACTIONS: tuple[str, ...] = (
    "idle",
    "move",
    "talk",
    "trade",
    "follow",
    "flee",
    "attack",
    "defend",
    "patrol",
    "use_item",
    "hide",
    "seek_cover",
    "flank",
    "set_trap",
)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class Personality:
    aggression: float = 0.5
    curiosity: float = 0.5
    sociability: float = 0.5
    caution: float = 0.5
    patience: float = 0.5
    loyalty: float = 0.5
    risk_tolerance: float = 0.5

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, clamp01(getattr(self, name)))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> "Personality":
        if not values:
            return cls()
        allowed = {field.name for field in dataclass_fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def vector(self) -> list[float]:
        return [
            self.aggression,
            self.curiosity,
            self.sociability,
            self.caution,
            self.patience,
            self.loyalty,
            self.risk_tolerance,
        ]


@dataclass(slots=True)
class Goal:
    name: str
    priority: float = 1.0
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | str) -> "Goal":
        if isinstance(values, str):
            return cls(name=values)
        return cls(
            name=str(values["name"]),
            priority=float(values.get("priority", 1.0)),
            active=bool(values.get("active", True)),
            metadata=dict(values.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentConfig:
    agent_id: str
    agent_type: AgentType | str = AgentType.ENEMY
    personality: Personality = field(default_factory=Personality)
    goals: list[Goal] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=lambda: list(DEFAULT_ACTIONS))
    memory_enabled: bool = True
    rl_policy: str = "heuristic"
    decision_interval_ms: int = 250
    lod_level: str = "medium"
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.agent_type, AgentType):
            self.agent_type = AgentType(str(self.agent_type))
        self.goals = [goal if isinstance(goal, Goal) else Goal.from_dict(goal) for goal in self.goals]
        self.personality = (
            self.personality
            if isinstance(self.personality, Personality)
            else Personality.from_dict(self.personality)
        )
        self.allowed_actions = [str(action) for action in self.allowed_actions]
        self.constraints = [str(constraint) for constraint in self.constraints]

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AgentConfig":
        data = dict(values)
        data["personality"] = Personality.from_dict(data.get("personality"))
        data["goals"] = [Goal.from_dict(goal) for goal in data.get("goals", [])]
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "personality": self.personality.to_dict(),
            "goals": [goal.to_dict() for goal in self.goals],
            "constraints": list(self.constraints),
            "allowed_actions": list(self.allowed_actions),
            "memory_enabled": self.memory_enabled,
            "rl_policy": self.rl_policy,
            "decision_interval_ms": self.decision_interval_ms,
            "lod_level": self.lod_level,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class GameState:
    agent_id: str
    agent_type: AgentType | str = AgentType.ENEMY
    health: float = 100.0
    stamina: float = 100.0
    location: str = "unknown"
    nearby_entities: list[str] = field(default_factory=list)
    threat_level: float = 0.0
    inventory: list[str] = field(default_factory=list)
    objective_status: str = "unknown"
    current_goal: str | None = None
    last_action: str | None = None
    recent_rewards: list[float] = field(default_factory=list)
    target_health: float | None = None
    distance_to_target: float | None = None
    cover_available: bool = False
    tick: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.agent_type, AgentType):
            self.agent_type = AgentType(str(self.agent_type))
        self.health = float(self.health)
        self.stamina = float(self.stamina)
        self.threat_level = clamp01(float(self.threat_level))
        self.nearby_entities = [str(entity) for entity in self.nearby_entities]
        self.inventory = [str(item) for item in self.inventory]

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GameState":
        known = {field.name for field in dataclass_fields(cls)}
        direct = {key: value for key, value in values.items() if key in known and key != "extras"}
        extras = dict(values.get("extras", {}))
        extras.update({key: value for key, value in values.items() if key not in known})
        return cls(**direct, extras=extras)

    def value(self, name: str, default: Any = None) -> Any:
        if hasattr(self, name):
            val = getattr(self, name)
            if val is not None:
                return val
        return self.extras.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "health": self.health,
            "stamina": self.stamina,
            "location": self.location,
            "nearby_entities": list(self.nearby_entities),
            "threat_level": self.threat_level,
            "inventory": list(self.inventory),
            "objective_status": self.objective_status,
            "current_goal": self.current_goal,
            "last_action": self.last_action,
            "recent_rewards": list(self.recent_rewards),
            "target_health": self.target_health,
            "distance_to_target": self.distance_to_target,
            "cover_available": self.cover_available,
            "tick": self.tick,
        }
        data.update(self.extras)
        return data


@dataclass(slots=True)
class MemoryEvent:
    agent_id: str
    text: str
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.importance = clamp01(self.importance)
        self.tags = [str(tag) for tag in self.tags]

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "MemoryEvent":
        return cls(
            agent_id=str(values["agent_id"]),
            text=str(values["text"]),
            importance=float(values.get("importance", 0.5)),
            tags=list(values.get("tags", [])),
            timestamp=float(values.get("timestamp", time())),
            metadata=dict(values.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DecisionTrace:
    policy_name: str
    policy_action: str
    final_action: str
    confidence: float
    reason: str
    fallback_used: bool
    valid_actions: list[str]
    goal_scores: dict[str, float]
    personality_influence: dict[str, float]
    memory_summary: str
    state_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ActionDecision:
    agent_id: str
    action: str
    confidence: float
    reason: str
    memory_update: str | None = None
    trace: DecisionTrace | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "agent_id": self.agent_id,
            "action": self.action,
            "confidence": self.confidence,
            "reason": self.reason,
            "memory_update": self.memory_update,
            "trace": self.trace.to_dict() if self.trace else None,
        }
        return data


def normalize_actions(actions: Sequence[str] | None) -> list[str]:
    if actions is None:
        return list(DEFAULT_ACTIONS)
    return [str(action) for action in actions]
