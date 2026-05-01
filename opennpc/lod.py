"""AI Level-of-Detail (LOD) engine for performance scaling.

Assigns an LOD tier to every agent each frame based on distance to camera,
visibility, narrative importance, and population density. The tier controls
how much compute each agent receives:

  - FULL: Full decision engine (memory + policy + goals + trace)
  - REDUCED: Heuristic-only with no trace, lower-frequency updates
  - MINIMAL: Scripted fallback with no memory writes
  - DORMANT: No AI tick — the agent replays its last action or idles

This allows hundreds of NPCs on screen while keeping the important ones sharp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class LODTier(IntEnum):
    """LOD tiers in order of decreasing fidelity."""

    FULL = 0
    REDUCED = 1
    MINIMAL = 2
    DORMANT = 3


@dataclass(slots=True)
class LODProfile:
    """Per-tier configuration knobs."""

    tier: LODTier
    decision_interval_ms: int
    memory_enabled: bool
    trace_enabled: bool
    policy_override: str | None
    max_memory_events: int

    @classmethod
    def default_profiles(cls) -> dict[LODTier, "LODProfile"]:
        return {
            LODTier.FULL: cls(
                tier=LODTier.FULL,
                decision_interval_ms=250,
                memory_enabled=True,
                trace_enabled=True,
                policy_override=None,
                max_memory_events=50,
            ),
            LODTier.REDUCED: cls(
                tier=LODTier.REDUCED,
                decision_interval_ms=500,
                memory_enabled=True,
                trace_enabled=False,
                policy_override="heuristic",
                max_memory_events=20,
            ),
            LODTier.MINIMAL: cls(
                tier=LODTier.MINIMAL,
                decision_interval_ms=1000,
                memory_enabled=False,
                trace_enabled=False,
                policy_override="scripted_combat",
                max_memory_events=0,
            ),
            LODTier.DORMANT: cls(
                tier=LODTier.DORMANT,
                decision_interval_ms=5000,
                memory_enabled=False,
                trace_enabled=False,
                policy_override=None,
                max_memory_events=0,
            ),
        }


@dataclass(slots=True)
class AgentLODState:
    """Runtime LOD state for a single agent."""

    agent_id: str
    tier: LODTier = LODTier.FULL
    distance_to_camera: float = 0.0
    is_visible: bool = True
    narrative_importance: float = 0.5
    in_combat: bool = False


@dataclass(slots=True)
class LODConfig:
    """Thresholds for tier assignment."""

    full_distance: float = 15.0
    reduced_distance: float = 35.0
    minimal_distance: float = 60.0
    narrative_importance_boost: float = 0.5
    combat_always_full: bool = True
    dormant_if_invisible: bool = True


class LODEngine:
    """Evaluates and assigns LOD tiers to all agents each frame."""

    def __init__(
        self,
        config: LODConfig | None = None,
        profiles: dict[LODTier, LODProfile] | None = None,
    ) -> None:
        self.config = config or LODConfig()
        self.profiles = profiles or LODProfile.default_profiles()
        self._agents: dict[str, AgentLODState] = {}

    def register(self, agent_id: str, **kwargs: Any) -> AgentLODState:
        """Register an agent for LOD tracking."""
        state = AgentLODState(agent_id=agent_id, **kwargs)
        self._agents[agent_id] = state
        return state

    def unregister(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def update(
        self,
        agent_id: str,
        distance_to_camera: float | None = None,
        is_visible: bool | None = None,
        narrative_importance: float | None = None,
        in_combat: bool | None = None,
    ) -> LODTier:
        """Update an agent's spatial/narrative data and re-evaluate its tier."""
        state = self._agents.get(agent_id)
        if state is None:
            state = self.register(agent_id)
        if distance_to_camera is not None:
            state.distance_to_camera = distance_to_camera
        if is_visible is not None:
            state.is_visible = is_visible
        if narrative_importance is not None:
            state.narrative_importance = max(0.0, min(1.0, narrative_importance))
        if in_combat is not None:
            state.in_combat = in_combat
        state.tier = self._evaluate_tier(state)
        return state.tier

    def evaluate_all(self) -> dict[str, LODTier]:
        """Re-evaluate tiers for every registered agent."""
        result: dict[str, LODTier] = {}
        for agent_id, state in self._agents.items():
            state.tier = self._evaluate_tier(state)
            result[agent_id] = state.tier
        return result

    def profile_for(self, agent_id: str) -> LODProfile:
        """Return the LODProfile for an agent's current tier."""
        state = self._agents.get(agent_id)
        tier = state.tier if state else LODTier.FULL
        return self.profiles[tier]

    def tier_for(self, agent_id: str) -> LODTier:
        state = self._agents.get(agent_id)
        return state.tier if state else LODTier.FULL

    def summary(self) -> dict[str, Any]:
        """Summary of current LOD distribution across all agents."""
        counts = {tier.name: 0 for tier in LODTier}
        for state in self._agents.values():
            counts[state.tier.name] += 1
        return {
            "total_agents": len(self._agents),
            "distribution": counts,
        }

    def _evaluate_tier(self, state: AgentLODState) -> LODTier:
        cfg = self.config

        # Combat agents always get full fidelity
        if cfg.combat_always_full and state.in_combat:
            return LODTier.FULL

        # Invisible + not important → dormant
        if cfg.dormant_if_invisible and not state.is_visible and state.narrative_importance < 0.5:
            return LODTier.DORMANT

        # Effective distance accounts for narrative importance
        effective_distance = state.distance_to_camera - (
            state.narrative_importance * cfg.narrative_importance_boost * 10
        )

        if effective_distance <= cfg.full_distance:
            return LODTier.FULL
        if effective_distance <= cfg.reduced_distance:
            return LODTier.REDUCED
        if effective_distance <= cfg.minimal_distance:
            return LODTier.MINIMAL
        return LODTier.DORMANT

    def apply_to_config(self, agent_id: str, config: "AgentConfig") -> "AgentConfig":
        """Return a modified AgentConfig reflecting the agent's current LOD tier.

        Does not mutate the input — returns a new instance with adjusted fields.
        """
        from opennpc.types import AgentConfig as AC

        profile = self.profile_for(agent_id)
        overrides: dict[str, Any] = {
            "decision_interval_ms": profile.decision_interval_ms,
            "memory_enabled": profile.memory_enabled,
            "lod_level": profile.tier.name.lower(),
        }
        if profile.policy_override:
            overrides["rl_policy"] = profile.policy_override

        data = config.to_dict()
        data.update(overrides)
        return AC.from_dict(data)
