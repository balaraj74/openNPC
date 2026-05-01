"""A small village environment for non-combat OpenNPC demos.

This environment simulates a village square with NPCs that can trade, talk,
explore, and react to danger events. It demonstrates how personality, memory,
and goals drive everyday behavior — not just combat.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from opennpc.types import AgentType, GameState


class VillageAction(str, Enum):
    IDLE = "idle"
    MOVE = "move"
    TALK = "talk"
    TRADE = "trade"
    FOLLOW = "follow"
    FLEE = "flee"
    PATROL = "patrol"
    USE_ITEM = "use_item"
    HIDE = "hide"


VILLAGE_ACTIONS: tuple[str, ...] = tuple(action.value for action in VillageAction)


class Location(str, Enum):
    MARKET = "market"
    TAVERN = "tavern"
    WORKSHOP = "workshop"
    GATE = "gate"
    SQUARE = "square"
    FOREST_EDGE = "forest_edge"


LOCATION_LIST: list[str] = [loc.value for loc in Location]


@dataclass(slots=True)
class VillageStepResult:
    state: GameState
    reward: float
    done: bool
    info: dict[str, float | int | str | bool]


@dataclass(slots=True)
class VillagerState:
    agent_id: str
    location: str = "square"
    gold: int = 50
    reputation: float = 0.5
    conversations_today: int = 0
    trades_today: int = 0
    items: list[str] = field(default_factory=lambda: ["bread", "potion"])


class VillageEnv:
    """Turn-based social sandbox for civilian and companion agents.

    The environment tracks location, gold, reputation, conversations, and an
    optional danger event. It rewards social interaction and punishes inaction.
    """

    actions = VILLAGE_ACTIONS

    def __init__(
        self,
        max_steps: int = 40,
        seed: int | None = None,
        danger_chance: float = 0.08,
    ) -> None:
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.seed = seed
        self.danger_chance = danger_chance
        self.step_count = 0
        self.done = False
        self.danger_active = False
        self.last_action: str | None = None
        self.villager = VillagerState(agent_id="civilian_01")

    def reset(self, seed: int | None = None) -> GameState:
        if seed is not None:
            self.rng.seed(seed)
        self.step_count = 0
        self.done = False
        self.danger_active = False
        self.last_action = None
        self.villager = VillagerState(agent_id="civilian_01")
        return self.state()

    def state(self) -> GameState:
        nearby = self._nearby_entities()
        threat = 0.85 if self.danger_active else 0.0
        return GameState(
            agent_id=self.villager.agent_id,
            agent_type=AgentType.CIVILIAN,
            health=100.0,
            stamina=max(0.0, 100.0 - self.step_count * 1.2),
            location=self.villager.location,
            nearby_entities=nearby,
            threat_level=threat,
            inventory=list(self.villager.items),
            objective_status="daily_routine",
            current_goal="work",
            last_action=self.last_action,
            tick=self.step_count,
            extras={
                "gold": self.villager.gold,
                "reputation": round(self.villager.reputation, 3),
                "conversations_today": self.villager.conversations_today,
                "trades_today": self.villager.trades_today,
                "danger_active": self.danger_active,
                "recent_events": self._recent_events(),
            },
        )

    def step(self, action: str) -> VillageStepResult:
        if self.done:
            return VillageStepResult(self.state(), 0.0, True, {"terminal": True})

        action = action if action in self.actions else VillageAction.IDLE.value
        reward = -0.02  # small step penalty to encourage action

        if action == VillageAction.MOVE.value:
            self.villager.location = self.rng.choice(LOCATION_LIST)
            reward += 0.03

        elif action == VillageAction.TALK.value:
            if self._has_nearby_npc():
                self.villager.conversations_today += 1
                self.villager.reputation = min(1.0, self.villager.reputation + 0.04)
                reward += 0.15
            else:
                reward -= 0.05

        elif action == VillageAction.TRADE.value:
            if self.villager.location in {"market", "tavern"} and self.villager.gold >= 5:
                self.villager.gold -= 5
                self.villager.items.append(self.rng.choice(["herbs", "rope", "gem", "tool"]))
                self.villager.trades_today += 1
                self.villager.reputation = min(1.0, self.villager.reputation + 0.02)
                reward += 0.12
            else:
                reward -= 0.05

        elif action == VillageAction.FOLLOW.value:
            if self._has_nearby_npc():
                reward += 0.05
            else:
                reward -= 0.03

        elif action == VillageAction.FLEE.value:
            if self.danger_active:
                self.villager.location = "gate"
                reward += 0.35
            else:
                reward -= 0.15

        elif action == VillageAction.HIDE.value:
            if self.danger_active:
                reward += 0.25
            else:
                reward -= 0.1

        elif action == VillageAction.PATROL.value:
            old_loc = self.villager.location
            new_loc = self.rng.choice([loc for loc in LOCATION_LIST if loc != old_loc])
            self.villager.location = new_loc
            reward += 0.04

        elif action == VillageAction.USE_ITEM.value:
            if self.villager.items:
                used = self.villager.items.pop(0)
                reward += 0.08
            else:
                reward -= 0.05

        # Random danger event
        if not self.danger_active and self.rng.random() < self.danger_chance:
            self.danger_active = True
        elif self.danger_active and self.rng.random() < 0.4:
            self.danger_active = False

        self.step_count += 1
        self.last_action = action
        self.done = self.step_count >= self.max_steps

        # End-of-day bonus
        if self.done:
            reward += self.villager.reputation * 0.5
            reward += min(0.3, self.villager.trades_today * 0.05)
            reward += min(0.3, self.villager.conversations_today * 0.04)

        return VillageStepResult(
            state=self.state(),
            reward=float(reward),
            done=self.done,
            info={
                "action": action,
                "gold": self.villager.gold,
                "reputation": round(self.villager.reputation, 3),
                "location": self.villager.location,
                "danger": self.danger_active,
            },
        )

    def render_text(self) -> str:
        danger = " [!DANGER!]" if self.danger_active else ""
        return (
            f"[{self.villager.location:^14}] "
            f"gold={self.villager.gold:>3} "
            f"rep={self.villager.reputation:.2f} "
            f"items={len(self.villager.items)}"
            f"{danger}"
        )

    def _nearby_entities(self) -> list[str]:
        entities: list[str] = []
        if self.villager.location in {"market", "tavern", "square"}:
            entities.append("merchant")
            if self.rng.random() < 0.5:
                entities.append("traveler")
        if self.villager.location == "gate":
            entities.append("guard")
        if self.danger_active:
            entities.append("bandit")
        return entities

    def _has_nearby_npc(self) -> bool:
        return self.villager.location in {"market", "tavern", "square", "gate"}

    def _recent_events(self) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        if self.danger_active:
            events.append({"text": "Bandits have been spotted near the village!", "importance": 0.8})
        if self.villager.reputation >= 0.75:
            events.append({"text": "The villager is well-respected in the community.", "importance": 0.5})
        if self.villager.gold <= 10:
            events.append({"text": "Running low on gold. Need to trade carefully.", "importance": 0.6})
        return events
