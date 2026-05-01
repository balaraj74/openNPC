"""A small grid combat environment for OpenNPC training and demos."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from opennpc.types import AgentType, GameState


class CombatAction(str, Enum):
    IDLE = "idle"
    MOVE = "move"
    ATTACK = "attack"
    DEFEND = "defend"
    FLEE = "flee"
    SEEK_COVER = "seek_cover"
    FLANK = "flank"
    SET_TRAP = "set_trap"
    PATROL = "patrol"


COMBAT_ACTIONS: tuple[str, ...] = tuple(action.value for action in CombatAction)


@dataclass(slots=True)
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: dict[str, float | int | str | bool]


class GridCombatEnv:
    """Tiny turn-based combat sandbox.

    The enemy tries to defeat a player on a one-dimensional lane with cover,
    traps, threat, and distance. It is intentionally simple but has enough
    structure for memory, personality, and RL policies to behave differently.
    """

    actions = COMBAT_ACTIONS

    def __init__(self, size: int = 8, max_steps: int = 60, seed: int | None = None) -> None:
        self.size = size
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.seed = seed
        self.enemy_pos = 0
        self.player_pos = size - 1
        self.enemy_health = 100.0
        self.player_health = 100.0
        self.cover_positions = {2, 5}
        self.trap_positions: set[int] = set()
        self.step_count = 0
        self.last_action: str | None = None
        self.done = False

    def reset(self, seed: int | None = None) -> GameState:
        if seed is not None:
            self.rng.seed(seed)
        self.enemy_pos = 0
        self.player_pos = self.size - 1
        self.enemy_health = 100.0
        self.player_health = 100.0
        self.cover_positions = {2, max(2, self.size - 3)}
        self.trap_positions = set()
        self.step_count = 0
        self.last_action = None
        self.done = False
        return self.state()

    def state(self) -> GameState:
        distance = abs(self.player_pos - self.enemy_pos)
        threat = self._threat_level(distance)
        nearby = ["player"] if distance <= 2 else []
        return GameState(
            agent_id="enemy_01",
            agent_type=AgentType.ENEMY,
            health=self.enemy_health,
            stamina=max(0.0, 100.0 - self.step_count * 0.8),
            location=f"lane:{self.enemy_pos}",
            nearby_entities=nearby,
            threat_level=threat,
            objective_status="combat",
            current_goal="attack_target",
            last_action=self.last_action,
            target_health=self.player_health,
            distance_to_target=float(distance),
            cover_available=self.enemy_pos in self.cover_positions,
            tick=self.step_count,
            extras={
                "enemy_pos": self.enemy_pos,
                "player_pos": self.player_pos,
                "trap_count": len(self.trap_positions),
                "recent_events": self._recent_events(distance),
            },
        )

    def step(self, action: str) -> StepResult:
        if self.done:
            return StepResult(self.state(), 0.0, True, {"terminal": True})

        action = action if action in self.actions else CombatAction.IDLE.value
        previous_distance = abs(self.player_pos - self.enemy_pos)
        reward = -0.01
        blocked_damage = False
        player_damage = 0.0
        enemy_damage = 0.0

        if action == CombatAction.MOVE.value:
            self.enemy_pos += self._direction_to_player()
            reward += 0.05 if abs(self.player_pos - self.enemy_pos) < previous_distance else -0.05
        elif action == CombatAction.FLANK.value:
            self.enemy_pos += self._direction_to_player()
            reward += 0.08
            blocked_damage = True
        elif action == CombatAction.FLEE.value:
            self.enemy_pos -= self._direction_to_player()
            reward += 0.05 if self.enemy_health < 45 else -0.05
            blocked_damage = True
        elif action == CombatAction.SEEK_COVER.value:
            self.enemy_pos = min(self.cover_positions, key=lambda pos: abs(pos - self.enemy_pos))
            reward += 0.1
            blocked_damage = True
        elif action == CombatAction.DEFEND.value:
            reward += 0.04
            blocked_damage = True
        elif action == CombatAction.ATTACK.value:
            if previous_distance <= 1:
                player_damage = self.rng.uniform(10, 18)
                reward += player_damage / 25.0
            else:
                reward -= 0.25
        elif action == CombatAction.SET_TRAP.value:
            trap_pos = self.enemy_pos + self._direction_to_player()
            if 0 <= trap_pos < self.size:
                self.trap_positions.add(trap_pos)
                reward += 0.08
            else:
                reward -= 0.05
        elif action == CombatAction.PATROL.value:
            self.enemy_pos += self.rng.choice([-1, 1])
            reward += 0.01

        self.enemy_pos = max(0, min(self.size - 1, self.enemy_pos))
        self.player_pos = self._move_player()

        if self.player_pos in self.trap_positions:
            player_damage += self.rng.uniform(12, 24)
            self.trap_positions.remove(self.player_pos)
            reward += 0.5

        distance = abs(self.player_pos - self.enemy_pos)
        if distance <= 1 and not blocked_damage:
            enemy_damage = self.rng.uniform(7, 15)
        elif distance <= 2 and not blocked_damage:
            enemy_damage = self.rng.uniform(2, 6)
        elif blocked_damage:
            reward += 0.03

        self.enemy_health = max(0.0, self.enemy_health - enemy_damage)
        self.player_health = max(0.0, self.player_health - player_damage)
        if enemy_damage:
            reward -= enemy_damage / 30.0

        self.step_count += 1
        self.last_action = action
        self.done = self.enemy_health <= 0 or self.player_health <= 0 or self.step_count >= self.max_steps
        if self.player_health <= 0:
            reward += 4.0
        if self.enemy_health <= 0:
            reward -= 4.0
        if self.done and self.enemy_health > 0 and self.player_health > 0:
            reward += (self.player_health < self.enemy_health) * 0.5

        return StepResult(
            state=self.state(),
            reward=float(reward),
            done=self.done,
            info={
                "action": action,
                "enemy_damage": round(enemy_damage, 3),
                "player_damage": round(player_damage, 3),
                "distance": distance,
                "enemy_health": round(self.enemy_health, 3),
                "player_health": round(self.player_health, 3),
            },
        )

    def render_text(self) -> str:
        cells = ["." for _ in range(self.size)]
        for cover in self.cover_positions:
            cells[cover] = "C"
        for trap in self.trap_positions:
            cells[trap] = "T"
        cells[self.enemy_pos] = "E"
        cells[self.player_pos] = "P" if cells[self.player_pos] == "." else cells[self.player_pos] + "P"
        return "".join(f"{cell:>2}" for cell in cells)

    def _direction_to_player(self) -> int:
        return 1 if self.player_pos > self.enemy_pos else -1 if self.player_pos < self.enemy_pos else 0

    def _move_player(self) -> int:
        distance = abs(self.player_pos - self.enemy_pos)
        if distance > 3:
            return self.player_pos - self._direction_to_player()
        if self.enemy_health < 35 and self.rng.random() < 0.45:
            return self.player_pos + self._direction_to_player()
        if self.rng.random() < 0.25:
            return self.player_pos + self.rng.choice([-1, 1])
        return self.player_pos

    def _threat_level(self, distance: int) -> float:
        proximity = max(0.0, 1.0 - distance / max(1, self.size - 1))
        health_pressure = max(0.0, 1.0 - self.enemy_health / 100.0)
        return max(0.0, min(1.0, proximity * 0.65 + health_pressure * 0.35))

    def _recent_events(self, distance: int) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        if distance <= 1:
            events.append({"text": "Player rushed aggressively into melee range.", "importance": 0.7})
        if self.enemy_health < 35:
            events.append({"text": "Enemy is badly hurt and survival is urgent.", "importance": 0.8})
        return events


def main() -> None:
    from opennpc import AgentConfig, DecisionEngine, Goal, Personality

    env = GridCombatEnv(seed=7)
    state = env.reset()
    engine = DecisionEngine()
    config = AgentConfig(
        agent_id="enemy_01",
        personality=Personality(aggression=0.75, caution=0.45, risk_tolerance=0.65),
        goals=[Goal("attack_target", 0.8), Goal("survive", 0.6), Goal("weaken_player", 0.5)],
        allowed_actions=list(COMBAT_ACTIONS),
    )
    total_reward = 0.0
    for _ in range(20):
        decision = engine.decide(config, state)
        result = env.step(decision.action)
        total_reward += result.reward
        print(env.render_text(), decision.action, f"reward={result.reward:.2f}")
        state = result.state
        if result.done:
            break
    print(f"total_reward={total_reward:.2f}")


if __name__ == "__main__":
    main()
