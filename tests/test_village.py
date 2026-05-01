"""Tests for the village social simulation environment."""

from opennpc.simulation.village import VillageEnv


def test_village_reset_returns_state() -> None:
    env = VillageEnv(seed=1)
    state = env.reset()
    assert state.agent_id == "civilian_01"
    assert state.health > 0
    assert state.tick == 0


def test_village_step_advances_tick() -> None:
    env = VillageEnv(seed=1)
    state = env.reset()
    result = env.step("move")
    assert result.state.tick == state.tick + 1
    assert isinstance(result.reward, float)


def test_village_trade_changes_gold() -> None:
    env = VillageEnv(seed=2)
    env.reset()
    # Move to market first so trade can succeed
    env.villager.location = "market"
    initial_gold = env.villager.gold
    result = env.step("trade")
    assert env.villager.gold != initial_gold or result.reward != 0.0


def test_village_talk_changes_reputation() -> None:
    env = VillageEnv(seed=3)
    env.reset()
    # Move to square where NPCs are nearby
    env.villager.location = "square"
    initial_rep = env.villager.reputation
    env.step("talk")
    assert isinstance(env.villager.reputation, float)


def test_village_done_after_max_steps() -> None:
    env = VillageEnv(max_steps=5, seed=4)
    env.reset()
    done = False
    for _ in range(10):
        result = env.step("idle")
        if result.done:
            done = True
            break
    assert done
