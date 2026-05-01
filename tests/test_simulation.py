from opennpc.simulation.environment import GridCombatEnv


def test_grid_combat_environment_steps() -> None:
    env = GridCombatEnv(seed=1)
    state = env.reset()

    result = env.step("move")

    assert result.state.tick == state.tick + 1
    assert isinstance(result.reward, float)
    assert "enemy_health" in result.info


def test_attack_damages_player_when_close() -> None:
    env = GridCombatEnv(seed=2)
    env.reset()
    env.enemy_pos = 3
    env.player_pos = 4

    result = env.step("attack")

    assert result.info["player_damage"] > 0
