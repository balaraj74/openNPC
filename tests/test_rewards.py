from opennpc import (
    AgentConfig,
    AgentType,
    CivilianRewardStrategy,
    EnemyRewardStrategy,
    GameState,
    VillainRewardStrategy,
)


def test_enemy_reward_values_damage_and_survival() -> None:
    strategy = EnemyRewardStrategy()
    config = AgentConfig(agent_id="enemy_01", agent_type=AgentType.ENEMY)
    state = GameState(agent_id="enemy_01", health=90, target_health=80, distance_to_target=2.0)
    next_state = GameState(agent_id="enemy_01", health=85, target_health=55, distance_to_target=1.0)

    reward = strategy.compute(config, state, "attack", next_state)

    assert reward.total > 0
    assert reward.components["damage_dealt"] == 1.0
    assert reward.components["damage_taken"] < 0


def test_civilian_reward_penalizes_ignored_danger() -> None:
    strategy = CivilianRewardStrategy()
    config = AgentConfig(agent_id="civilian_01", agent_type=AgentType.CIVILIAN)
    state = GameState.from_dict({"agent_id": "civilian_01", "danger_active": True, "reputation": 0.5})
    next_state = GameState.from_dict({"agent_id": "civilian_01", "danger_active": True, "reputation": 0.5})

    reward = strategy.compute(config, state, "talk", next_state)

    assert reward.components["ignored_danger"] < 0
    assert reward.total < 0


def test_villain_reward_values_area_control() -> None:
    strategy = VillainRewardStrategy()
    config = AgentConfig(agent_id="villain_01", agent_type=AgentType.VILLAIN)
    state = GameState.from_dict({"agent_id": "villain_01", "trap_count": 1, "target_health": 90})
    next_state = GameState.from_dict({"agent_id": "villain_01", "trap_count": 2, "target_health": 90})

    reward = strategy.compute(config, state, "set_trap", next_state)

    assert reward.components["control"] > 0
    assert reward.components["trap_setup"] > 0
