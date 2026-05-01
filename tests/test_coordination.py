from opennpc import AgentConfig, GameState, Goal, MultiAgentCoordinator, Personality


def _enemy(agent_id: str, aggression: float = 0.8) -> AgentConfig:
    return AgentConfig(
        agent_id=agent_id,
        personality=Personality(aggression=aggression, caution=0.2, risk_tolerance=0.8),
        goals=[Goal("attack_target", 1.0)],
        allowed_actions=["idle", "move", "attack", "defend", "flank", "set_trap"],
        seed=7,
    )


def _combat_state(agent_id: str, distance: float = 1.0) -> GameState:
    return GameState(
        agent_id=agent_id,
        health=90,
        threat_level=0.2,
        nearby_entities=["player"],
        target_health=35,
        distance_to_target=distance,
    )


def test_coordinator_limits_attackers_per_target() -> None:
    coordinator = MultiAgentCoordinator(max_attackers_per_target=1)
    configs = [_enemy("enemy_a", 0.95), _enemy("enemy_b", 0.8), _enemy("enemy_c", 0.7)]
    states = [_combat_state("enemy_a"), _combat_state("enemy_b"), _combat_state("enemy_c")]

    plan = coordinator.coordinate(configs, states)

    actions = [decision.action for decision in plan.decisions.values()]
    assert actions.count("attack") == 1
    assert set(actions) <= {"attack", "flank", "set_trap", "defend", "move", "idle"}
    assert plan.shared_context["max_attackers_per_target"] == 1


def test_coordinator_assigns_roles_for_actions() -> None:
    coordinator = MultiAgentCoordinator(max_attackers_per_target=2)
    configs = [_enemy("enemy_a"), _enemy("enemy_b")]
    states = [_combat_state("enemy_a"), _combat_state("enemy_b", distance=3.0)]

    plan = coordinator.coordinate(configs, states)

    assert plan.assignments["enemy_a"].role in {"assault", "flanker"}
    assert plan.assignments["enemy_a"].target_id == "player"
    assert plan.shared_context["agent_count"] == 2


def test_coordination_plan_serializes_to_dict() -> None:
    coordinator = MultiAgentCoordinator(max_attackers_per_target=1)
    plan = coordinator.coordinate([_enemy("enemy_a")], [_combat_state("enemy_a")])

    data = plan.to_dict()

    assert data["decisions"]["enemy_a"]["agent_id"] == "enemy_a"
    assert data["assignments"]["enemy_a"]["target_id"] == "player"
    assert data["shared_context"]["agent_count"] == 1


def test_coordinator_remembers_final_coordinated_action() -> None:
    coordinator = MultiAgentCoordinator(max_attackers_per_target=1)
    configs = [_enemy("enemy_a", 0.95), _enemy("enemy_b", 0.7)]
    states = [_combat_state("enemy_a"), _combat_state("enemy_b")]

    plan = coordinator.coordinate(configs, states)

    rerouted_id = next(agent_id for agent_id, decision in plan.decisions.items() if decision.action != "attack")
    recent = coordinator.decision_engine.memory_store.recent(rerouted_id, limit=1)
    assert recent[0].tags == ["decision", plan.decisions[rerouted_id].action]
