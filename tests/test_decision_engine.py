from opennpc import AgentConfig, DecisionEngine, GameState, Goal, MemoryEvent, Personality, RuntimeExperienceLogger


def test_aggressive_enemy_attacks_when_target_is_close() -> None:
    engine = DecisionEngine()
    config = AgentConfig(
        agent_id="enemy_01",
        personality=Personality(aggression=0.95, caution=0.1, risk_tolerance=0.9),
        goals=[Goal("attack_target", 1.0)],
        allowed_actions=["idle", "move", "attack", "defend", "flee", "seek_cover", "flank"],
        seed=1,
    )
    state = GameState(
        agent_id="enemy_01",
        health=90,
        threat_level=0.2,
        nearby_entities=["player"],
        target_health=24,
        distance_to_target=1.0,
        cover_available=True,
    )

    decision = engine.decide(config, state)

    assert decision.action == "attack"
    assert decision.trace is not None
    assert decision.trace.fallback_used is False


def test_cautious_enemy_survives_when_badly_hurt() -> None:
    engine = DecisionEngine()
    config = AgentConfig(
        agent_id="enemy_01",
        personality=Personality(aggression=0.1, caution=0.95, risk_tolerance=0.05),
        goals=[Goal("survive", 1.0), Goal("attack_target", 0.2)],
        allowed_actions=["idle", "move", "attack", "defend", "flee", "seek_cover", "flank"],
        seed=2,
    )
    state = GameState(
        agent_id="enemy_01",
        health=18,
        threat_level=0.9,
        nearby_entities=["player"],
        target_health=90,
        distance_to_target=1.0,
        cover_available=True,
    )

    decision = engine.decide(config, state)

    assert decision.action in {"flee", "seek_cover", "defend"}


def test_memory_changes_repeated_direct_attack() -> None:
    engine = DecisionEngine()
    engine.memory_store.add(
        MemoryEvent(
            agent_id="enemy_01",
            text="Direct attack failed attack against the player.",
            importance=0.9,
            tags=["combat"],
        )
    )
    config = AgentConfig(
        agent_id="enemy_01",
        personality=Personality(aggression=0.6, caution=0.6, risk_tolerance=0.4),
        goals=[Goal("attack_target", 0.7), Goal("weaken_player", 0.7)],
        allowed_actions=["idle", "move", "attack", "defend", "flee", "seek_cover", "flank", "set_trap"],
        seed=3,
    )
    state = GameState(
        agent_id="enemy_01",
        health=75,
        threat_level=0.45,
        nearby_entities=["player"],
        target_health=80,
        distance_to_target=1.0,
        cover_available=False,
    )

    decision = engine.decide(config, state)

    assert decision.action in {"flank", "set_trap"}


def test_invalid_policy_action_uses_fallback() -> None:
    class BadPolicy:
        name = "bad"

        def select_action(self, config, state, valid_actions, context=None):
            from opennpc.policy import PolicyDecision

            return PolicyDecision("teleport", 0.99, "Bad action.")

    engine = DecisionEngine(policies={"bad": BadPolicy()})
    config = AgentConfig(
        agent_id="enemy_01",
        rl_policy="bad",
        goals=[Goal("survive", 1.0)],
        allowed_actions=["idle", "defend", "flee"],
    )
    state = GameState(agent_id="enemy_01", health=20, threat_level=0.8)

    decision = engine.decide(config, state)

    assert decision.action in {"defend", "flee", "idle"}
    assert decision.trace is not None
    assert decision.trace.fallback_used is True


def test_dynamic_policy_loader_uses_model_path_and_caches_policy() -> None:
    from opennpc.policy import PolicyDecision

    load_calls = []

    class LoadedPolicy:
        name = "loaded_ppo"

        def select_action(self, config, state, valid_actions, context=None):
            return PolicyDecision("defend", 0.91, "Loaded runtime policy selected defend.")

    def loader(path: str):
        load_calls.append(path)
        return LoadedPolicy()

    engine = DecisionEngine(policy_loaders={"ppo": loader})
    config = AgentConfig(
        agent_id="enemy_01",
        rl_policy="ppo",
        allowed_actions=["idle", "attack", "defend"],
        metadata={"model_path": "artifacts/fake_ppo.pt"},
    )
    state = GameState(agent_id="enemy_01", health=90, nearby_entities=["player"], distance_to_target=1.0)

    first = engine.decide(config, state)
    second = engine.decide(config, state)

    assert first.action == "defend"
    assert second.action == "defend"
    assert load_calls == ["artifacts/fake_ppo.pt"]
    assert first.trace is not None
    assert first.trace.policy_name == "loaded_ppo"
    assert first.trace.fallback_used is False


def test_missing_rl_model_path_marks_fallback() -> None:
    engine = DecisionEngine()
    config = AgentConfig(
        agent_id="enemy_01",
        rl_policy="ppo",
        goals=[Goal("attack_target", 1.0)],
        allowed_actions=["idle", "attack", "defend"],
    )
    state = GameState(
        agent_id="enemy_01",
        health=90,
        nearby_entities=["player"],
        target_health=20,
        distance_to_target=1.0,
    )

    decision = engine.decide(config, state)

    assert decision.trace is not None
    assert decision.trace.fallback_used is True
    assert "requires config.metadata['model_path']" in decision.reason


def test_runtime_experience_logger_records_reward_feedback() -> None:
    logger = RuntimeExperienceLogger()
    engine = DecisionEngine(experience_logger=logger)
    config = AgentConfig(
        agent_id="enemy_01",
        goals=[Goal("attack_target", 1.0)],
        allowed_actions=["idle", "attack", "defend"],
    )
    first_state = GameState(
        agent_id="enemy_01",
        health=90,
        nearby_entities=["player"],
        target_health=25,
        distance_to_target=1.0,
    )
    next_state = GameState(
        agent_id="enemy_01",
        health=85,
        nearby_entities=["player"],
        target_health=10,
        distance_to_target=1.0,
        tick=1,
    )

    first = engine.decide(config, first_state)
    engine.decide(config, next_state, reward=1.25)

    recent = logger.recent("enemy_01", limit=1)
    assert len(recent) == 1
    assert recent[0].action == first.action
    assert recent[0].reward > 1.25
    assert recent[0].metadata["external_reward"] == 1.25
    assert recent[0].metadata["shaped_reward"]["total"] > 0
    assert logger.summary()["transitions"] == 1


def test_reward_shaping_can_be_disabled() -> None:
    logger = RuntimeExperienceLogger()
    engine = DecisionEngine(experience_logger=logger)
    config = AgentConfig(
        agent_id="enemy_01",
        metadata={"reward_shaping": False},
        goals=[Goal("attack_target", 1.0)],
        allowed_actions=["idle", "attack", "defend"],
    )
    first_state = GameState(
        agent_id="enemy_01",
        health=90,
        nearby_entities=["player"],
        target_health=25,
        distance_to_target=1.0,
    )
    next_state = GameState(
        agent_id="enemy_01",
        health=85,
        nearby_entities=["player"],
        target_health=10,
        distance_to_target=1.0,
        tick=1,
    )

    engine.decide(config, first_state)
    engine.decide(config, next_state, reward=1.25)

    recent = logger.recent("enemy_01", limit=1)
    assert recent[0].reward == 1.25
    assert recent[0].metadata["reward_shaping_weight"] == 0.0
