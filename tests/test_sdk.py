import asyncio

from opennpc import AgentConfig, GameState, Goal, OpenNPCSDK


def test_sdk_sync_decision_uses_cache_for_stable_state() -> None:
    sdk = OpenNPCSDK()
    config = AgentConfig(
        agent_id="sdk_enemy",
        goals=[Goal("survive", 1.0)],
        allowed_actions=["idle", "defend"],
    )
    state = GameState(agent_id="sdk_enemy", health=75)

    first = sdk.decide(config, state)
    second = sdk.decide(config, state)

    assert first.agent_id == "sdk_enemy"
    assert second.action == first.action
    assert "Cached decision" in second.reason
    sdk.close()


def test_sdk_async_decision_returns_action() -> None:
    sdk = OpenNPCSDK(cache_enabled=False)
    config = AgentConfig(
        agent_id="sdk_async_enemy",
        goals=[Goal("survive", 1.0)],
        allowed_actions=["idle", "defend"],
    )
    state = GameState(agent_id="sdk_async_enemy", health=75)

    decision = asyncio.run(sdk.decide_async(config, state))

    assert decision.action in config.allowed_actions
    sdk.close()

