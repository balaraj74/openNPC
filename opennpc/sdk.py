"""High-level SDK facade for embedding OpenNPC in game engines."""

from __future__ import annotations

from opennpc.async_engine import AsyncDecisionEngine
from opennpc.cache import DecisionCache
from opennpc.decision import DecisionEngine
from opennpc.memory import MemoryStore
from opennpc.policy import Policy
from opennpc.types import ActionDecision, AgentConfig, GameState


class OpenNPCSDK:
    """Production-friendly facade around the core decision engine.

    The facade gives game engine integrations one small surface for sync,
    async, cached, and scheduled decisions while still exposing the underlying
    engine for advanced policy registration.
    """

    def __init__(
        self,
        engine: DecisionEngine | None = None,
        memory_store: MemoryStore | None = None,
        policies: dict[str, Policy] | None = None,
        cache: DecisionCache | None = None,
        cache_enabled: bool = True,
        max_workers: int = 4,
    ) -> None:
        self.engine = engine or DecisionEngine(memory_store=memory_store, policies=policies)
        self.cache = cache or DecisionCache()
        self.cache_enabled = cache_enabled
        self.async_engine = AsyncDecisionEngine(engine=self.engine, max_workers=max_workers)

    def decide(
        self,
        config: AgentConfig,
        state: GameState,
        available_actions: list[str] | None = None,
        reward: float | None = None,
        event: str | None = None,
        use_cache: bool = True,
    ) -> ActionDecision:
        if self.cache_enabled and use_cache and reward is None and event is None:
            cached = self.cache.get(config, state)
            if cached is not None:
                return cached

        decision = self.engine.decide(config, state, available_actions, reward, event)
        if self.cache_enabled and use_cache:
            self.cache.put(config, state, decision)
        return decision

    async def decide_async(
        self,
        config: AgentConfig,
        state: GameState,
        available_actions: list[str] | None = None,
        reward: float | None = None,
        event: str | None = None,
    ) -> ActionDecision:
        return await self.async_engine.decide(config, state, available_actions, reward, event)

    def decide_scheduled(
        self,
        config: AgentConfig,
        state: GameState,
        available_actions: list[str] | None = None,
    ) -> ActionDecision:
        return self.engine.decide_scheduled(config, state, available_actions)

    def register_policy(self, name: str, policy: Policy) -> None:
        self.engine.register_policy(name, policy)
        self.cache.clear()

    def close(self) -> None:
        self.async_engine.shutdown()

    def __enter__(self) -> "OpenNPCSDK":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

