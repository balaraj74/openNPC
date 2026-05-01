"""Async wrapper for the OpenNPC decision engine.

Provides an ``AsyncDecisionEngine`` that wraps the synchronous
:class:`~opennpc.decision.DecisionEngine` and offloads ``decide`` calls to a
background thread pool. This prevents blocking the game loop's main thread in
engines (like Unreal or Godot) that receive AI results via callbacks.

Usage::

    engine = AsyncDecisionEngine()
    decision = await engine.decide(config, state)

For batch decisions::

    decisions = await engine.decide_batch([(config_a, state_a), (config_b, state_b)])
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from opennpc.decision import DecisionEngine
from opennpc.memory import MemoryStore
from opennpc.policy import Policy
from opennpc.types import ActionDecision, AgentConfig, GameState


class AsyncDecisionEngine:
    """Non-blocking async wrapper around :class:`DecisionEngine`.

    All CPU-bound decision work runs in a thread pool so the caller can
    ``await`` without blocking the event loop.
    """

    def __init__(
        self,
        engine: DecisionEngine | None = None,
        memory_store: MemoryStore | None = None,
        policies: dict[str, Policy] | None = None,
        max_workers: int = 4,
    ) -> None:
        self.engine = engine or DecisionEngine(memory_store=memory_store, policies=policies)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="opennpc")

    async def decide(
        self,
        config: AgentConfig,
        state: GameState,
        available_actions: list[str] | None = None,
        reward: float | None = None,
        event: str | None = None,
    ) -> ActionDecision:
        """Run the decision engine in a worker thread and return the result."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool,
            lambda: self.engine.decide(config, state, available_actions, reward, event),
        )

    async def decide_batch(
        self,
        requests: list[tuple[AgentConfig, GameState]],
        available_actions: list[str] | None = None,
    ) -> list[ActionDecision]:
        """Execute batch decisions concurrently across the thread pool."""
        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(
                self._pool,
                lambda cfg=config, st=state: self.engine.decide(cfg, st, available_actions=available_actions),
            )
            for config, state in requests
        ]
        return list(await asyncio.gather(*futures))

    async def decide_scheduled(
        self,
        config: AgentConfig,
        state: GameState,
        available_actions: list[str] | None = None,
    ) -> ActionDecision:
        """Async version of scheduled decision (respects decision_interval_ms)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool,
            lambda: self.engine.decide_scheduled(config, state, available_actions),
        )

    def register_policy(self, name: str, policy: Policy) -> None:
        """Register a policy on the underlying engine (not async — safe to call anytime)."""
        self.engine.register_policy(name, policy)

    @property
    def memory_store(self) -> MemoryStore:
        return self.engine.memory_store

    def shutdown(self) -> None:
        """Gracefully shut down the thread pool."""
        self._pool.shutdown(wait=False)
