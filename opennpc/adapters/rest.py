"""Small standard-library REST client for the OpenNPC API."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence
from urllib import request

from opennpc.types import ActionDecision, AgentConfig, DecisionTrace, GameState


class RestDecisionClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8787", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def decide(
        self,
        config: AgentConfig,
        state: GameState,
        available_actions: list[str] | None = None,
        reward: float | None = None,
        event: str | None = None,
    ) -> ActionDecision:
        payload = {
            "config": config.to_dict(),
            "state": state.to_dict(),
            "available_actions": available_actions,
            "reward": reward,
            "event": event,
        }
        return self._decision_from_dict(self._post("/decide", payload))

    def decide_batch(
        self,
        requests: Sequence[tuple[AgentConfig, GameState]],
        available_actions: Mapping[str, list[str]] | None = None,
    ) -> list[ActionDecision]:
        payload = {
            "requests": [
                {
                    "config": config.to_dict(),
                    "state": state.to_dict(),
                    "available_actions": (available_actions or {}).get(config.agent_id),
                }
                for config, state in requests
            ]
        }
        data = self._post("/batch/decide", payload)
        return [self._decision_from_dict(item) for item in data.get("decisions", [])]

    def coordinate(
        self,
        configs: Sequence[AgentConfig],
        states: Sequence[GameState],
        available_actions: Mapping[str, list[str]] | None = None,
        max_attackers_per_target: int = 2,
    ) -> dict[str, Any]:
        return self._post(
            "/coordinate",
            {
                "configs": [config.to_dict() for config in configs],
                "states": [state.to_dict() for state in states],
                "available_actions": dict(available_actions or {}),
                "max_attackers_per_target": max_attackers_per_target,
            },
        )

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def memory(self, agent_id: str) -> dict[str, Any]:
        return self._get(f"/memory/{agent_id}")

    def experience_summary(self) -> dict[str, Any]:
        return self._get("/debug/experience")

    def experience(self, agent_id: str, limit: int = 20) -> dict[str, Any]:
        return self._get(f"/debug/experience/{agent_id}?limit={limit}")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get(self, path: str) -> dict[str, Any]:
        with request.urlopen(f"{self.base_url}{path}", timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _decision_from_dict(self, data: Mapping[str, Any]) -> ActionDecision:
        raw_trace = data.get("trace")
        trace = DecisionTrace(**raw_trace) if raw_trace else None
        return ActionDecision(
            agent_id=data["agent_id"],
            action=data["action"],
            confidence=float(data["confidence"]),
            reason=data["reason"],
            memory_update=data.get("memory_update"),
            trace=trace,
        )
