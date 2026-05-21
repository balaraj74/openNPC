"""Tests for gRPC service layer — no live gRPC server needed."""

from __future__ import annotations

import json

import pytest

from opennpc.proto.opennpc_pb2 import (
    BatchDecideRequest,
    DecideRequest,
    HealthRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_servicer():
    from opennpc.api.grpc_service import _OpenNPCServicer
    from opennpc.decision import DecisionEngine
    from opennpc.experience import RuntimeExperienceLogger

    engine = DecisionEngine(experience_logger=RuntimeExperienceLogger())
    return _OpenNPCServicer(engine)


class _FakeContext:
    """Minimal stand-in for grpc.ServicerContext."""

    def __init__(self) -> None:
        self._code = None
        self._details = ""

    def set_code(self, code: object) -> None:
        self._code = code

    def set_details(self, details: str) -> None:
        self._details = details


def _decide_request(
    agent_id: str = "guard",
    location: str = "forest",
    actions: list[str] | None = None,
) -> DecideRequest:
    if actions is None:
        actions = ["patrol", "attack", "retreat"]
    config = {
        "agent_id": agent_id,
        "agent_type": "enemy",
        "allowed_actions": actions,
    }
    state = {"agent_id": agent_id, "location": location, "threat_level": 0.3}
    return DecideRequest(
        agent_config_json=json.dumps(config),
        game_state_json=json.dumps(state),
        available_actions=actions,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestGRPCHealth:
    def test_health_returns_ok(self):
        servicer = _make_servicer()
        resp = servicer.Health(HealthRequest(), _FakeContext())
        assert resp.status == "ok"
        assert resp.service == "opennpc"


# ---------------------------------------------------------------------------
# Decide
# ---------------------------------------------------------------------------

class TestGRPCDecide:
    def test_decide_returns_action(self):
        servicer = _make_servicer()
        req = _decide_request()
        resp = servicer.Decide(req, _FakeContext())
        assert resp.action in ["patrol", "attack", "retreat", "idle"]
        assert resp.agent_id == "guard"

    def test_decide_confidence_in_range(self):
        servicer = _make_servicer()
        resp = servicer.Decide(_decide_request(), _FakeContext())
        assert 0.0 <= resp.confidence <= 1.0

    def test_decide_invalid_json_falls_back(self):
        servicer = _make_servicer()
        req = DecideRequest(
            agent_config_json="NOT JSON",
            game_state_json="{}",
        )
        resp = servicer.Decide(req, _FakeContext())
        assert resp.action == "idle"
        assert "Invalid request" in resp.reason

    def test_decide_with_reward(self):
        servicer = _make_servicer()
        req = _decide_request()
        req.reward = 1.0
        req.has_reward = True
        resp = servicer.Decide(req, _FakeContext())
        assert resp.action != ""

    def test_decide_no_available_actions(self):
        servicer = _make_servicer()
        req = _decide_request(actions=[])
        resp = servicer.Decide(req, _FakeContext())
        assert resp.action == "idle"


# ---------------------------------------------------------------------------
# Batch Decide
# ---------------------------------------------------------------------------

class TestGRPCBatchDecide:
    def test_batch_returns_one_per_request(self):
        servicer = _make_servicer()
        batch = BatchDecideRequest(requests=[
            _decide_request("guard1"),
            _decide_request("guard2"),
            _decide_request("merchant", actions=["sell", "haggle"]),
        ])
        resp = servicer.BatchDecide(batch, _FakeContext())
        assert len(resp.decisions) == 3

    def test_batch_preserves_agent_ids(self):
        servicer = _make_servicer()
        batch = BatchDecideRequest(requests=[
            _decide_request("alice"),
            _decide_request("bob"),
        ])
        resp = servicer.BatchDecide(batch, _FakeContext())
        ids = {d.agent_id for d in resp.decisions}
        assert "alice" in ids
        assert "bob" in ids

    def test_batch_empty_request(self):
        servicer = _make_servicer()
        batch = BatchDecideRequest(requests=[])
        resp = servicer.BatchDecide(batch, _FakeContext())
        assert resp.decisions == []

    def test_serialization_roundtrip(self):
        """Ensure the JSON serialization helpers produce valid bytes."""
        from opennpc.api.grpc_service import (
            _deserialize_decide,
            _serialize_decide_response,
        )
        from opennpc.proto.opennpc_pb2 import DecideResponse

        resp = DecideResponse(
            agent_id="x",
            action="patrol",
            confidence=0.8,
            reason="test",
            memory_update="",
            trace_json="{}",
        )
        raw = _serialize_decide_response(resp)
        data = json.loads(raw)
        assert data["agent_id"] == "x"
        assert data["action"] == "patrol"
