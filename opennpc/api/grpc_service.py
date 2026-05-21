"""OpenNPC gRPC service.

Exposes the same decision-making pipeline as the REST API but over gRPC,
enabling lower-latency, strongly-typed engine integrations.

Requires grpcio::

    pip install 'opennpc[grpc]'

Usage
-----
    from opennpc.api.grpc_service import serve_grpc
    serve_grpc(port=50051)

Or from the CLI::

    opennpc-grpc --port 50051
"""

from __future__ import annotations

import json
import logging
from typing import Any

from opennpc.coordination import MultiAgentCoordinator
from opennpc.decision import DecisionEngine
from opennpc.experience import RuntimeExperienceLogger
from opennpc.proto.opennpc_pb2 import (
    BatchDecideRequest,
    BatchDecideResponse,
    DecideRequest,
    DecideResponse,
    HealthRequest,
    HealthResponse,
)
from opennpc.proto.opennpc_pb2_grpc import NPCServiceServicer
from opennpc.types import AgentConfig, GameState

logger = logging.getLogger(__name__)

_GRPC_AVAILABLE = False
try:
    import grpc  # type: ignore[import-untyped]
    from concurrent import futures

    _GRPC_AVAILABLE = True
except ImportError:
    grpc = None  # type: ignore[assignment]
    futures = None  # type: ignore[assignment]


def _require_grpc() -> None:
    if not _GRPC_AVAILABLE:
        raise ImportError(
            "grpcio is required for the gRPC service. "
            "Install it with: pip install 'opennpc[grpc]'"
        )


class _OpenNPCServicer(NPCServiceServicer):
    """Concrete gRPC servicer that delegates to the OpenNPC DecisionEngine."""

    def __init__(self, engine: DecisionEngine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Decide
    # ------------------------------------------------------------------

    def Decide(self, request: DecideRequest, context: Any) -> DecideResponse:
        try:
            config_dict = json.loads(request.agent_config_json)
            state_dict = json.loads(request.game_state_json)
            config = AgentConfig.from_dict(config_dict)
            state = GameState.from_dict(state_dict)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("gRPC Decide: bad request — %s", exc)
            return DecideResponse(
                action="idle",
                confidence=0.0,
                reason=f"Invalid request payload: {exc}",
            )

        reward: float | None = request.reward if request.has_reward else None
        event: str | None = request.event or None

        decision = self._engine.decide(
            config,
            state,
            available_actions=list(request.available_actions) or None,
            reward=reward,
            event=event,
        )
        return DecideResponse(
            agent_id=decision.agent_id,
            action=decision.action,
            confidence=decision.confidence,
            reason=decision.reason or "",
            memory_update=decision.memory_update or "",
            trace_json=json.dumps(decision.trace.to_dict() if decision.trace else {}),
        )

    # ------------------------------------------------------------------
    # Batch Decide
    # ------------------------------------------------------------------

    def BatchDecide(self, request: BatchDecideRequest, context: Any) -> BatchDecideResponse:
        responses: list[DecideResponse] = []
        for sub in request.requests:
            responses.append(self.Decide(sub, context))
        return BatchDecideResponse(decisions=responses)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def Health(self, request: HealthRequest, context: Any) -> HealthResponse:
        return HealthResponse(status="ok", service="opennpc")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_server(
    port: int = 50051,
    max_workers: int = 8,
    engine: DecisionEngine | None = None,
) -> "grpc.Server":
    """Create (but do not start) a gRPC server bound to ``port``.

    Parameters
    ----------
    port:
        Port to listen on.  Defaults to ``50051``.
    max_workers:
        Thread pool size for concurrent RPCs.
    engine:
        Optionally inject a pre-configured :class:`DecisionEngine`.
        If ``None`` a fresh engine is created with default settings.

    Returns
    -------
    grpc.Server
        A configured but **not yet started** server. Call ``.start()``
        followed by ``.wait_for_termination()`` to run it.
    """
    _require_grpc()
    if engine is None:
        exp_logger = RuntimeExperienceLogger()
        engine = DecisionEngine(experience_logger=exp_logger)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = _OpenNPCServicer(engine)

    # Manual method registration (avoids needing generated proto stubs at
    # install time).
    _register_servicer(server, servicer)
    server.add_insecure_port(f"[::]:{port}")
    return server


def serve_grpc(
    port: int = 50051,
    max_workers: int = 8,
    engine: DecisionEngine | None = None,
) -> None:
    """Start the gRPC server and block until interrupted."""
    _require_grpc()
    server = create_server(port=port, max_workers=max_workers, engine=engine)
    server.start()
    logger.info("OpenNPC gRPC server listening on port %d", port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5)
        logger.info("OpenNPC gRPC server stopped.")


def _register_servicer(server: "grpc.Server", servicer: _OpenNPCServicer) -> None:
    """Register all RPC methods on the server without relying on generated code."""
    generic_handler = grpc.method_service_handler  # type: ignore[attr-defined]

    def _json_decide(request_bytes: bytes, context: Any) -> bytes:
        req = _deserialize_decide(request_bytes)
        resp = servicer.Decide(req, context)
        return _serialize_decide_response(resp)

    def _json_batch(request_bytes: bytes, context: Any) -> bytes:
        req = _deserialize_batch(request_bytes)
        resp = servicer.BatchDecide(req, context)
        return _serialize_batch_response(resp)

    def _json_health(request_bytes: bytes, context: Any) -> bytes:
        resp = servicer.Health(HealthRequest(), context)
        return json.dumps({"status": resp.status, "service": resp.service}).encode()

    rpc_method_handlers = {
        "Decide": grpc.unary_unary_rpc_method_handler(
            _json_decide,
            request_deserializer=lambda b: b,
            response_serializer=lambda b: b,
        ),
        "BatchDecide": grpc.unary_unary_rpc_method_handler(
            _json_batch,
            request_deserializer=lambda b: b,
            response_serializer=lambda b: b,
        ),
        "Health": grpc.unary_unary_rpc_method_handler(
            _json_health,
            request_deserializer=lambda b: b,
            response_serializer=lambda b: b,
        ),
    }
    server.add_generic_rpc_handlers(
        (grpc.method_service_handler("opennpc.NPCService", rpc_method_handlers),)  # type: ignore[attr-defined]
    )


def _deserialize_decide(data: bytes) -> DecideRequest:
    d = json.loads(data)
    return DecideRequest(
        agent_config_json=d.get("agent_config_json", "{}"),
        game_state_json=d.get("game_state_json", "{}"),
        available_actions=d.get("available_actions", []),
        reward=float(d.get("reward", 0.0)),
        has_reward=bool(d.get("has_reward", False)),
        event=d.get("event", ""),
    )


def _deserialize_batch(data: bytes) -> BatchDecideRequest:
    d = json.loads(data)
    return BatchDecideRequest(
        requests=[
            _deserialize_decide(json.dumps(r).encode())
            for r in d.get("requests", [])
        ]
    )


def _serialize_decide_response(resp: DecideResponse) -> bytes:
    return json.dumps({
        "agent_id": resp.agent_id,
        "action": resp.action,
        "confidence": resp.confidence,
        "reason": resp.reason,
        "memory_update": resp.memory_update,
        "trace_json": resp.trace_json,
    }).encode()


def _serialize_batch_response(resp: BatchDecideResponse) -> bytes:
    return json.dumps({
        "decisions": [
            {
                "agent_id": d.agent_id,
                "action": d.action,
                "confidence": d.confidence,
                "reason": d.reason,
                "memory_update": d.memory_update,
                "trace_json": d.trace_json,
            }
            for d in resp.decisions
        ]
    }).encode()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="OpenNPC gRPC inference server")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--workers", type=int, default=8, help="Thread pool size")
    args = parser.parse_args()
    serve_grpc(port=args.port, max_workers=args.workers)


if __name__ == "__main__":
    main()
