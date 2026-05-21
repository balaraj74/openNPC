"""Pre-generated gRPC message stubs for OpenNPC.

Generated from opennpc/proto/opennpc.proto.
Regenerate with::

    python -m grpc_tools.protoc \\
        -I opennpc/proto \\
        --python_out=opennpc/proto \\
        --grpc_python_out=opennpc/proto \\
        opennpc/proto/opennpc.proto

This file is intentionally kept as a minimal hand-written stub so that the
package installs without grpcio-tools. Only the message attributes used by
``opennpc.api.grpc_service`` are defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DecideRequest:
    agent_config_json: str = ""
    game_state_json: str = ""
    available_actions: list[str] = field(default_factory=list)
    reward: float = 0.0
    has_reward: bool = False
    event: str = ""


@dataclass
class DecideResponse:
    agent_id: str = ""
    action: str = ""
    confidence: float = 0.0
    reason: str = ""
    memory_update: str = ""
    trace_json: str = ""


@dataclass
class BatchDecideRequest:
    requests: list[DecideRequest] = field(default_factory=list)


@dataclass
class BatchDecideResponse:
    decisions: list[DecideResponse] = field(default_factory=list)


@dataclass
class HealthRequest:
    pass


@dataclass
class HealthResponse:
    status: str = ""
    service: str = ""
