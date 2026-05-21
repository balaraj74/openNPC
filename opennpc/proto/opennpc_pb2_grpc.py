"""Pre-generated gRPC servicer stubs for OpenNPC.

Generated from opennpc/proto/opennpc.proto.
Regenerate with::

    python -m grpc_tools.protoc \\
        -I opennpc/proto \\
        --python_out=opennpc/proto \\
        --grpc_python_out=opennpc/proto \\
        opennpc/proto/opennpc.proto
"""

from __future__ import annotations

from opennpc.proto.opennpc_pb2 import (
    BatchDecideRequest,
    BatchDecideResponse,
    DecideRequest,
    DecideResponse,
    HealthRequest,
    HealthResponse,
)


class NPCServiceServicer:
    """Base class for the NPCService gRPC server implementation."""

    def Decide(self, request: DecideRequest, context: object) -> DecideResponse:
        context.set_code(StatusCode.UNIMPLEMENTED)  # type: ignore[attr-defined]
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def BatchDecide(self, request: BatchDecideRequest, context: object) -> BatchDecideResponse:
        context.set_code(StatusCode.UNIMPLEMENTED)  # type: ignore[attr-defined]
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def Health(self, request: HealthRequest, context: object) -> HealthResponse:
        context.set_code(StatusCode.UNIMPLEMENTED)  # type: ignore[attr-defined]
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


try:
    import grpc  # type: ignore[import-untyped]
    from grpc import StatusCode

    def add_NPCServiceServicer_to_server(servicer: NPCServiceServicer, server: grpc.Server) -> None:
        """Register the servicer on the given gRPC server."""
        from grpc import unary_unary_rpc_method_handler as _handler
        from grpc import method_service_handler as _service  # noqa: F401

        rpc_method_handlers = {
            "Decide": grpc.unary_unary_rpc_method_handler(
                servicer.Decide,
                request_deserializer=_proto_decode,
                response_serializer=_proto_encode,
            ),
            "BatchDecide": grpc.unary_unary_rpc_method_handler(
                servicer.BatchDecide,
                request_deserializer=_proto_decode,
                response_serializer=_proto_encode,
            ),
            "Health": grpc.unary_unary_rpc_method_handler(
                servicer.Health,
                request_deserializer=_proto_decode,
                response_serializer=_proto_encode,
            ),
        }
        generic_handler = grpc.method_service_handler("opennpc.NPCService", rpc_method_handlers)  # type: ignore[attr-defined]
        server.add_generic_rpc_handlers((generic_handler,))

except ImportError:  # pragma: no cover - grpcio is optional
    StatusCode = None  # type: ignore[assignment, misc]

    def add_NPCServiceServicer_to_server(servicer, server) -> None:  # type: ignore[misc]
        raise ImportError("grpcio is required. Install with: pip install 'opennpc[grpc]'")


def _proto_decode(data: bytes) -> bytes:
    return data


def _proto_encode(obj: object) -> bytes:
    return b""
