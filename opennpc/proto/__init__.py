"""OpenNPC gRPC proto package.

Contains the service definition (``opennpc.proto``), pre-generated message
stubs (``opennpc_pb2``), and servicer base classes (``opennpc_pb2_grpc``).

Regenerate stubs from the `.proto` source::

    python -m grpc_tools.protoc \\
        -I opennpc/proto \\
        --python_out=opennpc/proto \\
        --grpc_python_out=opennpc/proto \\
        opennpc/proto/opennpc.proto
"""
