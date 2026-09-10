# ZMQ Protos [isaacsim.zmq.protos]

The `isaacsim.zmq.protos` extension owns the protobuf message schemas for the Isaac Sim ZMQ
bridge. The `.proto` files under `proto/` are compiled by `protoc` at build time into:

- **C++** `*.pb.h` / `*.pb.cc`, compiled into `libisaacsim.zmq.protos.so`. The `isaacsim.zmq.nodes`
  OmniGraph nodes include these headers and link this library to serialize/parse messages in C++.
- **Python** `*_pb2.py` stubs, shipped alongside this package and re-exported from
  `isaacsim.zmq.protos` (e.g. `from isaacsim.zmq.protos import Clock, Image, Pose`). The bundled
  `protobuf` runtime makes them importable on their own — including from the standalone
  `zmq_bridge` tool.

Splitting the schemas into their own extension keeps `isaacsim.zmq.core` free of protobuf (it is
purely the ZeroMQ socket transport) and gives producers and consumers a single, shared source of
truth for the wire format.

## Messages

`Clock`, `Image` (with `GpuIpcImage` / `GpuIpcArray` for the CUDA-IPC transport), `Pose`,
`JointStates`, `JointCommand`, and `UpdatePrimAttribute`.

## Extending

Add a new `.proto` under `proto/`, register it in `premake5.lua`'s `files { ... }` list, and
rebuild — this regenerates both the C++ and Python bindings.
