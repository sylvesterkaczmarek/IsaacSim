# Overview

The ZMQ Bridge extension enables high-performance, low-latency communication between Isaac Sim and external processes using ZeroMQ (ZMQ) and Protocol Buffers (protobuf). It provides a complete integration layer for streaming simulation data — including camera images, bounding boxes, camera parameters, and clock signals — out of Isaac Sim via OmniGraph nodes and Action graphs. The bridge makes it straightforward to connect Isaac Sim simulations with external perception pipelines, machine learning systems, and robotic control loops.

**Important**: ZMQ publisher nodes are only active during simulation playback (when play is pressed). ZMQ sockets are created when playback starts and closed when it stops.

## Functionality

The extension enables unidirectional and bidirectional data exchange between Isaac Sim and external processes. Data flows out of Isaac Sim through OmniGraph publish nodes, which serialize simulation state into protobuf messages and push them over ZMQ PUSH sockets. An optional control channel allows external processes to send commands back to Isaac Sim over a ZMQ PULL socket. Communication is multiplexed over a single port using topic-prefixed multipart messages, so a single connection carries clock, image, bounding box, and camera parameter streams simultaneously.

### OmniGraph Integration

The bridge operates through OmniGraph publish nodes that handle ZMQ communication. Each node serializes one data type (clock, image, bounding box, or camera parameters) into a protobuf message and pushes it out on a configured ip:port. Nodes can be assembled into Action graphs driven by `OnPlaybackTick` to publish at simulation rate, or by `OnImpulseEvent` for manual triggering. Multiple nodes can target the same ip:port — each node owns its own PUSH socket, which (being a `connect` socket) coexists fine with others on the same endpoint.

### Activation Behavior

ZMQ publisher nodes remain dormant until simulation playback begins. When play is pressed, all configured publish nodes activate and begin pushing data. This ensures ZMQ traffic only occurs during active simulation, preventing spurious messages during scene setup or editing. When simulation is stopped, sockets are released.

## Integration

The extension consolidates the ZMQ Bridge stack into a single dependency:

- **isaacsim.zmq.core** provides the C++ ZMQ socket layer (`ZmqPublishSocket`, `ZmqSubscribeSocket`), protobuf schemas (`.proto` files), and generated C++ and Python bindings
- **isaacsim.zmq.nodes** supplies the OmniGraph node implementations (`ZMQPublishClock`, `ZMQPublishImage`, `ZMQPublishBbox2D`, `ZMQPublishCameraParams`) that perform actual ZMQ publishing

Enable this extension to activate the full ZMQ Bridge stack:

```toml
[dependencies]
"isaacsim.zmq.bridge" = {}
```

## Considerations

### External Process Setup

The external process receiving ZMQ messages must bind a PULL socket on the same port that Isaac Sim is configured to push to. A standalone example, `source/tools/zmq_bridge/zmq_server.py`, demonstrates a full receive-display-and-teleop workflow (streams a robot camera and drives the robot over the bridge).

### Protocol Buffers

The `.proto` schema files live in `isaacsim.zmq.core/proto/`. Python bindings are generated at build time and placed alongside the extension package. External processes that need the same protobuf definitions can regenerate them using the `generate_protos.py` script included with `isaacsim.zmq.bridge.examples`.
