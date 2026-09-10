# isaacsim.zmq.core

ZeroMQ socket infrastructure library for the Isaac Sim ZMQ Bridge. The protobuf message schemas
live in the separate `isaacsim.zmq.protos` extension.

> **Note:** This extension is provided for **demonstration purposes only**. It illustrates
> one way to stream data between Isaac Sim and an external application over ZeroMQ; it is
> **not** intended to be a complete, production-grade, or fully featured communication
> bridge.

## Overview

Provides the shared C++ backend and Python bindings for ZMQ-based simulation data streaming:

- **ZmqContext** — singleton `zmq::context_t` wrapper
- **ZmqPublishSocket** — non-blocking PUSH socket (HWM=1, LINGER=0) with multipart send `[topic, payload]`; each OGN publish node owns one (these are `connect` sockets, so many to one endpoint is fine)
- **ZmqSubscribeSocket** — SUB socket that connects to a remote PUB endpoint, subscribes to a single topic, and receives two-frame multipart messages `[topic, payload]` non-blocking; each OGN subscribe node owns one
- **Python bindings** — `ZmqPublishSocket`, `ZmqSubscribeSocket` exposed via pybind11

The protobuf message schemas (`Clock`, `Image`, `Pose`, `JointStates`, `UpdatePrimAttribute`, `JointCommand`) live in the separate `isaacsim.zmq.protos` extension.

## Topic Strings

All messages are two-frame multipart: `[topic, serialized_proto]`. The topic is the first frame — an opaque string used purely for routing/addressing (ZMQ SUB filters by byte-prefix); it is **not** a message-type tag, since each publisher and subscriber is already bound to one message type.

The topic is **caller-supplied**, not a fixed constant. The OGN publish/subscribe nodes expose a `topic` input that defaults to the message-type name below, so a single source works with no configuration:

| Direction | Default topic |
|---|---|
| Clock (Isaac → server) | `clock` |
| Image (Isaac → server) | `image` |
| Pose (Isaac → server) | `pose` |
| JointStates (Isaac → server) | `joint_states` |
| UpdatePrimAttribute (server → Isaac) | `update_prim_attribute` |
| JointCommand (server → Isaac) | `joint_command` |

**Multiple sources (e.g. multi-robot):** give each source a distinct topic so a consumer can tell them apart. A `<namespace>/<type>` prefix (e.g. `robot_0/image`, `robot_1/image`) composes naturally with ZMQ's prefix-based SUB filtering — a consumer can subscribe to `robot_0/` for one robot's whole stream, or `robot_0/image` for a single one.

## Usage

```python
from isaacsim.zmq.core import ZmqPublishSocket, ZmqSubscribeSocket

# Publish (PUSH) — each node owns its socket; topic is whatever the caller chooses
sock = ZmqPublishSocket("localhost", 5561)
sock.send_multipart("clock", serialized_bytes)        # or e.g. "robot_0/clock"

# Subscribe (SUB)
sub = ZmqSubscribeSocket("localhost", 5557, "update_prim_attribute")
payload = sub.try_recv()  # bytes, or None if nothing is available
```
