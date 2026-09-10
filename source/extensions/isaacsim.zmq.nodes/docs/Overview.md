# isaacsim.zmq.nodes

OmniGraph publish nodes and camera helper for ZMQ Bridge streaming.

## Overview

Provides OGN nodes that serialize simulation data to Protobuf and stream it over ZMQ. Publish nodes use PUSH sockets (Isaac Sim → server); subscribe nodes use SUB sockets (server → Isaac Sim). All nodes share sockets through the registries in `isaacsim.zmq.core`.

## Nodes

### C++ OGN Publish Nodes

All publish nodes connect to a single server PULL port (default `5561`) and send `[topic, payload]` multipart messages.

| Node | Topic | Proto type |
|---|---|---|
| `OgnZMQPublishClock` | `clock` | `Clock` |
| `OgnZMQPublishImage` | `image` | `Image` |
| `OgnZMQPublishPose` | `pose` | `Pose` |
| `OgnZMQPublishJointStates` | `joint_states` | `JointStates` |

### C++ OGN Subscribe Nodes

Subscribe nodes connect to the server PUB port (default `5557`). Each SUB socket filters by its own topic so they can share a port without interfering.

| Node | Topic | Proto type |
|---|---|---|
| `OgnZMQSubscribeUpdatePrimAttribute` | `update_prim_attribute` | `UpdatePrimAttribute` |
| `OgnZMQSubscribeJointCommand` | `joint_command` | `JointCommand` |

### Python OGN Nodes

| Node | Purpose |
|---|---|
| `OgnZMQCameraHelper` | Attaches annotators to a render product and builds the publish graph |

## OgnZMQPublishImage

The `encoding` token selects content type:
- `rgba8` — 4-channel uint8 color
- `rgb8` — 3-channel uint8 color
- `32FC1` — 1-channel float32 depth

## OgnZMQCameraHelper

Attaches the RGB and/or depth `ZMQPublishImage` writers to a render product, so a single graph node streams camera images without manual writer setup.
