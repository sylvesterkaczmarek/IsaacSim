# Changelog

## [1.1.1] - 2026-07-20
### Changed
- Renamed C++ headers from `.h` to `.hpp`; update downstream include directives.

## [1.1.0] - 2026-07-14
### Changed
- Depend on the new `isaacsim.zmq.protos` extension for the generated protobuf message library and Python schemas (previously hosted by `isaacsim.zmq.core`).

## [1.0.0] - 2026-07-07
### Added
- Initial release of the ZMQ bridge OmniGraph nodes.
- Publish nodes (PUSH): `OgnZMQPublishClock`, `OgnZMQPublishImage` (raw bytes or zero-copy CUDA IPC), `OgnZMQPublishPose`, `OgnZMQPublishJointStates`.
- Subscribe nodes (SUB, filtered by topic so they can share a port): `OgnZMQSubscribeJointCommand`, `OgnZMQSubscribeUpdatePrimAttribute`.
- `OgnZMQCameraHelper` Python node that attaches the RGB/depth image writers to a render product.
