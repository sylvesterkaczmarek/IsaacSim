# Changelog

## [1.2.1] - 2026-07-20
### Changed
- Renamed C++ headers from `.h` to `.hpp`; update downstream include directives.

## [1.2.0] - 2026-07-14
### Changed
- The protobuf message schemas moved to the new `isaacsim.zmq.protos` extension; `isaacsim.zmq.core` is now purely the ZeroMQ socket transport. Import message classes from `isaacsim.zmq.protos` (e.g. `from isaacsim.zmq.protos import Clock`) instead of `isaacsim.zmq.core`.

### Removed
- Protobuf codegen, the bundled protobuf/pyzmq runtime, and the `Clock`/`Image`/`Pose`/`JointStates`/`JointCommand`/`UpdatePrimAttribute` re-exports (now in `isaacsim.zmq.protos`).

## [1.1.0] - 2026-07-07
### Added
- `Pose` proto carrying a prim's world-space position and orientation.

### Removed
- `CameraParams` and `Bbox2D`/`BBox2DItem`/`BBox2DInfo` protos — unused by the ZMQ bridge example.

## [1.0.2] - 2026-07-07
### Changed
- Update the generated Python API inventory.

## [1.0.1] - 2026-06-29
### Fixed
- Build: suppress spurious `-Wundef` warnings from the generated protobuf sources and stop compiling each `.proto` twice.

## [1.0.0] - 2026-04-01
### Added
- Initial release of the ZeroMQ core library for the Isaac Sim ZMQ bridge.
- Publish (PUSH) and subscribe (SUB) socket wrappers, one socket per node, with two-frame `[topic, payload]` multipart messaging.
- Protobuf message schemas (importable from the package root) and pybind11 Python bindings.
