# Changelog

## [0.1.6] - 2026-09-01
### Fixed
- Use the Jazzy controller-manager trigger clock to prevent negative time deltas during controller switching.

## [0.1.5] - 2026-08-24
### Fixed
- Replace non-finite effort and velocity limits in synthesized URDFs so controller initialization succeeds.

## [0.1.4] - 2026-08-11
### Fixed
- Fill effort and velocity limits for continuous joints in synthesized URDF.

## [0.1.3] - 2026-07-21
### Added

- Run the ROS 2 Control smoke test from the extracted standalone package during native Python testing.

### Fixed

- Load each ROS 2 Control backend only from its distribution-specific `lib` directory, preventing duplicate plugin registration after archive extraction.
- Reject unsupported ROS 2 distributions with an explicit diagnostic instead of searching for a nonexistent C++ backend.

## [0.1.2] - 2026-07-20
### Changed
- Renamed C++ headers from `.h` to `.hpp`; update downstream include directives.

## [0.1.1] - 2026-06-30
### Fixed

- Propagate controller YAML parameters to dynamically loaded controllers on ROS 2 Humble.
- Stop the idle controller executor when no managers remain, avoiding a race with ROS shutdown.
- Avoid a teardown-side update that can deadlock controller switching.

## [0.1.0] - 2026-06-12
### Added

- In-process ros2_control `ControllerManager`, driven from the physics post-step callback.
- `IsaacSimSystem` `hardware_interface::SystemInterface` exposing joint position/velocity/effort command and state interfaces via the physics tensor API; engine-agnostic (PhysX and Newton).
- `Ros2ControlManager` Python API with a per-Play lifecycle; `isaacsim.ros2.nodes` provides the corresponding `ROS2ControlManager` OmniGraph node.
- Live URDF synthesis from the USD stage, latched `/robot_description`, and configurable `use_sim_time`.
- Multi-robot support via namespaced managers; a duplicate `targetPrim` is a hard error.
- IMU and force-torque sensor state interfaces that feed the upstream `imu_sensor_broadcaster` and `force_torque_sensor_broadcaster`.
- Engine-agnostic mimic joints.
- ROS 2 Humble and Jazzy.
