# Changelog

## [1.2.9] - 2026-08-17
### Fixed
- Waypoint follower: preserve waypoint orientation when creating Nav2 goals.

## [1.2.8] - 2026-08-04
### Fixed
- Updated MoveIt sample with Franka to use new gripper variant.

## [1.2.7] - 2026-07-21
### Changed
- Migrated robot asset references from `Isaac/Robots/` to `Isaac/Robots_Multiphysics/` for the new multiphysics-ready USDA assets.

## [1.2.6] - 2026-07-07
### Changed
- Use supported package-root imports for cross-extension APIs.

## [1.2.5] - 2026-06-29
### Changed
- Classify ROS 2 example modules as lifecycle-only; do not publish their Kit extension classes or UI helpers.

## [1.2.4] - 2026-06-09
### Fixed
- Fix linter errors and missing or incomplete docstrings, and update `python_api.md`.

## [1.2.3] - 2026-05-19
### Changed
- MoveIt extension and standalone samples: feed `ROS2PublishJointState` from a dedicated `IsaacReadJointState` node and use its `sensorTime` for the published timestamp

## [1.2.2] - 2026-05-14
### Changed
- Add missing type annotations and replace bare `except` clauses across the MoveIt, waypoint follower, and sample loader extensions

## [1.2.1] - 2026-04-27
### Removed
- Remove the `omni.isaac.ml_archive` dependency

## [1.2.0] - 2026-04-01
### Changed
- Removed deprecated `isaacsim.core.api` and `isaacsim.core.utils` dependencies
- Migrated to `isaacsim.core.experimental.utils`, `isaacsim.core.rendering_manager`, and `isaacsim.core.simulation_manager`
- Replaced `set_camera_view` with `ViewportManager.set_camera_view`
- Replaced `PhysicsContext` with `PhysicsScene` from `isaacsim.core.simulation_manager`

## [1.1.0] - 2026-03-17
### Changed
- Updated documentation with AI agent.

## [1.0.1] - 2026-01-30
### Changed
- Update waypoint follower action graph to use ReadPrimLocalTransform node

## [1.0.0] - 2025-10-29
### Added
- Initial release of `isaacsim.ros2.examples` extension
- Migrated ROS 2 sample code from `isaacsim.ros2.bridge` extension
