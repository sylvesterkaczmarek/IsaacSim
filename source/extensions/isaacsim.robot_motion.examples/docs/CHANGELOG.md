# Changelog

## [0.2.6] - 2026-08-26
### Fixed

- Register the legacy `isaacsim.robot.experimental.manipulators.examples` node library as hidden so the Franka behavior-tree nodes are no longer listed twice in the Node Library panel.

## [0.2.5] - 2026-08-12
### Fixed

- Hold the Franka grasp and drop poses briefly before closing or opening the gripper to prevent unstable cube contacts.

## [0.2.4] - 2026-08-10
### Changed

- Use cuMotion RMPflow for the Franka behavior-tree motion nodes and share one controller per robot.

### Fixed

- Keep retrying non-converging behavior-tree cube targets, warn every five seconds, and resume after convergence.

## [0.2.3] - 2026-08-10
### Fixed

- Fix UR10 palletizing restarts after resetting simulations with spawned bins.

## [0.2.2] - 2026-08-06
### Fixed

- Allow `PickPlaceTask` to recover after a controller `FAILED` phase when `reset()` sets `_needs_reset`, so interactive examples can step again after Reset.

## [0.2.1] - 2026-08-04
### Fixed

- Fix Franka behavior-tree orientation convergence and transient empty pose handling.

## [0.2.0] - 2026-07-29
### Added

- Added reusable Franka behavior-tree nodes with cooperative stacking and recovery.
- Added interactive manipulation examples, UR10 palletizing workflows, and replay data.

### Changed

- Consolidated graph-based path planning on the existing cuMotion Graph Planner example.

### Fixed

- Keep the UR10 palletizing conveyor kinematic, restore its travel direction, and add safer pickup timing and direct-place clearance.

## [0.1.1] - 2026-07-28
### Changed

- Tightened controller validation and adopted keyword-only surface-gripper construction.
- Simplified articulation initialization using experimental API broadcasting.

## [0.1.0] - 2026-07-24
### Added

- Added named Franka and UR10 joint, tool-frame, measured-link, and grasp-offset configuration.
- Added a shared plain-`Articulation` scenario with measured-site `RobotState` exchange.
- Added feedback-driven pick-and-place and observable gripper controllers composed with RMPFlow.
- Added robot-independent standalone manipulation examples for Franka and UR10.
