# Changelog

## [1.0.0] - 2026-07-30
### Changed
- Deprecated the extension in favor of `isaacsim.robot_motion.examples`.
- Moved the extension to the deprecated extension staging area.
- Reduced the extension to import shims for equivalent APIs and actionable errors for removed APIs.
- Redirected legacy path-planning users to the cuMotion Graph Planner example.
- Removed legacy Robotics Examples browser registration while retaining Python compatibility modules.

## [0.2.6] - 2026-07-22
### Added
- Add reusable Franka behavior-tree nodes and a two-robot cooperative stacking sample with physical grasping and reactive moved- or dropped-cube recovery.

### Fixed
- Handle transient prim failures without claiming cubes, reset recovery state correctly, and honor the configured stack completion threshold.

## [0.2.5] - 2026-07-21
### Changed
- Migrated robot asset references from `Isaac/Robots/` to `Isaac/Robots_Multiphysics/` for the new multiphysics-ready USDA assets.

## [0.2.4] - 2026-07-09
### Added
- Add a shared UR10 palletizing module for standalone and documentation examples.

## [0.2.3] - 2026-07-07
### Changed
- Use supported package-root imports for cross-extension APIs.

## [0.2.2] - 2026-06-29
### Changed
- Stop publishing Kit lifecycle classes from interactive example modules.

## [0.2.1] - 2026-06-09
### Fixed
- Fix linter errors and missing or incomplete docstrings, and update `python_api.md`.

## [0.2.0] - 2026-05-27
### Fixed
- Improve UR10 bin filling motion stability by allowing the lift phase to settle before moving the bin under the filling spout.
- Updated bin filling example to use new motion generation/CuMotion APIs

## [0.1.3] - 2026-05-18
### Fixed
- Improve Robo Factory stacking test reliability by waiting for stacking completion and allowing more time for the Franka move and release phases.

## [0.1.2] - 2026-05-11
### Fixed
- Move default cube placement position to [0.4, 0.2, 0.7] for follow_target.py example so all IK methods can reach default pose.

## [0.1.1] - 2026-05-01
### Fixed
- Move default cube placement target position to [0.0, 0.5, 0.14] so the placed cube is visible and not blocked by the robot arm.

## [0.1.0] - 2026-04-06
### Added
- Migrated experimental manipulator examples from isaacsim.robot.manipulators.examples.
- FrankaExperimental, FrankaPickPlace, and Franka Stacking examples.
- UR10Experimental and UR10 FollowTarget examples.
- Interactive pick-place and follow-target UI extensions.
