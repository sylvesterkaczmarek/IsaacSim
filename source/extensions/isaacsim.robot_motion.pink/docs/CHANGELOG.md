# Changelog

## [0.1.7] - 2026-09-01
### Fixed
- `load_pink_robot`: accept URDF exporter output that omits urdfdom-only joint limit attributes.

## [0.1.6] - 2026-08-26
### Fixed
- Support continuous revolute joints in Isaac Sim-to-Pinocchio configuration and controller-output mappings.

## [0.1.5] - 2026-07-06
### Changed
- Updates the class names in docs.

## [0.1.4] - 2026-07-03
### Changed
- Emit `carb.log_warn` on `PinkIKController` error paths when `forward()` returns `None` or `reset()` fails.

## [0.1.3] - 2026-06-09
### Fixed
- Fix linter errors and missing or incomplete docstrings, and update `python_api.md`.

## [0.1.2] - 2026-05-11
### Fixed
- Preserve default configuration and velocity safety limits when adding custom PINK IK limits.

## [0.1.1] - 2026-05-01
### Fixed
- Prevented OSQP sparse matrix conversion warnings from being emitted during PINK IK solves.

## [0.1.0] - 2026-03-25
### Added
- Initial PINK (Python Inverse Kinematics) integration with Isaac Sim motion generation API
- PinkIKController implementing BaseController for reactive differential IK
- Robot configuration loading from URDF via Pinocchio
- Transform utilities between Isaac Sim world frame and Pinocchio
- Support for FrameTask, PostureTask, and user-defined tasks, limits, and barriers
