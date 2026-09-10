# Changelog

## [0.2.0] - 2026-07-23
### Changed

- Move the interactive examples into **Robotics Examples > Motion Generation > PINK**.

## [0.1.6] - 2026-07-21
### Changed
- Migrated robot asset references from `Isaac/Robots/` to `Isaac/Robots_Multiphysics/` for the new multiphysics-ready USDA assets.

## [0.1.5] - 2026-07-09
### Fixed
- Expose each example's `IExt` entry point from its Python module declared in `extension.toml`.

## [0.1.4] - 2026-07-07
### Changed
- Use supported package-root imports for cross-extension APIs.

## [0.1.3] - 2026-06-29
### Changed
- Classify the example UI builders as internal lifecycle implementation details; they remain import-compatible but are no longer published Python API.

## [0.1.2] - 2026-06-09
### Fixed
- Fix linter errors and missing or incomplete docstrings, and update `python_api.md`.

## [0.1.1] - 2026-05-11
### Changed
- Updated PINK examples to target the Franka gripper fingertip midpoint and initialize from a bent reaching posture.
- Replaced the default stage ground plane with the flat grid environment.

## [0.1.0] - 2026-04-02
### Added
- IK Controller example demonstrating reactive end-effector tracking with PINK
- Multi-Task example demonstrating weighted frame + posture + damping tasks
