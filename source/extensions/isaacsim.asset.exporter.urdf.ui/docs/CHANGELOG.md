# Changelog

## [1.0.6] - 2026-08-14
### Fixed
- Preserve temporary PhysX-computed mass properties until URDF conversion completes.

## [1.0.5] - 2026-07-21
### Changed
- Migrated robot asset references from `Isaac/Robots/` to `Isaac/Robots_Multiphysics/` for the new multiphysics-ready USDA assets.

## [1.0.4] - 2026-07-07
### Changed
- Use supported package-root imports for cross-extension APIs.

## [1.0.3] - 2026-06-26
### Changed
- Hide UI implementation classes from the generated public Python API.

## [1.0.2] - 2026-06-09
### Fixed
- Fix linter errors and missing or incomplete docstrings, and update `python_api.md`.

## [1.0.1] - 2026-04-14
### Fixed
- Fixed issue with UR10e test asset path

## [1.0.0] - 2026-03-23
### Added
- Initial version: UI split from isaacsim.asset.exporter.urdf
