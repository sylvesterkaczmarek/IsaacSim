# Changelog

## [Unreleased]

### Added
- `stage_utils`: add explicit-stage helpers for querying and authoring time-code ranges.

### Changed
- Package `isaacsim.asset.importer.utils` in the independently built `isaacsim-asset` wheel.
- Include the latest PhysX-to-MuJoCo/Newton conversion helpers from the extension API.

### Fixed
- Register Newton and MuJoCo schema plugins before standalone URDF conversion constructs their schema definitions.
