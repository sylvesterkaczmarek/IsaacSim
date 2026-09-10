# Changelog

## [0.1.7] - 2026-08-11
### Changed
- Show how to enable this extension in the quick start example, since its nodes are unavailable until it is enabled.

## [0.1.6] - 2026-07-20
### Changed
- Renamed C++ headers from `.h` to `.hpp`; update downstream include directives.

## [0.1.5] - 2026-07-07
### Changed
- Use supported package-root imports for cross-extension APIs.

## [0.1.4] - 2026-06-26
### Changed
- Expose `register_scalar_colored_point_cloud_writer` as public API while hiding internal constants and the Kit lifecycle extension class.

## [0.1.3] - 2026-06-14
### Added
- `register_scalar_colored_point_cloud_writer` utility to color RTX point clouds by a scalar field (distance, intensity, ...).

## [0.1.2] - 2026-06-11
### Fixed

- `IsaacExtractRTXSensorPointCloud` properly passes-through Cartesian point clouds

## [0.1.1] - 2026-06-09
### Fixed
- Fix linter errors and missing or incomplete docstrings, and update `python_api.md`.

## [0.1.0] - 2026-04-08
### Added

- `IsaacExtractRTXSensorPointCloud` OmniGraph node for Cartesian point cloud extraction from GenericModelOutput.
- `IsaacExtractRTXSensorPointCloud` Replicator annotator.
- `RtxSensorDebugDrawPointCloud` Replicator writer for viewport debug draw visualisation.
