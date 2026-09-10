# Changelog

## [1.0.1] - 2026-08-24
### Changed
- Build, generate, test and package the schema through the standalone library CMake build, following the `Migrate USD schemas to source libraries` restructure on `develop`.
- Import `Usd` explicitly rather than relying on `pxr.Usd` being present. It is inside Kit, but not in a bare USD runtime, so the installed package failed to import there.

## [1.0.0] - 2026-08-07
### Added
- Initial release. `IsaacVirtualGantry` moved out of `isaacsim.robot.schema` into its own extension so the robot schema stays limited to describing robots. Loads at `order = -100` so the prim type is registered before the extensions that depend on it.
