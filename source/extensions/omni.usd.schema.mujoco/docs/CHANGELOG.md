# Changelog

## [1.1.4] - 2026-08-28
### Fixed
- Check the global plugin registry after MuJoCo schema registration.

## [1.1.3] - 2026-08-11
### Fixed
- Ship the `mjcPhysics` schema resources inside the extension (copy instead of linking to `target-deps`) so the plugin no longer goes missing when the Newton prebundle is refreshed or deployed without `target-deps`.
- Log an error when the `mjcPhysics` plugin fails to register instead of failing silently.
- Declare the Newton schema dependency required by composed MuJoCo APIs.

### Added
- Register the schema tests as an extension test module with a `startup` test asserting `mjcPhysics`/`MjcSceneAPI` availability.

## [1.1.2] - 2026-07-07
### Changed
- Update the generated Python API inventory.

## [1.1.1] - 2026-06-09
### Fixed
- Fix linter errors and missing or incomplete docstrings, and update `python_api.md`.

## [1.1.0] - 2026-03-04
### Changed
- Added Overview.md, python_api.md and updated docstrings

## [1.0.0] - 2026-02-03
### Changed
- Version bump.
