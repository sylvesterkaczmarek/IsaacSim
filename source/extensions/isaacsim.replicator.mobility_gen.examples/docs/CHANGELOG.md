# Changelog

## [0.4.1] - 2026-08-31
### Fixed
- Robot classes no longer resolve the Isaac assets root at import time, so an unreachable assets server no longer makes the extension (and the MobilityGen UI) fail to load.

## [0.4.0] - 2026-08-10
### Changed
- `PolicyMultiSensorRobot.build_sensor_rig`: mount the configured `front_camera_type` before building the rig, since the H1 and Spot assets carry no cameras to bind.
- `h1.yaml`, `spot.yaml`: capture a stereo pair, renaming their sensors to `front_camera_left` and `front_camera_right`; `jetbot.yaml`'s is `front_camera`.

### Fixed
- `jetbot.yaml`, `h1.yaml`, `spot.yaml`: point each sensor rig at camera prims that exist. The old paths did not, so those robots attached no cameras and recorded no images.

## [0.3.8] - 2026-07-27
### Changed
- Run the H1 and Spot policy robots through `RobotPolicyRunner`; correct Spot's physics timestep to 0.002 seconds.

## [0.3.7] - 2026-07-21
### Changed
- Migrated robot asset references from `Isaac/Robots/` to `Isaac/Robots_Multiphysics/` for the new multiphysics-ready USDA assets.

## [0.3.6] - 2026-07-15
### Changed
- Replace the local `_join_sdf_paths` helper (and the direct `pxr` import) with `join_prim_paths` from `isaacsim.core.experimental.utils`

## [0.3.5] - 2026-07-07
### Changed
- Use supported package-root imports for cross-extension APIs.
- Migrate sensor-rig generation to the current stage utilities.

## [0.3.4] - 2026-06-09
### Fixed
- Fix linter errors and missing or incomplete docstrings, and update `python_api.md`.

## [0.3.3] - 2026-06-04
### Removed
- Stale `multiGpu` disable from `[[test]]` args; the Kit 110.1.1 multi-GPU startup crash no longer reproduces.

## [0.3.2] - 2026-04-28
### Fixed
- `RandomAccelerationScenario.step`: remove redundant `update_state()` call after `write_action()`; pose is unchanged until the next physics step.

## [0.3.1] - 2026-04-23
### Removed
- Remove the `omni.isaac.ml_archive` test dependency

## [0.3.0] - 2026-04-22
### Added
- `WheeledMultiSensorRobot` and `PolicyMultiSensorRobot` base classes for YAML-driven multi-camera robots
- `CarterMultiSensorRobot`, `JetbotMultiSensorRobot`, `H1MultiSensorRobot`, `SpotMultiSensorRobot` concrete robots
- YAML robot configs (`carter.yaml`, `jetbot.yaml`, `h1.yaml`, `spot.yaml`)
- `generate_sensor_rigs.py` script to discover sensor prims in a robot USD and scaffold `sensor_rig:` YAML blocks

## [0.2.2] - 2026-04-18
### Changed
- Added return type annotations, `from __future__ import annotations`, and imperative-mood docstrings

## [0.2.1] - 2026-03-19
### Changed
- Migrate to use `isaacsim.core.experimental.prims` (`Articulation`) in place of `isaacsim.core.prims`
- Force USD payload loading before `Articulation` initialization to ensure `ArticulationRootAPI` is visible

## [0.2.0] - 2026-03-04
### Changed
- Added Overview.md, python_api.md and updated docstrings

## [0.1.12] - 2026-02-06
### Changed
- Change occupancy map radius for Carter Robot

## [0.1.11] - 2025-12-10
### Changed
- Fix USD path for placement of front camera on Carter with latest USD asset
- Fix issue where H1 and Spot policies command must be provided as torch tensor

## [0.1.10] - 2025-10-27
### Changed
- Make omni.isaac.ml_archive an explicit test dependency

## [0.1.9] - 2025-09-30
### Fixed
- Add dialog message for incorrect occupancy map paths

## [0.1.8] - 2025-07-07
### Fixed
- Correctly enable omni.kit.loop-isaac in test dependency (fixes issue from 0.1.7)

## [0.1.7] - 2025-07-03
### Changed
- Make omni.kit.loop-isaac an explicit test dependency

## [0.1.6] - 2025-06-25
### Changed
- Add --reset-user to test args
- Fix SDF paths on windows

## [0.1.5] - 2025-06-12
### Changed
- Fix broken jetbot and nova carter asset links

## [0.1.4] - 2025-06-10
### Changed
- Use default asset root for all assets

## [0.1.3] - 2025-05-31
### Changed
- Use default nucleus server for all tests

## [0.1.2] - 2025-05-19
### Changed
- Update copyright and license to apache v2.0

## [0.1.1] - 2025-05-10
### Changed
- Enable FSD in test settings

## [0.1.0] - 2025-04-28
### Added
- Initial version of MobilityGen Examples
