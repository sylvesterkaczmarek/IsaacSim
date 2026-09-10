# Changelog

## [1.9.1] - 2026-08-07
### Added
- Added a dependency on `omni.usd.schema.mujoco`
- Added `mjcjoint` api to joints
- PhysX-to-MJC conversion now completes joints that already have `MjcJointAPI` applied (previously skipped): it authors the `mjc:*` attributes and (re)creates the `MjcActuator`, keeping re-runs idempotent.
- `MjcActuator` names no longer collide when joints share a leaf name across the hierarchy (for example `/robot/arm/joint1` and `/robot/leg/joint1`). The colliding actuator is now given a unique full-path-based name with a warning instead of being silently dropped or overwritten.

### Changed
- Load `isaacsim.asset.importer.utils` from its own projected prebundle of the shared `isaacsim-asset` wheel.

### Removed
- Removed deprecated `urdf:limit:velocity` to `physxJoint:maxJointVelocity` conversion. The upstream converter now authors `newton:velocityLimit` directly.

## [1.9.0] - 2026-08-06
### Changed
- Support the Scene Optimizer Core 110.2.1 update when merging mesh
- Pip wheel now depends on usd-optimize=1.1, kit free
- Update import names from omni.scene.optimizer to usd_optimize

## [1.8.4] - 2026-07-28
### Fixed
- Preserve inherited collision purposes during mesh merging and use world-anchoring joints (any physics joint type) as fixed-base articulation roots.
- Fixed-base detection treats a joint as a world anchor when `body1` is an articulation rigid body and `body0` is unset or non-rigid; joints with `body1` empty are ignored.

## [1.8.3] - 2026-07-21
### Changed
- Migrated robot asset references from `Isaac/Robots/` to `Isaac/Robots_Multiphysics/` for the new multiphysics-ready USDA assets.

## [1.8.2] - 2026-07-07
### Changed
- Update the generated Python API inventory.

## [1.8.1] - 2026-06-09
### Fixed
- Fix linter errors and missing or incomplete docstrings, and update `python_api.md`.

## [1.8.0] - 2026-05-14
### Added
- `parse_robot_name` helper in `importer_utils` for validating input file extension and parse name
- Collapse multiple single-axis joints sharing the same body pair into a single PhysX D6 joint and remove the redundant joints from the PhysX variant
- `apply_floating_base` helper in `asset_utils` that removes any `FixedJoint` anchoring the articulation root link to the world (the inverse of `apply_fix_base`), so importers can deliberately convert a fixed-base asset into a floating-base one.

### Fixed
- Top-level package `__init__.py` now re-exports the public helpers

### Changed
- Added `__all__` to `asset_utils` and added `create_robot_schema` / `ROBOT_TYPE_TOKENS` to `importer_utils.__all__` so the package re-exports cover every name documented in `config/python_api.md`.

## [1.7.0] - 2026-05-07
### Changed
- `enable_self_collision` no longer applies `PhysxArticulationAPI` or writes `physxArticulation:enabledSelfCollisions`. Articulation roots are now marked with the standard `UsdPhysics.ArticulationRootAPI` plus `NewtonArticulationRootAPI` (`newton:selfCollisionEnabled`); the runtime consumes the Newton articulation schema directly. The `PhysxSchema.ARTICULATION_API` and `PhysxAttr.ARTICULATION_SELF_COLLISION` enums remain available for code that still reads existing PhysX articulation data.

### Removed
- `create_physx_mimic_joint` helper from `importer_utils`. The URDF and MJCF importers now leave joints as `NewtonMimicAPI` and the runtime consumes the Newton mimic schema directly, so an equivalent `PhysxMimicJointAPI` is no longer authored. The `PhysxMimicAttr`, `PhysxMimicRel`, and `PhysxSchema.MIMIC_JOINT_API` enums remain available for code that still reads existing PhysX mimic data (e.g. the URDF exporter).

### Changed
- `apply_link_density()` in `asset_utils` now applies `MassAPI` to any rigid-body link that doesn't already have it before authoring the density value, so callers no longer need to invoke `add_rigid_body_schemas()` first as a separate pass. Density is still skipped on links that already have an authored positive `physics:mass`, and prims without `RigidBodyAPI` (or `PhysicsRigidBodyAPI`) are never touched.
- `importer_utils.add_rigid_body_schemas()` is retained for backwards compatibility, will be removed in the future.

## [1.6.0] - 2026-04-20
### Changed
- Added robot schema related helpers
- Added utilities to preset joint density and joint dynamics
- Updated merge mesh utils to use the scene optimzier apis directly
- Added file conflict collision checks

## [1.5.0] - 2026-04-14
### Changed
- Replace `pxr.PhysxSchema` typed-API usage with direct `prim.ApplyAPI()` / `prim.CreateAttribute()` calls across all importer utility modules
- Add utils for physx types

## [1.4.0] - 2026-04-14
### Changed
- Defer `isaacsim.asset.transformer` import to `run_asset_transformer_profile()` function body to remove module-level cross-extension dependency

## [1.3.0] - 2026-04-09
### Changed
- Update `run_asset_transformer_profile` to use renamed `input_stage` parameter

## [1.2.0] - 2026-04-08
### Changed
- Improve Python API documentation (`config/python_api.md` and/or module docstrings).

## [1.1.2] - 2026-04-02
### Changed
- Moved compare usd util funciton isaacsim.test.util

## [1.1.1] - 2026-03-05
### Changed
- Fix api and docs syntax issues

## [1.1.0] - 2026-02-26
### Added
- Utilities to convert between mjc, physx, urdf attributes
- Utilities to add physx mimic joints based on Newton Minic API

## [1.0.0] - 2026-01-16
### Added
- Initial release of shared asset importer utilities.
- Collision-from-visuals helpers.
- Mesh merge helpers.
- Self-collision utilities.
- Stage utilities
- Asset structure profile runner.
