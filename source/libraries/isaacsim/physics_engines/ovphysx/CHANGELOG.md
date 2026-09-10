# Changelog

All notable changes to the `isaacsim.physics_engines.ovphysx` module are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- The tensor factories register under a simulation name, which callers pass to `create_entity` and
  `create_simulation_view` and which entity factories resolve their handle from. Unregistering now
  removes the entity factories as well as the simulation-view factory.
- `TensorRegistry.create_entity` builds real views for all seven entity types instead of
  placeholders, resolved from the active OvPhysX simulation. It shares one implementation with
  `SimulationView.create_*_view`, so the two construction paths cannot drift apart.
- Contact and SDF views take their extent from the caller's tensors on each read, rebuilding larger
  or smaller. Nothing resizes on its own: an overflowing read still fails and the caller retries
  with a bigger buffer. Views built through `create_entity` start at 1000 contact records and one
  SDF query point. A rebuilt contact binding reports no contacts until the next step populates it.
- Views that match nothing report no support instead of coming back empty, which was
  indistinguishable from a missing registration.
- `create_entity` rejects a multi-path request for `rigid-contact` and `sdf-shape`, which take a
  single pattern, rather than silently using the first path.
- Initial `isaacsim.physics_engines.ovphysx` module providing an OvPhysX
  simulation and tensor backend with explicit activation and shutdown through
  the backend-neutral physics manager.
- Private OVStage runtime setup through `isaacsim.physics_engines.ovstage`.
- Native `disable-gravities` tensor operations for rigid bodies and articulation links.
- `drive-types` reads the live per-DOF drive type from the engine instead of authored
  USD, which removes the last pxr StageCache dependency from the tensor path. The
  values keep their documented meaning (0=none, 1=force, 2=acceleration) but now
  reflect the running simulation and arrive in tensor DOF order, so a symmetric robot
  no longer collides on leaf names. Read-only, as the engine reports rather than
  accepts it. Reads of the boolean uint8 bindings are unchanged: they promise only
  "nonzero", so they are still normalised to 0 or 1.

### Changed

- The registered simulation name is now `ovphysx` (was `OvPhysX`), matching the name callers
  already pass to `create_entity` and `create_simulation_view`. Callers comparing
  `get_simulation_name()` against the old spelling must update.

### Fixed

- Native-only builds no longer stage the OVStage Python helper package.
- Synchronous stepping now propagates OvPhysX failures before advancing physics-manager time and step accounting.
- OvPhysX activation no longer replaces or extends the process-wide `pxr` package.
- Standalone Python installs now include the complete OvPhysX runtime and plugin layout.
- `SimulationView.update_articulations_kinematic()` now refreshes OvPhysX
  articulation link transforms without advancing the simulation.
