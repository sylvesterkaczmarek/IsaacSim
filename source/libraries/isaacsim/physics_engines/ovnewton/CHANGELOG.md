# Changelog

All notable changes to the `isaacsim.physics_engines.ovnewton` module are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed

- Initialize Newton from a caller-owned ovstage via PyPI ``ovnewton.attach_ovstage``
  instead of ``newton.ModelBuilder.add_usd`` / USD StageCache lookup.
- After attach, rewrite ``model.articulation_label`` from synthetic names to USD
  ``ArticulationRootAPI`` paths so tensor ``create_articulation_view`` patterns match.
- Drop unused ``_newton_*`` compatibility aliases; ``release_physics_objects`` clears
  the live binding/model state. Build failures leave the backend uninitialized so
  ``start_simulation`` can retry.
- Register ``has_attached_stage`` so a retained ovstage after a failed model build
  reports ``eFailedDirty`` and keeps the caller-owned stage alive.

### Added

- `register()` accepts a `simulation_name`, registering the simulation and its tensor factories under
  it. Distinct names let several Newton simulations coexist: `create_entity` and
  `create_simulation_view` take that name, and entity factories resolve their stage by exact match.
  Omitting it keeps the previous single-simulation behaviour. `unregister()` removes the factories a
  named simulation added.
- `TensorRegistry.create_entity` builds real views for `articulation`, `rigid-body` and
  `rigid-contact`, resolved from the active Newton simulation, sharing one implementation with
  `SimulationView.create_*_view`.
- `sdf-shape` and the three deformable types, which Newton does not implement, now report no
  support instead of returning empty views, as do views that match nothing.
- Initial `isaacsim.physics_engines.ovnewton` module providing a Kit-independent
  Newton simulation backend with partial tensor-view support.

### Changed

- The registered simulation name is now `newton` (was `Newton`), matching the name callers
  already pass to `create_entity` and `create_simulation_view`. Callers comparing
  `get_simulation_name()` against the old spelling must update.
