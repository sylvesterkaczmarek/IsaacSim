# Changelog

All notable changes to the `isaacsim.physics.registration` module are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed

- Python ``SimulationFns.initialize`` callbacks receive the ovstage address as
  ``int`` (matching ``physics_manager.initialize``) instead of a nanobind
  ``void*`` capsule.

### Added

- `get_active_simulation_id(name)` returns the active simulation registered under a name, matched
  exactly, and raises when more than one active simulation carries that name.
- `TensorRegistry` keeps engines and simulations apart. `register_engine` declares which engines exist
  and `list_engines()` reports those, for the life of the process: an engine describes what can be
  simulated, so it stays listed once its simulations end. `list_simulations()` reports the names that
  have registered factories, which is what `create_entity` and `create_simulation_view` accept.
- Initial `isaacsim.physics.registration` module providing direct free-function
  APIs for registering physics simulation backends and tensor factories.
- Shared simulation, event, scene-query, interaction, benchmark, and tensor
  vocabulary types.
