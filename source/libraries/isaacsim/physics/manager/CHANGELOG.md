# Changelog

## [Unreleased]

### Added

- Entity views whose extent is chosen by the caller report shape hints that track the buffers they
  read into, so a subsequent read sizes its allocation from the current extent.
- Initial `isaacsim.physics.manager` module providing a backend-neutral
  simulation manager with direct lifecycle, callback, scene-query, interaction,
  and benchmarking operations.
- Entity and simulation tensor views with Warp frontend support.
