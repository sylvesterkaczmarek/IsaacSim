# Changelog

## [Unreleased]

### Added
- `isaacsim.core.experimental.utils`: initial Kit-independent release of a scoped subset of the USD utility functions
  (`stage`, `prim`, `ops`, `backend`, `bounds`, `transform`, `foundation`). The implementation no longer depends on
  `carb` or `omni.*`; the current stage is resolved through a process-level default stage id and the USD stage cache
  instead of the Kit USD context, and `usdrt` is imported lazily so the pure-OpenUSD path does not require the Fabric
  runtime. Install the `isaacsim-deprecated` distribution to use this legacy API.
