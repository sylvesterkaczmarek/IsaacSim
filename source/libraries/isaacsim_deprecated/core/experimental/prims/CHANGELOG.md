# Changelog

## [Unreleased]

### Added
- `isaacsim.core.experimental.prims`: Kit-independent deprecated compatibility layer that preserves the
  `isaacsim.core.experimental.prims` import path. `Prim` aliases `isaacsim.foundation.objects`; `XformPrim` and
  `GeomPrim` preserve the legacy non-destructive construction defaults while using Foundation implementations.
  Install the `isaacsim-deprecated` distribution to use this legacy API.
