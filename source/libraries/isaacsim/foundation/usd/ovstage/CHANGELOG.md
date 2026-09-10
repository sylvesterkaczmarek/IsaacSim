# Changelog

## [Unreleased]

### Fixed
- Seal populated stages before exposing them so OVPhysX 0.5.11 observes imported physics data.

### Added
- Add a thin C++ wrapper over ovstage exposing stage lifecycle, prim authoring, schema queries, and
  attribute I/O through a stable ABI based on opaque `int64_t` stage handles and standard C++ types.
