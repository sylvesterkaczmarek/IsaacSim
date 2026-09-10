# Changelog

## [Unreleased]

### Added
- Add a thin C++ wrapper over OpenUSD exposing stage lifecycle, prim authoring, schema queries, variant selection, and
  attribute I/O through a stable ABI based on opaque `int64_t` stage handles and standard C++ types.

### Fixed
- Register the packaged OpenUSD DLL directory before loading dependent Isaac Sim bindings on Windows.
