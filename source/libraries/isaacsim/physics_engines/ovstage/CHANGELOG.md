# Changelog

All notable changes to the `isaacsim.physics_engines.ovstage` module are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Initial `isaacsim.physics_engines.ovstage` module providing an idempotent `setup()`
  facility that exposes OVStage's private Python and native runtimes without
  modifying the process-wide `pxr` package.
- Public `get_native_handle()` bridge for APIs that consume an OVStage native address.
- `lookup_stage()` recovers the Python Stage previously registered by
  `get_native_handle()` so Python backends can attach the same instance.
- `as_native_handle()` normalizes int handles and nanobind ``void*`` capsules
  (``nb_handle``) used when C++ invokes Python ``initialize`` callbacks.
