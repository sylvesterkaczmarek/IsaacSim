# Changelog

## [1.0.1] - 2026-07-21
### Changed
- Move policy-specific CUDA library preloading to `isaacsim.robot.policy.examples`.

## [1.0.0] - 2026-06-15
### Added
- Added an ONNX pip archive extension for policy inference examples, initially bundling ONNX Runtime.
- Added CPU inference support on Linux AArch64 and reused `isaacsim.pip.nv` for Linux GPU dependencies.
- Added internal CUDA library preloading through the shared NVIDIA archive.
