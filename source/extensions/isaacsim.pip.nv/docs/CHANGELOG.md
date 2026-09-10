# Changelog

## [1.0.2] - 2026-07-08
### Changed
- Build this extension on Linux only. The bundled CUDA runtime wheels are Linux-only, so it is no longer registered or tested on Windows.

## [1.0.1] - 2026-07-06
### Changed
- Classify the CUDA pip archive as lifecycle-only; its Kit extension class is not public Python API.

## [1.0.0] - 2026-06-26
### Added
- Initial version of NVIDIA CUDA Pip Archive. Bundles the `nvidia-*-cu12` CUDA runtime wheels and `cuda-bindings` split out from `pip_ml.toml`.
