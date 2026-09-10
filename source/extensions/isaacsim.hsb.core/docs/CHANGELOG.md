# Changelog

## [1.2.0] - 2026-07-22
### Added
- VB1940 calibration EEPROM served over I²C from the shared `HSBEmulator` at peripheral `0x51`, so hololink receivers get real intrinsics + stereo extrinsics via the standard code path (no sim-aware fallbacks in the receiver).
- `CameraCalibration` struct and `HSBSender::setCalibration()` API to stage values from the OGN layer; applied at `connect()`.

## [1.1.1] - 2026-07-20
### Changed
- Renamed C++ headers from `.h` to `.hpp`; update downstream include directives.

## [1.1.0] - 2026-06-30
### Added
- Multi-sensor stereo support: `HSBSender`s sharing the same `ipAddress` now share one underlying `HSBEmulator`, matching real Leopard Eagle topology (one board, multiple VB1940 imagers).
- `HSBSender::~HSBSender` disconnects on destruction.

### Changed
- `HSBSender::connect()` rejects duplicate `sensorId` on the same IP, and bumps `hsb_ip_version` past the current hololink `MINIMUM_HSB_IP_VERSION`.

## [1.0.2] - 2026-06-25
### Removed
- Removed unused `omni.usd.libs` runtime dependency.

## [1.0.1] - 2026-06-09
### Fixed
- Fix linter errors and missing or incomplete docstrings.

## [1.0.0] - 2026-04-03
### Added
- Initial release. Backend C++/CUDA library extracted from `isaacsim.hsb.bridge`.
- `HSBSender`: HSB emulator communication layer.
- `RGBToVB1940Kernels`: GPU-accelerated RGB→VB1940 CSI format conversion.
- `IHsbCore` Carbonite plugin interface and Python bindings.
