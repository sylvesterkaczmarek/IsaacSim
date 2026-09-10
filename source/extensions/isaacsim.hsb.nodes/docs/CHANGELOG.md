# Changelog

## [1.1.0] - 2026-07-22
### Added
- `OgnHSBSend`: optional `calibrationIntrinsics` (`double[]`) and `calibrationTranslation` (`double[3]`) inputs, forwarded to `HSBSender::setCalibration` and programmed into the emulated VB1940 rig EEPROM (I²C peripheral `0x51`).
- `OgnHSBCameraHelper.py`: reads the parent USDA Camera prim to compute fx/fy/cx/cy and packs them into the new `OgnHSBSend` inputs, with a default Eagle stereo baseline.

## [1.0.4] - 2026-07-20
### Changed
- Renamed C++ headers from `.h` to `.hpp`; update downstream include directives.

## [1.0.3] - 2026-06-09
### Fixed
- Fix linter errors and missing or incomplete docstrings.

## [1.0.2] - 2026-05-14
### Changed
- Address ruff lint errors: add type annotations and docstrings to `OgnHSBCameraHelper`, extension lifecycle methods, and tests; rewrite `dict()` call as a dict literal.

## [1.0.1] - 2026-05-05
### Removed
- Enabled multitick, removed `frameSkipCount` input as it's controlled directly on sensor prim.

## [1.0.0] - 2026-04-03
### Added
- Initial release. OmniGraph nodes extracted from `isaacsim.hsb.bridge`.
- `HSBSend`: Send DLTensor data buffers via HSB emulator.
- `RGBToVB1940`: GPU-accelerated RGB→VB1940 CSI format conversion node.
- `HSBCameraHelper`: Python helper node for Replicator-based camera publishing.
- Node type IDs use `isaacsim.hsb.nodes` namespace.
