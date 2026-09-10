# Changelog

## [Unreleased]

### Added
- `isaacsim.robot_motion.experimental.motion_generation`: initial Kit-independent release of the motion generation
  library (trajectory following, obstacle handling, controller interfaces). The public API and import path are
  unchanged from the Kit extension of the same name; the implementation no longer depends on `carb` or `omni.*` and
  resolves USD through the bundled Kit-free OpenUSD build.
