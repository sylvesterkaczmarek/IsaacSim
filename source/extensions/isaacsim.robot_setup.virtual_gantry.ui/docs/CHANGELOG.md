# Changelog

## [1.0.0] - 2026-07-28
### Added
- Initial release. Split out of `isaacsim.robot_setup.virtual_gantry` so the backend can be enabled without any UI dependencies (e.g. in headless experiences). Provides the `Create > Robotics > Virtual Gantry` menu entry, the `IsaacVirtualGantry` property panel with Enable/Disable, and the `G` / `[` / `]` hotkeys. No behaviour change; the code moved verbatim.
