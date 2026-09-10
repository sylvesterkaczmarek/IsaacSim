# Changelog

## [Unreleased]

### Added

- Add an `H`-key debug HUD showing the visible viewport's frame rate.

### Changed

- Replace the X11-specific window, input, presentation, and HUD code with a privately linked, pinned SDL3 dependency.
- Fetch the pinned `stb_image` source during module dependency preparation instead of storing a copy in the module.
- Cache remote material images and dome-light HDRs through the OmniClient runtime bundled with OVStage.

### Fixed

- Build and run the OVGL debug viewport on Windows with an SDL-managed desktop OpenGL context and Windows runtime
  dependency staging.
- Use FPS-style mouse-look direction and absolute drag input without relative-mode cursor jumps.
- Keep visible rendering at the configured viewport resolution on scaled desktop displays.
- Make debug viewport movement continuous while movement keys are held.
- Decode official OVStage asset attributes as authored/resolved token pairs.
- Preserve populated hierarchy and scene-graph instance transforms in the OVGL viewport mirror.

## [6.1.0]

### Added

- Add the concrete OVGL debug viewport for populated OVStage scenes.
