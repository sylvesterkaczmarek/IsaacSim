<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OVGL Viewport

This C++ example loads an optional USD stage through Foundation's public OVStage adapter and displays it in the
concrete `isaacsim.ovgl_viewport.debug` window. With no stage argument, it authors a visible cube and light in its
application-owned USDA layer.

The example authors its camera, RenderVar, and RenderProduct in the same application-owned layer. The viewport camera
callback updates the camera prim through the Foundation OVStage adapter. The viewport renders only the authored
product; it does not own scene content or simulation.

The camera uses the Isaac Sim perspective viewport defaults: a 20.955 horizontal aperture, an 18.147562 focal length
(60-degree horizontal field of view), and a clipping range of 1 to 10,000,000 stage units.

The application-owned overlay does not override the input stage's units or up-axis. OVStage resolves USD composition
dependencies, and the viewport caches supported remote texture and dome-light image assets through OmniClient.

On Linux, visible runs require an accessible X11 display and a compatible EGL/OpenGL implementation;
headless runs use EGL without a display server. On Windows, visible and headless runs require a desktop-capable OpenGL
4.1 session because the renderer uses an SDL-managed OpenGL context. The module package carries its private SDL
runtime, so applications do not need an SDL development package. Build and run it through the examples runner after
installing the `isaacsim_foundation` and `isaacsim_ovgl_viewport` native SDKs into `CMAKE_PREFIX_PATH`:

```bash
python examples.py run ovgl_viewport.cpp
python examples.py run ovgl_viewport.cpp -- /path/to/stage.usd
python examples.py run ovgl_viewport.cpp -- /path/to/stage.usd --frames 10
python examples.py run ovgl_viewport.cpp -- --headless --frames 1
```

For a Linux desktop on a non-default display, prefix an interactive command with a display assignment such as
`DISPLAY=:1`.

The authored RenderProduct remains at 1280 x 720. Resizing the window scales and letterboxes the rendered image without
changing the stage-authored resolution. `--frames COUNT` bounds a run for smoke testing; omit it for an interactive
session. `--headless` captures frames without creating a user-visible presentation window and is used by the automated
smoke test. On Windows, this mode still creates a hidden SDL window and OpenGL context and therefore is not independent
of the desktop graphics session. Create, use, and destroy the viewport on the application main thread. The first frame
populates OVGL's resident scene and is expected to take longer than subsequent frames. Set `OVGL_PROFILE=1` to print
the public OVStage update, scene synchronization, and rendering timings when diagnosing a large stage.

Controls:

- Left drag: first-person mouse look.
- Mouse wheel: dolly.
- Hold W/S: move forward/backward.
- Hold A/D: strafe left/right.
- Hold Q/E: move down/up.
- Shift: accelerate movement.
- R: reset the camera.
- H: toggle the frame-rate HUD.
- Escape: close the viewport.
