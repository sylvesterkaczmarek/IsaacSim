<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OVGL Viewport

This series contains focused examples for inspecting USD stages with the lightweight OVGL debug viewport. SDL owns
the native window and input layer. Linux uses EGL/OpenGL ES for offscreen rendering. Windows uses an SDL-managed
desktop OpenGL 4.1 context and does not require ANGLE.

From the examples collection root, run the C++ example with an authored test scene, a local stage, or an
HTTP/Omniverse URL:

```bash
python examples.py run ovgl_viewport.cpp
python examples.py run ovgl_viewport.cpp -- /path/to/stage.usd
```

The example uses Foundation C++ objects to own the OVStage stage and author the viewport camera pose. A native
Foundation package links its supplied OpenUSD libraries. When Foundation comes from a C++ and Python package, the
examples runner supplies the OpenUSD runtime required by the application entry point: the Linux C++ example links the
usd-exchange libraries, while the Windows runner adds the usd-exchange DLL directories to `PATH`. Python entry points
can instead import Foundation before loading dependent C++ bindings. The viewport remains independent of Foundation
and consumes only the native OVStage handle and a camera-pose callback.

OVStage resolves the stage's USD composition dependencies. The viewport uses OmniClient's local cache for remote image
textures and dome-light HDR files before passing them to the OpenGL image decoders.

Press `H` in the viewport window to toggle the debug HUD. It currently reports the viewport frame rate; additional
debug information can be added to the same window-side overlay without changing rendered frame captures.

Run interactive examples from a graphical desktop and keep the viewport lifecycle on the application main thread. On
Linux, invisible rendering uses EGL without X11 or Wayland. On Windows, invisible rendering omits the user-visible
window but still requires a desktop-capable OpenGL session because it uses a hidden SDL window and context.
