<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Visualize the Simulation

This step runs the falling-cube physics scenario from the previous example and continuously presents the evolving
OVStage through the concrete `isaacsim.ovgl_viewport.debug.Viewport` window.

Physics owns the simulated pose, Foundation owns stage authoring, and the viewport only renders the stage. Before each
displayed frame, the example reads the raw `xyzw` tensor transform through `RigidBodyEntity`, converts its orientation
to Foundation's `wxyz` convention, and writes it to OVStage through `Xform`. This explicit coordination keeps
physics-to-stage synchronization and representation conversion out of both the physics library and the viewport.
`PhysicsManager` owns initialization, fixed-step advancement, and invalidation of the selected OvPhysX engine.

The scene authors the viewport camera and RenderProduct before it is populated into OVStage. Interactive navigation
updates that camera on OVStage through the viewport callback, while the renderer only renders the authored product.
The camera uses the Isaac Sim perspective viewport defaults: a 20.955 horizontal aperture, an 18.147562 focal length
(60-degree horizontal field of view), and a clipping range of 1 to 10,000,000 stage units.

Run it through the installed examples runner on Linux or Windows from a terminal with access to a graphical desktop:

```bash
python examples.py run falling_cube.visualize_simulation
```

The configured automated test runs the same stage, explicit pose synchronization, simulation, and OVGL viewport without
opening a user-visible presentation window. It renders before and after simulation and fails if the cube's OVStage
transform updates do not change the rendered image:

```bash
python examples.py test falling_cube.visualize_simulation:headless
```

On Linux, this headless path uses EGL without X11 or Wayland. On Windows, it still requires a desktop-capable OpenGL
session because the renderer creates a hidden SDL window and context.

Use left-drag to look, hold WASD to move, hold Q/E to move down/up, the wheel to dolly, R to reset, and Escape to quit.
The terminal prints when the drop starts. The stage-authored RenderProduct remains at 1280 x 720, while the viewport
scales and letterboxes it when the window is resized.
