<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Motion Generation Pick and Place

This example adapts the Franka flow from
`source/standalone_examples/api/isaacsim.robot_motion.examples/manipulation/pick_place.py` to the Kit-free libraries. It
loads a nine-DOF Franka articulation, time-parameterizes named joint-space pick/place waypoints with
`isaacsim.robot_motion.experimental.motion_generation`, applies the resulting targets through an OvPhysX articulation
entity, and displays the same live OVStage through the OVGL viewport.

The standalone reference uses `ManipulationScenario`, cuMotion RMPflow, and `PickPlaceController`. The migrated
Kit-free library currently provides the backend-neutral data model, path timing, controller composition, and world
binding, but its solver-specific cuMotion planner remains in a Kit extension. This example therefore uses precomputed
Franka joint waypoints with `Path.to_minimal_time_joint_trajectory()` and `TrajectoryFollower`; it does not add a
workflow-specific planning API.

The example references the Isaac Sim 6.1 Franka asset at `/World/Franka`. It uses `ISAACSIM_ASSET_ROOT` when configured
and otherwise falls back to NVIDIA's public asset root. The example authors the remote reference without resolving it in
the standalone OpenUSD stage, then OVStage's bundled OmniUsdResolver composes the complete asset and lets OmniClient
manage its local cache. OvPhysX simulates the asset's authored articulation and OVGL renders its authored geometry.
After each physics step, the application publishes the physics entities' measured link and cube poses to the same
OVStage immediately before rendering. The example converts the raw physics tensors' `xyzw` quaternions to Foundation's
`wxyz` convention at this synchronization boundary. The example does not replace or overlay the robot with procedural
geometry.

The grasp is contact-driven, matching the Franka path in the standalone reference. The example commands both finger
joints toward their closed positions and relies exclusively on the asset's finger colliders, drive force, and friction
to move the cube. It does not create an attachment, disable dynamics, or write the cube's physics pose after scene
initialization. The grasp gate requires the measured fingers to stop short around the cube and the measured rigid body
to remain aligned with the hand at least 0.05 m above the support for 0.2 seconds; overall completion requires a 0.15 m
lift, placement within 0.05 m, and settled linear speed below 0.10 m/s. The headless validation additionally requires
the initial and completed OVGL frames to differ.

The example authors its shapes, light, camera, and physics schemas through Foundation. It accesses the underlying
OpenUSD layer to author the intentionally unresolved robot reference and the Render attributes and relationships that
Foundation does not currently expose.

Run it through the examples runner from a terminal with access to a graphical desktop:

```bash
python examples.py run motion_generation.pick_place
```

Run the bounded headless validation with:

```bash
python examples.py run motion_generation.pick_place --headless
```

On Linux, this headless path uses EGL without X11 or Wayland. On Windows, it omits the user-visible window but still
requires a desktop-capable OpenGL session because the renderer creates a hidden SDL window and context.

Use left-drag to look, `WASD` to move, `Q`/`E` to move down/up, the mouse wheel to dolly, `R` to reset the view, and
Escape to close the viewport.
