# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize the falling-cube simulation with the simple OpenGL renderer."""

from __future__ import annotations

import argparse
import os
import time

_PHYSICS_DT = 1.0 / 60.0


def _author_stage() -> str:
    """Author the simulated and rendered scene as neutral USDA text.

    Returns:
        USDA representation of the authored root layer.
    """
    from isaacsim.foundation.objects import Camera as FoundationCamera
    from isaacsim.foundation.objects import Cube, DistantLight, Prim
    from isaacsim.foundation.objects import Stage as FoundationStage
    from isaacsim.foundation.prims import ColliderBody, RigidBody
    from pxr import Gf, Sdf, Usd, UsdUtils

    foundation_stage = FoundationStage("openusd").create_stage()
    stage_id = foundation_stage.get_stage_id()
    usd_stage = UsdUtils.StageCache.Get().Find(Usd.StageCache.Id.FromLongInt(stage_id))
    if not usd_stage:
        foundation_stage.close_stage()
        raise RuntimeError("Foundation did not create a cached USD stage.")

    try:
        foundation_stage.set_up_axis("Z")
        foundation_stage.set_units(meters_per_unit=1.0, kilograms_per_unit=1.0)
        foundation_stage.define_prim("/World")

        Cube(
            "/World/Ground",
            sizes=1.0,
            translations=[0.0, 0.0, -0.5],
            scales=[10.0, 10.0, 1.0],
        )
        Cube(
            "/World/Cube",
            sizes=1.0,
            translations=[0.0, 0.0, 2.0],
        )

        foundation_stage.define_prim("/World/PhysicsScene", "PhysicsScene")
        physics_scene = Prim("/World/PhysicsScene")
        physics_scene.set_attribute_values("physics:gravityDirection", [0.0, 0.0, -1.0])
        physics_scene.set_attribute_values("physics:gravityMagnitude", 9.81)
        ColliderBody("/World/Ground")
        ColliderBody("/World/Cube")
        RigidBody("/World/Cube")

        light = DistantLight("/World/KeyLight")
        light.set_intensities([[3000.0]])
        light.set_colors([[1.0, 0.92, 0.78]])

        viewport_camera = FoundationCamera(
            "/ViewportCamera",
            positions=[[0.0, 0.0, 7.0]],
            orientations=[[1.0, 0.0, 0.0, 0.0]],
        )
        # Match the authored defaults of Isaac Sim's `/OmniverseKit_Persp` camera. The Foundation camera API expresses
        # focal length in scene units; USD stores it in tenths of a scene unit, so `1.8147562` authors `18.147562`.
        viewport_camera.set_focal_lengths(1.8147562)
        viewport_camera.set_clipping_ranges(1.0, 10_000_000.0)
        foundation_stage.define_prim("/Render/Vars/LdrColor", "RenderVar")
        render_var = usd_stage.GetPrimAtPath("/Render/Vars/LdrColor")
        render_var.CreateAttribute(
            "dataType", Sdf.ValueTypeNames.Token, custom=False, variability=Sdf.VariabilityUniform
        ).Set("color4f")
        render_var.CreateAttribute(
            "sourceName", Sdf.ValueTypeNames.String, custom=False, variability=Sdf.VariabilityUniform
        ).Set("LdrColor")
        foundation_stage.define_prim("/Render/ViewportProduct", "RenderProduct")
        render_product = usd_stage.GetPrimAtPath("/Render/ViewportProduct")
        render_product.GetRelationship("camera").SetTargets(["/ViewportCamera"])
        render_product.GetRelationship("orderedVars").SetTargets(["/Render/Vars/LdrColor"])
        render_product.CreateAttribute(
            "resolution", Sdf.ValueTypeNames.Int2, custom=False, variability=Sdf.VariabilityUniform
        ).Set(Gf.Vec2i(1280, 720))

        return foundation_stage.export_stage_to_string()
    finally:
        foundation_stage.close_stage()


def main() -> None:
    """Simulate the cube while continuously rendering its OVStage updates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="run and validate without opening a user-visible viewport window"
    )
    arguments = parser.parse_args()

    os.environ.setdefault("OVGL_SS", "1")
    from isaacsim.physics_engines.ovphysx import activate, set_suppress_readback, shutdown

    if not activate():
        raise RuntimeError("OvPhysX physics engine is unavailable.")

    ovstage_stage = None
    physics_manager = None
    viewport = None
    try:
        stage_text = _author_stage()

        import numpy as np
        from isaacsim.foundation.objects import Stage as FoundationStage
        from isaacsim.foundation.objects import Xform
        from isaacsim.ovgl_viewport.debug import Camera, CameraPose, Viewport, ViewportConfig
        from isaacsim.physics.entities import RigidBodyEntity
        from isaacsim.physics.manager import PhysicsManager

        ovstage_stage = FoundationStage("ovstage").import_stage_from_string(stage_text)
        physics_manager = PhysicsManager.get_instance()
        if not physics_manager.switch_physics_engine("ovphysx"):
            raise RuntimeError("OvPhysX physics engine is unavailable.")
        physics_manager.setup(_PHYSICS_DT)
        set_suppress_readback(False)
        if not physics_manager.initialize(ovstage_stage.get_stage_ptr(), 0):
            raise RuntimeError("Physics initialization failed.")
        simulated_cube = RigidBodyEntity("ovphysx", "/World/Cube")

        camera = Camera()
        camera.target = (0.0, 0.0, 0.75)
        camera.distance = 7.0
        config = ViewportConfig()
        config.title = "Falling Cube Simulation"
        config.visible = not arguments.headless
        config.camera = camera
        config.render_product_path = "/Render/ViewportProduct"
        camera_xform = Xform("/ViewportCamera", resolve_paths=False, reset_xform_op_properties=False)
        cube_xform = Xform("/World/Cube", resolve_paths=False, reset_xform_op_properties=False)

        def write_camera_pose(pose: CameraPose) -> None:
            camera_xform.set_world_poses(
                positions=np.asarray([pose.position], dtype=np.float64),
                orientations=np.asarray([pose.orientation], dtype=np.float64),
            )

        viewport = Viewport(ovstage_stage.get_stage_ptr(), write_camera_pose, config)
        if not viewport.poll_events():
            raise RuntimeError("OVGL viewport closed before its initial render.")
        initial_frame = viewport.render()
        initial_pixels = initial_frame.rgba
        initial_ordinal = initial_frame.stage_ordinal

        print("Dropping the cube now.")
        positions, _ = simulated_cube.get_world_poses()
        initial_height = float(positions.numpy()[0, 2])
        final_height = initial_height
        simulation_step = 0
        last_frame_time = time.monotonic()
        accumulated_simulation_time = 0.0

        while viewport.poll_events():
            if arguments.headless:
                step_count = 120 - simulation_step
            else:
                current_time = time.monotonic()
                accumulated_simulation_time += min(current_time - last_frame_time, 0.25)
                last_frame_time = current_time
                step_count = min(int(accumulated_simulation_time / _PHYSICS_DT), 15, 120 - simulation_step)
                accumulated_simulation_time -= step_count * _PHYSICS_DT

            simulation_step += physics_manager.step(steps=step_count)

            if step_count:
                transforms = simulated_cube.get_data("transforms").numpy()
                final_height = float(transforms[0, 2])
                # Physics tensor poses use `xyzw`; Foundation `Xform` expects `wxyz`.
                cube_xform.set_world_poses(
                    positions=transforms[:, :3].astype(np.float64),
                    orientations=transforms[:, [6, 3, 4, 5]].astype(np.float64),
                )

            frame = viewport.render()
            if simulation_step == 120:
                break

        if final_height >= initial_height:
            raise RuntimeError("The simulated cube did not fall.")
        print(f"Cube top: {initial_height:.3f} -> {final_height:.3f}.")
        if arguments.headless:
            if frame.stage_ordinal <= initial_ordinal or frame.rgba == initial_pixels:
                raise RuntimeError("OVGL did not render the falling cube's OVStage transform updates.")
            print("Headless rendering smoke test passed.")
        else:
            print(
                "Controls: left-drag look, hold WASD to move, hold Q/E for down/up, wheel dolly, R reset, Escape quit."
            )
            while viewport.poll_events():
                viewport.render()
        print(f"Viewport rendered {frame.frame_number} frame(s).")
    finally:
        viewport = None
        try:
            if physics_manager is not None and physics_manager.is_initialized():
                physics_manager.invalidate()
        finally:
            try:
                if ovstage_stage is not None:
                    ovstage_stage.close_stage()
            finally:
                shutdown()


if __name__ == "__main__":
    main()
