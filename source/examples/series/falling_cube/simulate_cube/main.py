# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulate a cube falling onto a ground plane."""

from __future__ import annotations


def _author_stage() -> str:
    """Author the scene with Foundation and return its USDA text.

    Returns:
        Flattened USDA representation.

    """
    from isaacsim.foundation.objects import Cube, Prim
    from isaacsim.foundation.objects import Stage as FoundationStage
    from isaacsim.foundation.prims import ColliderBody, RigidBody

    foundation_stage = FoundationStage("openusd").create_stage()
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
        return foundation_stage.export_stage_to_string()
    finally:
        foundation_stage.close_stage()


def main() -> None:
    """Author and simulate a falling cube."""
    from isaacsim.physics_engines.ovphysx import activate, set_suppress_readback, shutdown

    if not activate():
        raise RuntimeError("OvPhysX physics engine is unavailable.")

    ovstage_stage = None
    simulation_initialized = False

    try:
        stage_text = _author_stage()

        import warp as wp
        from isaacsim.foundation.objects import Stage as FoundationStage
        from isaacsim.physics.manager import PhysicsManager, close, create_simulation_view, initialize, simulate
        from isaacsim.physics.registration import get_simulation_ids, get_simulation_name

        wp.init()
        ovstage_stage = FoundationStage("ovstage").import_stage_from_string(stage_text, make_default=False)
        if not PhysicsManager.get_instance().switch_physics_engine("ovphysx"):
            raise RuntimeError("OvPhysX physics engine is unavailable.")
        set_suppress_readback(False)

        if not initialize(ovstage_stage.get_stage_ptr(), "0", owner=ovstage_stage):
            raise RuntimeError("Physics initialization failed.")
        simulation_initialized = True
        simulation_id = next(
            (
                simulation_id
                for simulation_id in get_simulation_ids()
                if get_simulation_name(simulation_id) == "ovphysx"
            ),
            None,
        )
        if simulation_id is None:
            raise RuntimeError("OvPhysX simulation is not registered.")
        simulation_view = create_simulation_view("ovphysx", "warp", int(simulation_id))
        if simulation_view is None:
            raise RuntimeError("OvPhysX simulation view is unavailable.")
        cube = simulation_view.create_rigid_body_view("/World/Cube")

        def cube_height() -> float:
            return float(wp.from_dlpack(cube.get_data("transforms")).numpy()[0, 2])

        time_step = 1.0 / 60.0
        simulate(time_step, 0.0)
        initial_height = cube_height()

        for step in range(1, 120):
            simulate(time_step, step * time_step)
        final_height = cube_height()

        print(f"Cube height: {initial_height:.3f} -> {final_height:.3f}.")
        if final_height >= initial_height:
            raise RuntimeError("The cube did not fall.")
        print("The cube fell.")
    finally:
        if simulation_initialized:
            close()
        if ovstage_stage is not None:
            ovstage_stage.close_stage()
        shutdown()


if __name__ == "__main__":
    main()
