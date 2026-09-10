# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulate a cube falling onto a ground plane."""

from __future__ import annotations

from isaacsim.ovsim.api import make_client
from isaacsim.physics_engines.ovphysx import activate, shutdown


def main() -> None:
    """Author and simulate a falling cube."""
    client = make_client("in-process")  # local or in-process
    simulation_initialized = False
    stage_open = False

    try:
        # ---------
        # authoring

        if not client.control.authoring.create_stage():
            raise RuntimeError("Stage creation failed.")
        stage_open = True

        cube = "/World/Cube"
        client.control.authoring.define_prim(cube, "Cube")
        client.data.write(cube, "size", 1.0)
        client.data.write(cube, "position", [0.0, 0.0, 2.0])
        client.control.authoring.define_prim(cube, "ColliderBody")
        client.control.authoring.define_prim(cube, "RigidBody")

        physics_scene = "/World/PhysicsScene"
        client.control.authoring.define_prim(physics_scene, "PhysicsScene")
        client.data.write(physics_scene, "physics:gravityDirection", [0.0, 0.0, -1.0])
        client.data.write(physics_scene, "physics:gravityMagnitude", 9.81)

        # ----------
        # simulation

        if not activate():
            raise RuntimeError("OvPhysX physics engine is unavailable.")

        client.control.simulation.set_parameter("physics", "physics-engine", "ovphysx")
        client.control.simulation.initialize()
        simulation_initialized = True

        initial_height = float(client.data.read(cube, "position").numpy()[0, 2])
        for _ in range(10):
            client.control.simulation.step()
        final_height = float(client.data.read(cube, "position").numpy()[0, 2])

        print(f"Cube height: {initial_height:.3f} -> {final_height:.3f}.")
        if final_height >= initial_height:
            raise RuntimeError("The cube did not fall.")
        print("The cube fell.")
    finally:
        if simulation_initialized:
            client.control.simulation.invalidate()
        if stage_open:
            client.control.authoring.close_stage()
        shutdown()


if __name__ == "__main__":
    main()
