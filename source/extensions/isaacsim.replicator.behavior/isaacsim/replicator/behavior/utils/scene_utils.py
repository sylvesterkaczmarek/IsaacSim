# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavior-specific PhysX simulation helpers."""

from __future__ import annotations

import carb
import carb.settings
import omni.kit.app
import omni.physics.core
import omni.physx


async def run_simulation_async(sim_steps: int, physx_dt: float, render: bool = True) -> None:
    """Run the simulation for the specified number of steps. Optionally render the simulation by advancing the app.

    Args:
        sim_steps: Number of simulation steps to run.
        physx_dt: Physics time step duration.
        render: Whether to render each simulation step.
    """
    physx_sim_interface = omni.physx.get_physx_simulation_interface()
    for _ in range(sim_steps):
        physx_sim_interface.simulate(physx_dt, 0)
        physx_sim_interface.fetch_results()
        if render:
            await omni.kit.app.get_app().next_update_async()


async def apply_forces_and_simulate_async(
    stage_id: int,
    body_ids: list[int],
    forces: list[tuple[float, float, float]],
    positions: list[tuple[float, float, float]],
    sim_steps: int,
    physx_dt: float,
    render: bool = True,
) -> None:
    """Apply forces to assets at the specified positions, and simulate for the given number of steps.

    Args:
        stage_id: The stage identifier.
        body_ids: List of body identifiers to apply forces to.
        forces: List of force vectors to apply.
        positions: List of positions where forces are applied.
        sim_steps: Number of simulation steps to run.
        physx_dt: Physics time step duration.
        render: Whether to render each simulation step.
    """
    physx_sim_interface = omni.physx.get_physx_simulation_interface()

    # Apply the forces
    for body_id, force, position in zip(body_ids, forces, positions):
        physx_sim_interface.apply_force_at_pos(stage_id, body_id, carb.Float3(*force), carb.Float3(*position))

    # Run the simulation for the specified number of steps
    for _ in range(sim_steps):
        physx_sim_interface.simulate(physx_dt, 0)
        physx_sim_interface.fetch_results()
        if render:
            await omni.kit.app.get_app().next_update_async()


def disable_simulation_reset_on_stop() -> None:
    """Disable the simulation reset on stop setting. Needed to preserve the simulation state after play+stop."""
    carb.settings.get_settings().set(omni.physx.bindings._physx.SETTING_RESET_ON_STOP, False)


def reset_simulation_and_enable_reset_on_stop() -> None:
    """Reset the simulation and enable the reset-on-stop setting after preserving simulated state."""
    if carb.settings.get_settings().get(omni.physx.bindings._physx.SETTING_RESET_ON_STOP):
        carb.log_warn(
            "Expected 'omni.physx.bindings._physx.SETTING_RESET_ON_STOP' to be False, skipping reset_simulation"
        )
        return
    physics_stage_update_interface = omni.physics.core.get_physics_stage_update_interface()
    physics_stage_update_interface.reset_simulation()
    carb.settings.get_settings().set(omni.physx.bindings._physx.SETTING_RESET_ON_STOP, True)
