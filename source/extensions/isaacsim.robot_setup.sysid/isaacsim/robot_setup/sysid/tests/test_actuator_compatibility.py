# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for simulation and actuator capability resolution."""

import omni.kit.test
from isaacsim.robot_setup.sysid.actuator_compatibility import (
    ACTUATOR_RUNTIME_EXPLICIT,
    ACTUATOR_RUNTIME_IMPLICIT,
    ACTUATOR_RUNTIME_MIXED,
    PHYSICS_BACKEND_NEWTON,
    PHYSICS_BACKEND_PHYSX,
    resolve_actuator_dof_plan,
    resolve_simulation_compatibility,
)
from isaacsim.robot_setup.sysid.parameter_types import SysIdParameterType
from isaacsim.robot_setup.sysid.run_spec import ParameterRunSpec, SimulationRunSpec


class ActuatorCompatibilityTests(omni.kit.test.AsyncTestCase):
    """Verify execution-axis and per-DOF actuator resolution."""

    async def test_isaac_sim_newton_and_mixed_actuators_are_independent_axes(
        self,
    ) -> None:
        """Resolve physics and actuator selections independently."""
        simulation = SimulationRunSpec(
            engine="isaac_sim",
            physics_backend="newton",
            actuator_runtime="mixed",
        )

        resolved = resolve_simulation_compatibility(simulation)

        self.assertEqual(resolved.physics_backend, PHYSICS_BACKEND_NEWTON)
        self.assertEqual(resolved.actuator_runtime, ACTUATOR_RUNTIME_MIXED)
        self.assertTrue(resolved.supports_delay)

    async def test_isaac_sim_defaults_to_physx_implicit(self) -> None:
        """Use PhysX implicit drives for the default Isaac Sim configuration."""
        resolved = resolve_simulation_compatibility(SimulationRunSpec())

        self.assertEqual(resolved.physics_backend, PHYSICS_BACKEND_PHYSX)
        self.assertFalse(resolved.uses_explicit_actuators)

    async def test_differentiable_newton_auto_resolves_to_explicit_pd(self) -> None:
        """Resolve differentiable Newton rollouts to explicit PD actuation."""
        simulation = SimulationRunSpec(engine="newton")
        simulation.newton.solver = "featherstone_diff"

        resolved = resolve_simulation_compatibility(simulation)

        self.assertEqual(resolved.actuator_runtime, ACTUATOR_RUNTIME_EXPLICIT)
        self.assertFalse(resolved.supports_delay)

    async def test_per_dof_ownership_is_total_and_unique_for_every_runtime(
        self,
    ) -> None:
        """Assign exactly one actuator owner to every DOF."""
        delay = ParameterRunSpec(
            param_type=SysIdParameterType.ACTUATOR_COMMAND_DELAY_SECONDS.value,
            dof_index=1,
        ).to_entry()

        implicit = resolve_actuator_dof_plan(ACTUATOR_RUNTIME_IMPLICIT, 3, parameter_entries=[delay])
        explicit = resolve_actuator_dof_plan(ACTUATOR_RUNTIME_EXPLICIT, 3, parameter_entries=[delay])
        mixed = resolve_actuator_dof_plan(
            ACTUATOR_RUNTIME_MIXED,
            3,
            authored_explicit_dofs=[0],
            parameter_entries=[delay],
        )

        self.assertEqual([item.explicit for item in implicit], [False, False, False])
        self.assertEqual([item.explicit for item in explicit], [True, True, True])
        self.assertEqual([item.explicit for item in mixed], [True, True, False])
