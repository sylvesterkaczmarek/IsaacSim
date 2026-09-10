# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Verify contact sensor readings on CPU and GPU physics simulation devices."""

import omni.kit.app
import omni.kit.test
import omni.timeline
from isaacsim.core.experimental.objects import Cube, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.physics import Contact, ContactSensor

from .common import is_cuda_sim_available


class TestContactSensorDevice(omni.kit.test.AsyncTestCase):
    """Contact sensor physics-only stepping on CPU and CUDA simulation devices."""

    async def setUp(self) -> None:
        """Store the active physics device and timeline for restoration."""
        self._physics_rate = 60
        self._timeline = omni.timeline.get_timeline_interface()
        self._original_physics_device = SimulationManager.get_physics_sim_device()

    async def tearDown(self) -> None:
        """Restore the original physics device and invalidate cached physics state."""
        if self._timeline.is_playing():
            self._timeline.stop()
        SimulationManager.set_physics_sim_device(self._original_physics_device)
        SimulationManager.invalidate_physics()
        await omni.kit.app.get_app().next_update_async()

    async def _run_physics_only_contact_test(self, device: str) -> None:
        """Step physics without render updates and assert contact sensor output.

        Args:
            device: Physics simulation device passed to ``SimulationManager.set_physics_sim_device``.
        """
        SimulationManager.set_physics_sim_device(device)
        await omni.kit.app.get_app().next_update_async()

        await stage_utils.create_new_stage_async()
        await omni.kit.app.get_app().next_update_async()

        stage_utils.set_stage_units(meters_per_unit=1.0)
        SimulationManager.setup_simulation(dt=1.0 / self._physics_rate)

        GroundPlane("/World/GroundPlane", sizes=10.0)
        Cube("/World/Cube", sizes=1.0, positions=[0.0, 0.0, 0.5])
        GeomPrim("/World/Cube", apply_collision_apis=True)
        RigidPrim("/World/Cube", masses=[1.0])
        await omni.kit.app.get_app().next_update_async()

        sensor = ContactSensor(
            Contact.create(
                "/World/Cube/contact_sensor",
                min_threshold=0,
                max_threshold=10000000,
                radius=-1,
            )
        )
        self.assertFalse(sensor.get_sensor_reading().is_valid, "Reading should be invalid before physics steps")

        try:
            from isaacsim.sensors.experimental.physics.impl.extension import get_contact_sensor_interface

            self._timeline.play()
            SimulationManager.initialize_physics()
            iface = get_contact_sensor_interface()
            self.assertTrue(iface.create_sensor("/World/Cube/contact_sensor"))
            SimulationManager.step(steps=120, update_fabric=False)

            reading = sensor.get_sensor_reading()
            self.assertTrue(reading.is_valid, f"Reading should be valid on device '{device}' after physics-only steps")
            self.assertTrue(reading.in_contact, f"Cube should contact the ground on device '{device}'")
            self.assertGreater(reading.value, 0.0)
        finally:
            sensor.reset()
            if self._timeline.is_playing():
                self._timeline.stop()
                await omni.kit.app.get_app().next_update_async()

    async def test_physics_only_step_outputs_contact_data_cpu(self) -> None:
        """ContactSensor produces data when physics runs on the CPU device."""
        await self._run_physics_only_contact_test("cpu")

    async def test_physics_only_step_outputs_contact_data_cuda(self) -> None:
        """ContactSensor produces data when physics runs on the CUDA device."""
        if not is_cuda_sim_available():
            self.skipTest("CUDA is not available for physics simulation")
        await self._run_physics_only_contact_test("cuda")
