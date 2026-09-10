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

"""Isaac Lab-to-Isaac Sim Cartpole policy transfer regression tests.

Each test deploys one Isaac Lab-trained Cartpole policy in standalone Isaac Sim. The active Isaac Sim
physics engine and host platform select the matching golden trajectory, so the extension's PhysX and
Newton test sections together cover both same-engine and cross-engine policy-transfer paths.
"""

from __future__ import annotations

import asyncio
import json
import platform
from pathlib import Path

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.app
import omni.kit.test
import omni.physx
import omni.usd
from isaacsim.core.experimental.utils.stage import define_prim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.robot.policy.examples import PolicyEnvConfig, RobotPolicyRunner, get_cartpole_spec
from isaacsim.robot.policy.examples.interactive.utils import (
    restore_physics_simulation_state,
    snapshot_physics_simulation_state,
)


class TestCartpoleIsaacLabTransfer(omni.kit.test.AsyncTestCase):
    """Verify exported Isaac Lab Cartpole policies deploy in Isaac Sim from policy.onnx plus env.yaml."""

    DEVICE = "cuda:0"
    EXPECTED_GOLDEN_DEVICE = None
    EXPECTED_ONNX_PROVIDER = "CUDAExecutionProvider"

    async def setUp(self) -> None:
        """Prepare a clean simulation state for each transfer case."""
        self._engine = (SimulationManager.get_active_physics_engine() or "physx").lower()
        if self._engine not in ("physx", "newton"):
            self.skipTest(f"No Cartpole golden trajectory for physics engine {self._engine!r}.")
        self._golden_dir = Path(__file__).resolve().parents[5] / "data" / "tests"
        self._initial_device, self._initial_fabric = snapshot_physics_simulation_state()
        self._initial_warm_start_enabled = SimulationManager.is_default_callback_enabled("warm_start")
        self._initial_default_physics_scene = SimulationManager.get_default_physics_scene()
        self._initial_physics_dt = (
            SimulationManager.get_physics_dt(self._initial_default_physics_scene)
            if self._initial_default_physics_scene is not None
            else None
        )
        self._physics_callback_id = None
        self._timeline_started = False
        self._cartpole = None

    async def tearDown(self) -> None:
        """Stop the scene and restore process-wide simulation state."""
        await self._stop_scene()
        self._restore_simulation_manager_state()
        restore_physics_simulation_state(self._initial_device, self._initial_fabric)
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()

    async def test_isaaclab_physx_policy_transfers_to_active_isaacsim_engine(self) -> None:
        """Verify transfer of the PhysX-trained policy to the active engine."""
        await self._assert_isaaclab_policy_transfers_to_active_engine(
            policy_training_engine="physx",
        )

    async def test_isaaclab_newton_policy_transfers_to_active_isaacsim_engine(self) -> None:
        """Verify transfer of the Newton-trained policy to the active engine."""
        await self._assert_isaaclab_policy_transfers_to_active_engine(
            policy_training_engine="newton",
        )

    def _golden_path(
        self,
        *,
        policy_training_engine: str,
        isaacsim_engine: str,
    ) -> Path:
        platform_name = platform.system().lower()
        self.assertIn(platform_name, ("linux", "windows"))
        device_suffix = f"_{self.EXPECTED_GOLDEN_DEVICE}" if self.EXPECTED_GOLDEN_DEVICE else ""
        golden_path = (
            self._golden_dir
            / f"cartpole_{policy_training_engine}_onnx_policy"
            / platform_name
            / f"trajectory_{isaacsim_engine}{device_suffix}_300_steps.npz"
        )
        self.assertTrue(golden_path.is_file(), f"Cartpole golden trajectory fixture is missing: {golden_path}")
        return golden_path

    def _load_golden(self, golden_path: Path) -> dict[str, object]:
        with np.load(golden_path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            self.assertEqual(meta["backend"], "isaaclab")
            self.assertEqual(meta["engine"], self._engine)
            if self.EXPECTED_GOLDEN_DEVICE is not None:
                self.assertEqual(meta.get("device"), self.EXPECTED_GOLDEN_DEVICE)
            self.assertFalse(meta["failed"])
            return {
                "meta": meta,
                **{
                    field: np.asarray(data[field], dtype=np.float32)
                    for field in ("t", "cart_pos", "pole_pos", "cart_vel", "pole_vel", "action", "effort")
                },
            }

    async def _assert_isaaclab_policy_transfers_to_active_engine(
        self,
        *,
        policy_training_engine: str,
    ) -> None:
        golden_path = self._golden_path(
            policy_training_engine=policy_training_engine,
            isaacsim_engine=self._engine,
        )
        golden = self._load_golden(golden_path)

        artifact = get_cartpole_spec().engines[policy_training_engine]
        env_config_path = artifact.env_config_path
        policy_path = artifact.model_path
        env_config = PolicyEnvConfig.from_file(env_config_path)
        physics_dt = env_config.timing.physics_dt
        decimation = env_config.timing.decimation
        policy_dt = physics_dt * decimation
        self.assertAlmostEqual(float(golden["meta"]["dt"]), policy_dt)

        # Guard the Isaac Lab xyzw -> Isaac Sim wxyz spawn-orientation conversion: these fixtures spawn
        # at identity, so a regression that passed init_state.rot through unconverted would yield
        # [0, 0, 0, 1] (a 180-deg-about-Z spawn) and silently break same-engine bitwise parity.
        _, spawn_orientation = env_config.initial_root_pose
        self.assertEqual(spawn_orientation, [1.0, 0.0, 0.0, 0.0])

        await self._create_cartpole_scene(
            policy_training_engine=policy_training_engine,
            physics_dt=physics_dt,
        )
        self._assert_onnx_provider_active(policy_path)
        # Strict trajectory parity needs the same rollout state sample used by the Isaac Lab golden.
        self._reset_state_from_golden(golden)

        self._actual = {field: [] for field in golden if field != "meta"}
        self._physics_step = 0
        self._target_policy_steps = int(golden["cart_pos"].shape[0])
        self._decimation = decimation
        self._policy_dt = policy_dt
        if self._engine == "physx":
            self._run_direct_physx_rollout(physics_dt)
        else:
            self._physics_callback_id = SimulationManager.register_callback(
                self._on_physics_step, event=IsaacEvents.PRE_PHYSICS_STEP
            )

            try:
                max_updates = self._target_policy_steps * decimation + 30
                for _ in range(max_updates):
                    if len(self._actual["t"]) >= self._target_policy_steps:
                        break
                    await omni.kit.app.get_app().next_update_async()
            finally:
                if self._physics_callback_id is not None:
                    SimulationManager.deregister_callback(self._physics_callback_id)
                    self._physics_callback_id = None
        self.assertEqual(len(self._actual["t"]), self._target_policy_steps)

        failures = []
        for field, expected in golden.items():
            if field == "meta":
                continue
            measured = np.asarray(self._actual[field], dtype=np.float32)
            if measured.shape != expected.shape:
                failures.append(f"{field}: shape mismatch actual={measured.shape} expected={expected.shape}")
                continue
            if np.array_equal(measured, expected):
                continue
            mismatch = measured != expected
            diff = np.abs(measured - expected)
            first_index = int(np.flatnonzero(mismatch)[0])
            failures.append(
                f"{field}: {int(np.count_nonzero(mismatch))}/{expected.size} values differed, "
                f"max_abs={float(diff.max()):.9g}, "
                f"first[{first_index}] actual={float(measured.flat[first_index]):.9g} "
                f"expected={float(expected.flat[first_index]):.9g}"
            )
        if failures:
            transfer_kind = "same-engine" if policy_training_engine == self._engine else "cross-engine"
            self.fail(
                f"Isaac Lab Cartpole {policy_training_engine}-trained policy failed {transfer_kind} transfer to "
                f"Isaac Sim {self._engine} against {golden_path}:\n" + "\n".join(failures)
            )

    def _assert_onnx_provider_active(self, policy_path: str) -> None:
        if self._cartpole is None:
            raise RuntimeError("Cartpole scene has not been initialized.")
        providers = getattr(self._cartpole._controller._model, "providers", None)
        self.assertIsNotNone(providers, f"Cartpole policy {policy_path} did not expose ONNXRuntime providers.")
        self.assertIn(
            self.EXPECTED_ONNX_PROVIDER,
            providers,
            f"Cartpole golden regression requires ONNXRuntime {self.EXPECTED_ONNX_PROVIDER} for {policy_path}; "
            f"active providers: {providers}",
        )

    async def _create_cartpole_scene(
        self,
        *,
        policy_training_engine: str,
        physics_dt: float,
    ) -> None:
        await self._stop_scene()
        await stage_utils.create_new_stage_async()
        define_prim("/physicsScene", "PhysicsScene")
        SimulationManager.enable_warm_start_callback(True)
        SimulationManager.set_default_physics_scene("/physicsScene")
        SimulationManager.set_physics_sim_device(self.DEVICE)
        SimulationManager.set_physics_dt(physics_dt, "/physicsScene")

        self._cartpole = RobotPolicyRunner(
            get_cartpole_spec(),
            prim_path="/World/Cartpole",
            training_engine=policy_training_engine,
        )
        self._cartpole.spawn()
        await omni.kit.app.get_app().next_update_async()
        app_utils.play()
        self._timeline_started = True
        await omni.kit.app.get_app().next_update_async()
        self._cartpole.initialize()
        await omni.kit.app.get_app().next_update_async()

    async def _stop_scene(self) -> None:
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None
        if self._timeline_started:
            app_utils.stop()
            await omni.kit.app.get_app().next_update_async()
            self._timeline_started = False
        self._cartpole = None

    def _restore_simulation_manager_state(self) -> None:
        SimulationManager.enable_warm_start_callback(self._initial_warm_start_enabled)
        SimulationManager._default_physics_scene_path = self._initial_default_physics_scene
        if self._initial_physics_dt is not None:
            SimulationManager.set_physics_dt(self._initial_physics_dt, self._initial_default_physics_scene)

    def _reset_state_from_golden(self, golden: dict[str, object]) -> None:
        if self._cartpole is None:
            raise RuntimeError("Cartpole scene has not been initialized.")
        dof_names = list(self._cartpole.articulation.dof_names)
        positions = np.zeros((1, len(dof_names)), dtype=np.float32)
        velocities = np.zeros((1, len(dof_names)), dtype=np.float32)
        positions[0, dof_names.index("slider_to_cart")] = float(golden["cart_pos"][0])
        positions[0, dof_names.index("cart_to_pole")] = float(golden["pole_pos"][0])
        velocities[0, dof_names.index("slider_to_cart")] = float(golden["cart_vel"][0])
        velocities[0, dof_names.index("cart_to_pole")] = float(golden["pole_vel"][0])
        self._cartpole.articulation.set_dof_efforts(np.zeros((1, len(dof_names)), dtype=np.float32))
        self._cartpole.articulation.set_dof_positions(positions)
        self._cartpole.articulation.set_dof_velocities(velocities)
        self._cartpole.articulation.set_dof_position_targets(np.zeros((1, len(dof_names)), dtype=np.float32))
        self._cartpole.articulation.set_dof_velocity_targets(np.zeros((1, len(dof_names)), dtype=np.float32))

    def _read_joint_state(self) -> tuple[dict[str, float], dict[str, float]]:
        """Read per-joint positions and velocities by name from the deployed articulation.

        Returns:
            The position and velocity maps keyed by DOF name.
        """
        articulation = self._cartpole.articulation
        names = list(articulation.dof_names)
        positions = np.asarray(articulation.get_dof_positions().numpy(), dtype=np.float32).reshape(-1)
        velocities = np.asarray(articulation.get_dof_velocities().numpy(), dtype=np.float32).reshape(-1)
        return (
            {name: float(positions[i]) for i, name in enumerate(names)},
            {name: float(velocities[i]) for i, name in enumerate(names)},
        )

    def _read_efforts(self) -> dict[str, float]:
        """Read the efforts the runner applied, by joint name.

        Returns:
            The applied effort per DOF name.
        """
        articulation = self._cartpole.articulation
        names = list(articulation.dof_names)
        efforts = np.asarray(articulation.get_dof_efforts().numpy(), dtype=np.float32).reshape(-1)
        return {name: float(efforts[i]) for i, name in enumerate(names)}

    def _run_direct_physx_rollout(self, physics_dt: float) -> None:
        physx_sim = omni.physx.get_physx_simulation_interface()
        max_physics_steps = self._target_policy_steps * self._decimation
        for _ in range(max_physics_steps):
            if len(self._actual["t"]) >= self._target_policy_steps:
                break
            self._step_policy(physics_dt)
            physx_sim.simulate(physics_dt, 0.0)
            physx_sim.fetch_results()

    def _on_physics_step(self, step_size: float, context: object) -> None:
        self._step_policy(step_size)

    def _step_policy(self, step_size: float) -> None:
        if self._cartpole is None:
            return
        if len(self._actual["t"]) >= self._target_policy_steps:
            return

        if self._physics_step % self._decimation == 0:
            # Match the Isaac Lab golden sampling convention: state before the policy tick,
            # action/effort computed for that same tick.
            pos, vel = self._read_joint_state()
            self._cartpole.step(step_size)
            # The raw pre-affine model output lives on the controller's feedback buffer; the
            # runner deliberately exposes no accessor for it, so the golden reads it directly.
            action = self._cartpole._controller._last_action
            efforts = self._read_efforts()

            step = len(self._actual["t"])
            self._actual["t"].append(step * self._policy_dt)
            self._actual["cart_pos"].append(pos["slider_to_cart"])
            self._actual["pole_pos"].append(pos["cart_to_pole"])
            self._actual["cart_vel"].append(vel["slider_to_cart"])
            self._actual["pole_vel"].append(vel["cart_to_pole"])
            self._actual["action"].append(float(action[0]))
            self._actual["effort"].append(efforts["slider_to_cart"])
        else:
            self._cartpole.step(step_size)

        self._physics_step += 1


class TestCartpoleIsaacLabCpuTransfer(TestCartpoleIsaacLabTransfer):
    """Verify CPU Isaac Sim Cartpole rollouts match Isaac Lab CPU-device golden trajectories."""

    DEVICE = "cpu"
    EXPECTED_GOLDEN_DEVICE = "cpu"
    EXPECTED_ONNX_PROVIDER = "CPUExecutionProvider"
