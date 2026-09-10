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

"""Test for Newton hydroelastic contact integration."""

import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.physics.newton as newton_ext
import newton
import numpy as np
import omni.kit.app
import omni.kit.test
import omni.physics.tensors as physics_tensors
import omni.timeline
import omni.usd
import warp as wp
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.physics.newton import (
    CollisionConfig,
    HydroelasticConfig,
    NewtonConfig,
    configure_newton,
    get_newton_config,
)
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def _apply_newton_hydroelastic_sdf(
    prim: Usd.Prim,
    *,
    enabled: bool = True,
    stiffness: float = 1.0e7,
    max_resolution: int = 64,
) -> None:
    prim.AddAppliedSchema("NewtonSDFCollisionAPI")
    prim.CreateAttribute("newton:hydroelasticEnabled", Sdf.ValueTypeNames.Bool, custom=True).Set(enabled)
    prim.CreateAttribute("newton:hydroelasticStiffness", Sdf.ValueTypeNames.Float, custom=True).Set(stiffness)
    prim.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(max_resolution)


def _extract_hydroelastic_contact_forces(
    contacts: newton.Contacts, model: newton.Model, state: newton.State
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return world-frame data for active hydroelastic penalty contacts.

    Args:
        contacts: Contact buffers to inspect.
        model: Newton model that owns the contact shapes.
        state: Newton state containing body transforms.

    Returns:
        World-space points on shape 0, world-space points on shape 1, contact normals, and penalty-force
        magnitudes for active contacts.
    """
    n = int(contacts.rigid_contact_count.numpy()[0])
    empty = np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
    if n == 0 or contacts.rigid_contact_stiffness is None:
        return empty

    normals = contacts.rigid_contact_normal.numpy()[:n]
    p0 = contacts.rigid_contact_point0.numpy()[:n]
    p1 = contacts.rigid_contact_point1.numpy()[:n]
    stiffness = contacts.rigid_contact_stiffness.numpy()[:n]
    shape0 = contacts.rigid_contact_shape0.numpy()[:n]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n]
    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy()

    b0 = shape_body[shape0]
    b1 = shape_body[shape1]
    off0 = np.where((b0 != -1)[:, None], body_q[np.maximum(b0, 0), :3], 0.0)
    off1 = np.where((b1 != -1)[:, None], body_q[np.maximum(b1, 0), :3], 0.0)
    p0w = p0 + off0
    p1w = p1 + off1
    depth = np.einsum("ij,ij->i", p0w - p1w, -normals) / 2.0
    mask = (stiffness > 0) & (depth < 0)
    if not np.any(mask):
        return empty

    force_mag = stiffness[mask] * (-depth[mask])
    return p0w[mask], p1w[mask], normals[mask], force_mag


def _compute_hydroelastic_net_force(contacts: newton.Contacts, model: newton.Model, state: newton.State) -> np.ndarray:
    """Compute net force from hydroelastic penalty contacts.

    Args:
        contacts: Contact buffers to inspect.
        model: Newton model that owns the contact shapes.
        state: Newton state containing body transforms.

    Returns:
        Net contact force in world coordinates.
    """
    _, _, normals, force_mag = _extract_hydroelastic_contact_forces(contacts, model, state)
    if len(force_mag) == 0:
        return np.zeros(3)
    return np.sum(force_mag[:, None] * (-normals), axis=0)


class TestNewtonHydroelasticConfig(omni.kit.test.AsyncTestCase):
    """Tests for configure_newton() runtime configuration."""

    async def setUp(self) -> None:
        """Method called to prepare the test fixture."""
        super().setUp()
        self._saved_cfg = get_newton_config()

    async def tearDown(self) -> None:
        """Method called immediately after the test method has been called."""
        if self._saved_cfg is not None:
            configure_newton(self._saved_cfg)
        super().tearDown()

    # --------------------------------------------------------------------
    async def test_configure_newton_updates_collision_config(self) -> None:
        """Test configure_newton() replaces collision_cfg on the live Newton stage."""
        cfg = NewtonConfig(
            collision_cfg=CollisionConfig(
                broad_phase="explicit",
                hydroelastic=HydroelasticConfig(buffer_fraction=0.3, mc_edge_clamp_min=0.0),
            ),
        )
        configure_newton(cfg)

        live = get_newton_config()
        self.assertIsNotNone(live)
        self.assertAlmostEqual(live.collision_cfg.hydroelastic.buffer_fraction, 0.3)
        self.assertAlmostEqual(live.collision_cfg.hydroelastic.mc_edge_clamp_min, 0.0)
        self.assertEqual(live.collision_cfg.broad_phase, "explicit")


class TestNewtonHydroelasticIntegration(omni.kit.test.AsyncTestCase):
    """Tests for hydroelastic pipeline wiring and contact forces through SimulationManager."""

    async def setUp(self) -> None:
        """Method called to prepare the test fixture."""
        super().setUp()
        if not wp.get_device().is_cuda:
            self.skipTest("Hydroelastic SDF tests require a CUDA device")

        await stage_utils.create_new_stage_async()
        self.stage = stage_utils.get_current_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self._saved_cfg = get_newton_config()

    async def tearDown(self) -> None:
        """Method called immediately after the test method has been called."""
        self.timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        if self._saved_cfg is not None:
            configure_newton(self._saved_cfg)

        await omni.usd.get_context().close_stage_async()
        super().tearDown()

    async def _create_physics_scene(self) -> None:
        """Helper to create a physics scene for tests that need it."""
        scene_prim = stage_utils.define_prim("/World/PhysicsScene")
        UsdPhysics.Scene.Define(self.stage, scene_prim.GetPath())
        scene = UsdPhysics.Scene(scene_prim)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(9.81)
        SimulationManager.set_default_physics_scene("/World/PhysicsScene")
        await omni.kit.app.get_app().next_update_async()

    def _create_hydroelastic_cube(
        self,
        path: str,
        center_z: float,
        *,
        hydroelastic: bool = True,
        mass: float = 1.0,
    ) -> None:
        """Add a rigid body with a box collision shape.

        Args:
            path: USD path of the rigid body prim.
            center_z: World-space height of the rigid body center.
            hydroelastic: Whether to enable hydroelastic collision.
            mass: Mass of the rigid body in kilograms.
        """
        body = self.stage.DefinePrim(path, "Xform")
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(mass)
        UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, center_z))

        collision_path = f"{path}/Collision"
        cube = UsdGeom.Cube.Define(self.stage, collision_path)
        cube.GetSizeAttr().Set(0.5)
        coll_prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(coll_prim)

        if hydroelastic:
            _apply_newton_hydroelastic_sdf(coll_prim)

    def _create_ground(self) -> None:
        """Helper to add a large static box collider with its top face at z=0."""
        ground = UsdGeom.Cube.Define(self.stage, "/World/Ground")
        ground.GetSizeAttr().Set(1.0)
        prim = ground.GetPrim()
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.5))
        xform.AddScaleOp().Set(Gf.Vec3f(10.0, 10.0, 1.0))
        UsdPhysics.CollisionAPI.Apply(prim)

    async def _build_stacked_cube_scene(self) -> None:
        """Build two stacked hydroelastic cubes in contact on a static ground."""
        await self._create_physics_scene()
        self._create_ground()
        self._create_hydroelastic_cube("/World/Cube0", center_z=0.25)
        self._create_hydroelastic_cube("/World/Cube1", center_z=0.75)
        await omni.kit.app.get_app().next_update_async()

    def _assert_model_has_hydroelastic_shapes(self) -> None:
        """Assert the loaded Newton model contains at least one hydroelastic shape."""
        flags = self.newton_stage.model.shape_flags.numpy()
        has_hydro = bool((flags & int(newton.ShapeFlags.HYDROELASTIC)).any())
        self.assertTrue(
            has_hydro,
            "Expected at least one hydroelastic shape after USD parse; " f"shape_flags={flags.tolist()}",
        )

    async def _play(self, cfg: NewtonConfig) -> None:
        """Apply runtime configuration, start playback, and wait for Newton initialization.

        Args:
            cfg: Newton configuration to apply before playback.
        """
        configure_newton(cfg)
        self.timeline.play()
        self.newton_stage = newton_ext.acquire_stage()
        self.assertIsNotNone(self.newton_stage)

        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()
            if self.newton_stage.initialized:
                break

        self.assertTrue(self.newton_stage.initialized, "Newton stage failed to initialize")

    # --------------------------------------------------------------------
    async def test_collision_pipeline_enables_hydroelastic_for_hydro_shapes(self) -> None:
        """Test hydroelastic shapes with enabled runtime config build a hydroelastic pipeline."""
        await self._build_stacked_cube_scene()
        await self._play(
            NewtonConfig(
                collision_cfg=CollisionConfig(
                    hydroelastic=HydroelasticConfig(enabled=True, buffer_fraction=1.0),
                ),
            )
        )

        self.assertIsNotNone(self.newton_stage.collision_pipeline)
        self._assert_model_has_hydroelastic_shapes()
        narrow = self.newton_stage.collision_pipeline.narrow_phase
        self.assertIsNotNone(narrow.hydroelastic_sdf)

    async def test_collision_pipeline_applies_contact_capacity_config(self) -> None:
        """Test rigid_contact_max and the hydroelastic buffer multipliers reach the pipeline."""
        await self._build_stacked_cube_scene()
        await self._play(
            NewtonConfig(
                collision_cfg=CollisionConfig(
                    rigid_contact_max=4096,
                    hydroelastic=HydroelasticConfig(enabled=True, buffer_mult_iso=2),
                ),
            )
        )

        pipeline = self.newton_stage.collision_pipeline
        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.contacts().rigid_contact_max, 4096)
        self.assertEqual(pipeline.narrow_phase.hydroelastic_sdf.config.buffer_mult_iso, 2)

    async def test_collision_pipeline_respects_hydroelastic_disabled_config(self) -> None:
        """Test hydroelastic.enabled=False skips building the hydroelastic path."""
        await self._build_stacked_cube_scene()
        await self._play(
            NewtonConfig(
                collision_cfg=CollisionConfig(
                    hydroelastic=HydroelasticConfig(enabled=False),
                ),
            )
        )

        self.assertIsNotNone(self.newton_stage.collision_pipeline)
        self._assert_model_has_hydroelastic_shapes()
        narrow = self.newton_stage.collision_pipeline.narrow_phase
        self.assertIsNone(narrow.hydroelastic_sdf)

    async def test_hydroelastic_pair_detected_after_collide(self) -> None:
        """Test two hydroelastic bodies in contact route at least one sdf-sdf pair."""
        await self._build_stacked_cube_scene()
        await self._play(
            NewtonConfig(
                collision_cfg=CollisionConfig(
                    hydroelastic=HydroelasticConfig(enabled=True, reduce_contacts=True),
                ),
            )
        )

        self._assert_model_has_hydroelastic_shapes()
        pipeline = self.newton_stage.collision_pipeline
        self.assertIsNotNone(pipeline)

        pipeline.collide(self.newton_stage.state_0, self.newton_stage.contacts)
        wp.synchronize()

        sdf_sdf_count = int(pipeline.narrow_phase.shape_pairs_sdf_sdf_count.numpy()[0])
        self.assertGreaterEqual(
            sdf_sdf_count,
            1,
            f"Expected at least one hydroelastic sdf-sdf pair, got {sdf_sdf_count}",
        )

    async def test_hydroelastic_penalty_normal_force_at_initial_contact(self) -> None:
        """Test hydroelastic collide produces a vertical repulsive normal force between cubes."""
        await self._build_stacked_cube_scene()
        await self._play(
            NewtonConfig(
                collision_cfg=CollisionConfig(
                    hydroelastic=HydroelasticConfig(enabled=True, reduce_contacts=True),
                ),
            )
        )

        self._assert_model_has_hydroelastic_shapes()
        pipeline = self.newton_stage.collision_pipeline
        pipeline.collide(self.newton_stage.state_0, self.newton_stage.contacts)

        net_force = _compute_hydroelastic_net_force(
            self.newton_stage.contacts,
            self.newton_stage.model,
            self.newton_stage.state_0,
        )
        _, _, normals, force_mag = _extract_hydroelastic_contact_forces(
            self.newton_stage.contacts,
            self.newton_stage.model,
            self.newton_stage.state_0,
        )

        self.assertGreater(len(force_mag), 0, "Expected active hydroelastic contacts")
        self.assertGreater(abs(net_force[2]), 0.0, f"Expected nonzero normal force, got {net_force}")
        xy_norm = np.linalg.norm(normals[:, :2], axis=1)
        self.assertLess(xy_norm.max(), 0.2, "Contact normals should be mostly vertical")
        self.assertGreater(np.abs(normals[:, 2]).min(), 0.8)

    async def test_hydroelastic_runtime_net_force_supports_upper_cube_weight(self) -> None:
        """Test simulated hydroelastic contact forces support the upper cube under gravity."""
        await self._build_stacked_cube_scene()
        await self._play(
            NewtonConfig(
                collision_cfg=CollisionConfig(
                    hydroelastic=HydroelasticConfig(enabled=True, reduce_contacts=True),
                ),
            )
        )

        sim = physics_tensors.create_simulation_view("warp", backend="newton", stage_id=-1)
        contact_view = sim.create_rigid_contact_view("/World/Cube1", filter_patterns=["/World/Cube0"])
        body_view = sim.create_rigid_body_view("/World/Cube1")

        for _ in range(120):
            await omni.kit.app.get_app().next_update_async()

        physics_dt = SimulationManager.get_physics_dt()
        net_forces = contact_view.get_net_contact_forces(dt=physics_dt).numpy()
        net_force = net_forces[0]

        expected_weight = 1.0 * 9.81
        support_force = abs(net_force[2])
        self.assertGreater(
            support_force,
            0.5 * expected_weight,
            f"Expected support force magnitude, got net force {net_force}",
        )
        self.assertLess(
            abs(support_force - expected_weight),
            0.5 * expected_weight,
            f"Support force {support_force:.2f} N far from weight {expected_weight:.2f} N",
        )

        transforms = body_view.get_transforms().numpy()
        cube1_z = transforms[0, 2]
        self.assertGreater(cube1_z, 0.55, f"Upper cube fell through stack, z={cube1_z:.3f}")
