# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Verify Isaac Randomizers script-editor snippets and their BasicWriter output."""

import os
import tempfile
from typing import Any

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit
import omni.usd
from isaacsim.test.utils.file_validation import validate_folder_contents
from isaacsim.test.utils.image_comparison import compare_images_in_directories


class TestSDGRandomizerSnippets(omni.kit.test.AsyncTestCase):
    """Runs Isaac Randomizers script-editor snippets and validates captured output files."""

    RGB_MEAN_DIFF_TOLERANCE = 5
    GOLDEN_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "golden")

    async def setUp(self) -> None:
        """Create a clean stage before running randomizer snippets."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()

    async def tearDown(self) -> None:
        """Close the stage and wait for pending loads before the next snippet test."""
        stage_utils.close_stage()
        await app_utils.update_app_async()
        # In some cases the test will end before the asset is loaded, in this case wait for assets to load
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await app_utils.update_app_async()

    async def test_sdg_randomizer_light_sources(self) -> None:
        """Run the light-source randomizer snippet and validate BasicWriter RGB output."""
        import asyncio

        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        import omni.replicator.core as rep

        out_dir = tempfile.mkdtemp(prefix="test_rand_lights_")
        print(f"Output directory: {out_dir}")

        golden_dir = os.path.join(self.GOLDEN_ROOT, "_out_rand_lights")

        NUM_FRAMES = 10
        NUM_LIGHTS = 10
        WRITE_DATA = True
        DELAY = 0.2
        SEED = 42

        async def run_randomizations_async(
            num_frames: int,
            lights: list[Any],
            write_data: bool = True,
            delay: float | None = None,
            rng: Any | None = None,
        ) -> None:
            if rng is None:
                rng = rep.rng.ReplicatorRNG(seed=SEED)
            gen = rng.generator
            if write_data:
                print(f"Writing data to {out_dir}..")
                backend = rep.backends.get("DiskBackend")
                backend.initialize(output_dir=out_dir)
                writer = rep.WriterRegistry.get("BasicWriter")
                writer.initialize(backend=backend, rgb=True)
                cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), name="Camera")
                rp = rep.create.render_product(cam, resolution=(512, 512))
                writer.attach(rp)

            for _ in range(num_frames):
                for light in lights:
                    rep.functional.modify.pose(
                        light,
                        position_value=(
                            gen.uniform(-5, 5),
                            gen.uniform(-5, 5),
                            gen.uniform(4, 6),
                        ),
                        scale_value=gen.uniform(0.5, 1.5),
                    )
                    rep.functional.modify.attribute(light, "inputs:colorTemperature", gen.normal(4500, 1500))
                    rep.functional.modify.attribute(light, "inputs:intensity", gen.normal(25000, 5000))
                    rep.functional.modify.attribute(
                        light,
                        "inputs:color",
                        (
                            gen.uniform(0.1, 0.9),
                            gen.uniform(0.1, 0.9),
                            gen.uniform(0.1, 0.9),
                        ),
                    )

                if write_data:
                    await rep.orchestrator.step_async(rt_subframes=16)
                else:
                    await app_utils.update_app_async()
                if delay is not None and delay > 0:
                    await asyncio.sleep(delay)

            if write_data:
                await rep.orchestrator.wait_until_complete_async()
                writer.detach()
                rp.destroy()

        async def run_example_async(
            num_frames: int,
            num_lights: int,
            write_data: bool,
            delay: float | None = None,
            rng: Any | None = None,
        ) -> None:
            if rng is None:
                rep.set_global_seed(SEED)
                rng = rep.rng.ReplicatorRNG(seed=SEED)
            await stage_utils.create_new_stage_async()
            rep.functional.create.xform(name="World")
            rep.functional.create.sphere(
                parent="/World", name="Sphere", position=(0.0, 1.0, 1.0), semantics={"class": "sphere"}
            )
            rep.functional.create.cube(
                parent="/World", name="Cube", position=(0.0, -2.0, 2.0), semantics={"class": "cube"}
            )
            rep.functional.create.plane(parent="/World", name="Plane", scale=(10, 10, 1))
            rep.functional.create.scope(name="Lights", parent="/World")
            lights = rep.functional.create_batch.sphere_light(
                count=num_lights,
                parent="/World/Lights",
                enable_color_temperature=True,
                radius=0.5,
            )
            await run_randomizations_async(
                num_frames=num_frames, lights=lights, write_data=write_data, delay=delay, rng=rng
            )

        # Run the test
        test_num_frames = 3
        test_delay = None
        await run_example_async(test_num_frames, NUM_LIGHTS, WRITE_DATA, delay=test_delay)

        folder_contents_success = validate_folder_contents(
            path=out_dir, expected_counts={"png": test_num_frames}, recursive=True
        )
        self.assertTrue(folder_contents_success, f"Output directory contents validation failed for {out_dir}")
        rgb_result = compare_images_in_directories(
            golden_dir=golden_dir,
            test_dir=out_dir,
            path_pattern=r"^rgb_.*\.png$",
            allclose_rtol=None,
            allclose_atol=None,
            mean_tolerance=self.RGB_MEAN_DIFF_TOLERANCE,
            print_all_stats=False,
        )
        self.assertTrue(
            rgb_result["all_passed"],
            f"RGB image comparison failed (tol={self.RGB_MEAN_DIFF_TOLERANCE}). "
            f"Golden dir: {golden_dir}, output dir: {out_dir}",
        )

    async def test_sdg_randomizer_textures(self) -> None:
        """Run the texture randomizer snippet and validate BasicWriter RGB output."""
        import asyncio

        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        import omni.replicator.core as rep
        from isaacsim.storage.native import get_assets_root_path_async
        from pxr import UsdShade

        out_dir = tempfile.mkdtemp(prefix="test_rand_textures_")
        print(f"Output directory: {out_dir}")

        golden_dir = os.path.join(self.GOLDEN_ROOT, "_out_rand_textures")

        NUM_FRAMES = 10
        NUM_CUBES = 10
        WRITE_DATA = True
        DELAY = 0.2
        SEED = 42

        async def run_randomizations_async(
            num_frames: int,
            shapes: list[Any],
            materials: list[Any],
            write_data: bool = True,
            delay: float | None = None,
            rng: Any | None = None,
        ) -> None:
            if rng is None:
                rng = rep.rng.ReplicatorRNG(seed=SEED)
            gen = rng.generator
            assets_root_path = await get_assets_root_path_async()
            textures = [
                assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/aggregate_exposed_diff.jpg",
                assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_diff.jpg",
                assets_root_path
                + "/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_multi_R_rough_G_ao.jpg",
                assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/rough_gravel_rough.jpg",
            ]

            if write_data:
                print(f"Writing data to {out_dir}..")
                backend = rep.backends.get("DiskBackend")
                backend.initialize(output_dir=out_dir)
                writer = rep.WriterRegistry.get("BasicWriter")
                writer.initialize(backend=backend, rgb=True)
                cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), name="Camera")
                rp = rep.create.render_product(cam, resolution=(512, 512))
                writer.attach(rp)

            initial_materials = {}
            for i, shape in enumerate(shapes):
                cur_mat, _ = UsdShade.MaterialBindingAPI(shape).ComputeBoundMaterial()
                initial_materials[shape] = cur_mat
                rep.functional.modify.material(shape, materials[i])

            for _ in range(num_frames):
                for mat in materials:
                    rep.functional.modify.attribute(mat, "inputs:diffuse_texture", gen.choice(textures))
                    project_uvw = gen.choice([True, False], p=[0.9, 0.1])
                    rep.functional.modify.attribute(mat, "inputs:project_uvw", bool(project_uvw))
                    texture_scale = gen.uniform(0.1, 1)
                    rep.functional.modify.attribute(mat, "inputs:texture_scale", (texture_scale, texture_scale))
                    rep.functional.modify.attribute(mat, "inputs:texture_rotate", gen.uniform(0, 45))

                if write_data:
                    await rep.orchestrator.step_async(rt_subframes=16)
                else:
                    await app_utils.update_app_async()
                if delay is not None and delay > 0:
                    await asyncio.sleep(delay)

            if write_data:
                await rep.orchestrator.wait_until_complete_async()
                writer.detach()
                rp.destroy()

            for shape, mat in initial_materials.items():
                if mat:
                    rep.functional.modify.material(shape, mat.GetPrim())
                else:
                    UsdShade.MaterialBindingAPI(shape).UnbindAllBindings()

        async def run_example_async(
            num_frames: int,
            num_cubes: int,
            write_data: bool,
            delay: float | None = None,
            rng: Any | None = None,
        ) -> None:
            if rng is None:
                rep.set_global_seed(SEED)
                rng = rep.rng.ReplicatorRNG(seed=SEED)
            gen = rng.generator
            await stage_utils.create_new_stage_async()
            rep.functional.create.xform(name="World")
            rep.functional.create.scope(name="Looks", parent="/World")
            rep.functional.create.dome_light(intensity=1000, parent="/World")

            sphere = rep.functional.create.sphere(
                parent="/World", name="Sphere", position=(0.0, 0.0, 1.0), semantics={"class": "sphere"}
            )
            cubes = rep.functional.create_batch.cube(count=num_cubes, parent="/World", semantics={"class": "cube"})
            for cube in cubes:
                scale_rand = gen.uniform(0.25, 0.5)
                rep.functional.modify.pose(
                    cube,
                    position_value=(gen.uniform(-3.5, 3.5), gen.uniform(-3.5, 3.5), 1),
                    scale_value=scale_rand,
                )
            rep.functional.create.plane(parent="/World", name="Plane", scale=(10, 10, 1))

            shapes = [sphere] + list(cubes)
            materials = rep.functional.create_batch.material(
                mdl="OmniPBR.mdl", count=len(shapes), parent="/World/Looks"
            )
            await run_randomizations_async(num_frames, shapes, materials, write_data, delay, rng=rng)

        # Run the test
        test_num_frames = 3
        test_delay = None
        await run_example_async(test_num_frames, NUM_CUBES, WRITE_DATA, delay=test_delay)

        folder_contents_success = validate_folder_contents(
            path=out_dir, expected_counts={"png": test_num_frames}, recursive=True
        )
        self.assertTrue(folder_contents_success, f"Output directory contents validation failed for {out_dir}")
        rgb_result = compare_images_in_directories(
            golden_dir=golden_dir,
            test_dir=out_dir,
            path_pattern=r"^rgb_.*\.png$",
            allclose_rtol=None,
            allclose_atol=None,
            mean_tolerance=self.RGB_MEAN_DIFF_TOLERANCE,
            print_all_stats=False,
        )
        self.assertTrue(
            rgb_result["all_passed"],
            f"RGB image comparison failed (tol={self.RGB_MEAN_DIFF_TOLERANCE}). "
            f"Golden dir: {golden_dir}, output dir: {out_dir}",
        )

    async def test_sdg_randomizer_sequential_randomizations(self) -> None:
        """Run the sequential sphere-scan snippet and validate dual-view BasicWriter output."""
        import asyncio
        import itertools

        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.bounds as bounds_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        import isaacsim.core.experimental.utils.transform as transform_utils
        import isaacsim.core.experimental.utils.xform as xform_utils
        import numpy as np
        import omni.replicator.core as rep
        from isaacsim.storage.native import get_assets_root_path_async

        out_dir = tempfile.mkdtemp(prefix="test_rand_sphere_scan_")
        print(f"Output directory: {out_dir}")

        golden_dir = os.path.join(self.GOLDEN_ROOT, "_out_rand_sphere_scan")

        NUM_FRAMES = 90
        FORKLIFT_PATH = "/Isaac/Props/Forklift/forklift.usd"
        PALLET_PATH = "/Isaac/Props/Pallet/pallet.usd"
        BIN_PATH = "/Isaac/Props/KLT_Bin/small_KLT_visual.usd"
        DOME_TEXTURES = [
            "/NVIDIA/Assets/Skies/Cloudy/champagne_castle_1_4k.hdr",
            "/NVIDIA/Assets/Skies/Clear/evening_road_01_4k.hdr",
            "/NVIDIA/Assets/Skies/Clear/mealie_road_4k.hdr",
            "/NVIDIA/Assets/Skies/Clear/qwantani_4k.hdr",
        ]
        WRITE_DATA = True
        DELAY = 0.2
        SEED = 42

        # Fibonacci sphere algorithm: https://arxiv.org/pdf/0912.4540
        def next_point_on_sphere(
            idx: int,
            num_points: int,
            radius: float = 1,
            origin: tuple[float, float, float] = (0, 0, 0),
        ) -> list[float]:
            offset = 2.0 / num_points
            inc = np.pi * (3.0 - np.sqrt(5.0))
            z = ((idx * offset) - 1) + (offset / 2)
            phi = ((idx + 1) % num_points) * inc
            r = np.sqrt(1 - pow(z, 2))
            y = np.cos(phi) * r
            x = np.sin(phi) * r
            return [(x * radius) + origin[0], (y * radius) + origin[1], (z * radius) + origin[2]]

        async def run_randomizations_async(
            num_frames: int,
            forklift_path: str,
            pallet_path: str,
            bin_path: str,
            dome_textures: list[str],
            write_data: bool = True,
            delay: float | None = None,
            rng: Any | None = None,
        ) -> None:
            if rng is None:
                rng = rep.rng.ReplicatorRNG(seed=SEED)
            gen = rng.generator
            assets_root_path = await get_assets_root_path_async()

            await stage_utils.create_new_stage_async()
            rep.functional.create.xform(name="World")
            rep.functional.create.scope(name="Lights", parent="/World")
            dome_light = rep.functional.create.dome_light(intensity=1000, parent="/World/Lights")

            rep.functional.create.reference(
                usd_path=assets_root_path + forklift_path,
                parent="/World",
                name="Forklift",
                position=(-4.5, -4.5, 0),
            )
            pallet = rep.functional.create.reference(
                usd_path=assets_root_path + pallet_path,
                parent="/World",
                name="Pallet",
            )
            bin_prim = rep.functional.create.reference(
                usd_path=assets_root_path + bin_path,
                parent="/World",
                name="Bin",
            )
            view_cam = rep.functional.create.camera(parent="/World", name="Camera")

            dome_textures_full = [assets_root_path + tex for tex in dome_textures]
            textures_cycle = itertools.cycle(dome_textures_full)

            if write_data:
                print(f"Writing data to {out_dir}..")
                backend = rep.backends.get("DiskBackend")
                backend.initialize(output_dir=out_dir)
                writer = rep.WriterRegistry.get("BasicWriter")
                writer.initialize(backend=backend, rgb=True)
                persp_cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), name="PerspCamera")
                rp_persp = rep.create.render_product(persp_cam, (512, 512), name="PerspView")
                rp_view = rep.create.render_product(view_cam, (512, 512), name="SphereView")
                writer.attach([rp_view, rp_persp])

            bb_cache = bounds_utils.create_bbox_cache()
            pallet_aabb = bounds_utils.compute_aabb(pallet, bbox_cache=bb_cache)
            bin_aabb = bounds_utils.compute_aabb(bin_prim, bbox_cache=bb_cache)
            pallet_size = pallet_aabb[3:6] - pallet_aabb[0:3]
            bin_size = bin_aabb[3:6] - bin_aabb[0:3]
            pallet_length = float(np.linalg.norm(pallet_size))

            for i in range(num_frames):
                if i % 5 == 0:
                    rep.functional.modify.attribute(dome_light, "inputs:texture:file", next(textures_cycle))
                    await app_utils.update_app_async()

                rand_z_rot = gen.uniform(-90, 90)
                rep.functional.modify.pose(
                    pallet,
                    position_value=(gen.uniform(-1.5, 1.5), gen.uniform(-1.5, 1.5), 0),
                    rotation_value=(0, 0, rand_z_rot),
                )
                pallet_pos, pallet_quat = xform_utils.get_world_pose(pallet)
                pallet_pos = tuple(map(float, pallet_pos.numpy()))
                pallet_euler = (
                    transform_utils.quaternion_to_euler_angles(pallet_quat.numpy(), degrees=True).numpy().flatten()
                )

                rand_transl_x = gen.uniform(-pallet_size[0] / 2 + bin_size[0] / 2, pallet_size[0] / 2 - bin_size[0] / 2)
                rand_transl_y = gen.uniform(-pallet_size[1] / 2 + bin_size[1] / 2, pallet_size[1] / 2 - bin_size[1] / 2)

                rand_z_rot_rad = np.deg2rad(rand_z_rot)
                rot_adjusted_transl_x = rand_transl_x * np.cos(rand_z_rot_rad) - rand_transl_y * np.sin(rand_z_rot_rad)
                rot_adjusted_transl_y = rand_transl_x * np.sin(rand_z_rot_rad) + rand_transl_y * np.cos(rand_z_rot_rad)
                rep.functional.modify.pose(
                    bin_prim,
                    position_value=(
                        float(pallet_pos[0] + rot_adjusted_transl_x),
                        float(pallet_pos[1] + rot_adjusted_transl_y),
                        float(pallet_pos[2] + pallet_size[2] + bin_size[2] / 2),
                    ),
                    rotation_value=tuple(map(float, pallet_euler)),
                )

                rand_radius = gen.normal(3, 0.5) * pallet_length
                bin_pos, _ = xform_utils.get_world_pose(bin_prim)
                bin_pos = tuple(map(float, bin_pos.numpy()))
                cam_pos = tuple(
                    map(float, next_point_on_sphere(i, num_points=num_frames, radius=rand_radius, origin=bin_pos))
                )
                rep.functional.modify.pose(
                    view_cam, position_value=cam_pos, look_at_value=bin_pos, look_at_up_axis=(0, 0, 1)
                )

                if write_data:
                    await rep.orchestrator.step_async(rt_subframes=8, delta_time=0.0)
                else:
                    await app_utils.update_app_async()
                if delay is not None and delay > 0:
                    await asyncio.sleep(delay)

            if write_data:
                await rep.orchestrator.wait_until_complete_async()
                writer.detach()
                rp_persp.destroy()
                rp_view.destroy()

        async def run_example_async(
            num_frames: int,
            write_data: bool,
            delay: float | None = None,
            rng: Any | None = None,
        ) -> None:
            if rng is None:
                rep.set_global_seed(SEED)
                rng = rep.rng.ReplicatorRNG(seed=SEED)
            await run_randomizations_async(
                num_frames,
                FORKLIFT_PATH,
                PALLET_PATH,
                BIN_PATH,
                DOME_TEXTURES,
                write_data=write_data,
                delay=delay,
                rng=rng,
            )

        # Run the test
        test_num_frames = 3
        test_delay = None
        await run_example_async(test_num_frames, WRITE_DATA, delay=test_delay)

        views = ("PerspView", "SphereView")
        folder_contents_success = validate_folder_contents(
            path=out_dir, expected_counts={"png": test_num_frames * len(views)}, recursive=True
        )
        self.assertTrue(folder_contents_success, f"Output directory contents validation failed for {out_dir}")
        for view in views:
            view_golden_dir = os.path.join(golden_dir, view, "rgb")
            view_out_dir = os.path.join(out_dir, view, "rgb")
            rgb_result = compare_images_in_directories(
                golden_dir=view_golden_dir,
                test_dir=view_out_dir,
                path_pattern=r"^rgb_.*\.png$",
                allclose_rtol=None,
                allclose_atol=None,
                mean_tolerance=self.RGB_MEAN_DIFF_TOLERANCE,
                print_all_stats=False,
            )
            self.assertTrue(
                rgb_result["all_passed"],
                f"RGB image comparison failed (tol={self.RGB_MEAN_DIFF_TOLERANCE}). "
                f"Golden dir: {view_golden_dir}, output dir: {view_out_dir}",
            )

    async def test_sdg_randomizer_physics_volume_filling(self) -> None:
        """Run the physics-based volume filling snippet and validate the final capture."""
        from itertools import chain

        import carb
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.bounds as bounds_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        import isaacsim.core.experimental.utils.transform as transform_utils
        import isaacsim.core.experimental.utils.xform as xform_utils
        import numpy as np
        import omni.physx
        import omni.replicator.core as rep
        from isaacsim.core.experimental.materials import RigidBodyMaterial
        from isaacsim.core.experimental.objects import Cube
        from isaacsim.core.experimental.prims import GeomPrim, XformPrim
        from isaacsim.storage.native import get_assets_root_path_async
        from pxr import PhysicsSchemaTools, UsdUtils

        NUM_PALLETS = 6
        ENV_URL = "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        WRITE_DATA = True
        SEED = 42

        def resolve_scene_root_path(stage: Any | None) -> str | None:
            if stage is None:
                return None
            for root_path in ("/World", "/Root"):
                if stage.GetPrimAtPath(root_path).IsValid():
                    return root_path
            default_prim = stage.GetDefaultPrim()
            if default_prim.IsValid():
                return str(default_prim.GetPath())
            return None

        def add_rigid_body_dynamics(
            prim: Any, disable_gravity: bool = False, angular_damping: float | None = None
        ) -> None:
            # This flow adds bodies at runtime, so author the APIs without creating a persistent tensor view.
            rigid_body_attributes = {
                "rigidBodyEnabled": True,
                "disableGravity": disable_gravity,
                "maxDepenetrationVelocity": 3.0,
                "maxLinearVelocity": float("inf"),
            }
            if angular_damping is not None:
                rigid_body_attributes["angularDamping"] = angular_damping
            rep.functional.physics.apply_rigid_body(prim, with_collider=False, **rigid_body_attributes)

        def create_asset(
            asset_url: str,
            path: str,
            *,
            translations: Any | None = None,
            orientations: Any | None = None,
            scales: Any | None = None,
        ) -> Any:
            prim_path = stage_utils.generate_next_free_path(path, prepend_default_prim=False)
            prim = stage_utils.add_reference_to_stage(usd_path=asset_url, path=prim_path)
            XformPrim(
                prim_path,
                translations=translations,
                orientations=orientations,
                scales=scales,
                reset_xform_op_properties=True,
            )
            return prim

        def create_asset_with_colliders(
            asset_url: str,
            path: str,
            *,
            translations: Any | None = None,
            orientations: Any | None = None,
            scales: Any | None = None,
        ) -> Any:
            prim = create_asset(asset_url, path, translations=translations, orientations=orientations, scales=scales)
            pallet_geom = GeomPrim(f"{prim.GetPath()}/.*", apply_collision_apis=True)
            pallet_geom.set_collision_approximations("convexHull")
            return prim

        def place_collision_walls(
            pallet_prim: Any,
            walls_root: str,
            bbox_cache: Any,
            height: float = 2,
            thickness: float = 0.3,
            material: Any | None = None,
            visible: bool = False,
        ) -> None:
            bbox_cache.Clear()
            aabb = bounds_utils.compute_aabb(pallet_prim, bbox_cache=bbox_cache, space="untransformed")
            width, depth, local_height = aabb[3:] - aabb[:3]
            mid = (aabb[:3] + aabb[3:]) * 0.5 + np.array([0.0, 0.0, local_height / 2])

            walls = [
                ("floor", (mid[0], mid[1], mid[2] - thickness / 2), (width, depth, thickness)),
                ("ceiling", (mid[0], mid[1], mid[2] + height + thickness / 2), (width, depth, thickness)),
                (
                    "left_wall",
                    (mid[0] - (width + thickness) / 2, mid[1], mid[2] + height / 2),
                    (thickness, depth, height),
                ),
                (
                    "right_wall",
                    (mid[0] + (width + thickness) / 2, mid[1], mid[2] + height / 2),
                    (thickness, depth, height),
                ),
                (
                    "front_wall",
                    (mid[0], mid[1] + (depth + thickness) / 2, mid[2] + height / 2),
                    (width, thickness, height),
                ),
                (
                    "back_wall",
                    (mid[0], mid[1] - (depth + thickness) / 2, mid[2] + height / 2),
                    (width, thickness, height),
                ),
            ]

            pallet_position, pallet_orientation = xform_utils.get_world_pose(pallet_prim)
            stage = stage_utils.get_current_stage(backend="usd")
            for name, location, size in walls:
                wall_path = f"{walls_root}/{name}"
                wall_scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
                world_pos = tuple(
                    transform_utils.transform_local_to_world(location, pallet_position, pallet_orientation).numpy()
                )
                if stage.GetPrimAtPath(wall_path).IsValid():
                    wall = XformPrim(
                        wall_path,
                        positions=world_pos,
                        orientations=pallet_orientation.numpy(),
                        scales=wall_scale,
                        reset_xform_op_properties=True,
                    )
                    wall_geom = GeomPrim(wall_path)
                else:
                    wall = Cube(
                        wall_path,
                        sizes=2.0,
                        positions=world_pos,
                        orientations=pallet_orientation.numpy(),
                        scales=wall_scale,
                        reset_xform_op_properties=True,
                    )
                    wall_geom = GeomPrim(wall_path, apply_collision_apis=True)
                    wall_geom.set_collision_approximations("convexHull")
                if material is not None:
                    wall_geom.apply_physics_materials(material, weaker_than_descendants=[True])
                wall.set_visibilities([visible])

        async def apply_forces_async(
            boxes: list[Any], pallet: Any, strength: float = 550, strength_center_multiplier: float = 2
        ) -> None:
            stage = stage_utils.get_current_stage(backend="usd")
            app_utils.play()
            pallet_center, pallet_orientation = xform_utils.get_world_pose(pallet)
            force_forward = (
                transform_utils.rotate_vectors_by_quaternion([1.0, 0.0, 0.0], pallet_orientation).numpy() * strength
            )
            force_right = (
                transform_utils.rotate_vectors_by_quaternion([0.0, 1.0, 0.0], pallet_orientation).numpy() * strength
            )

            physx_simulation_interface = omni.physx.get_physx_simulation_interface()
            stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
            directional_forces = [force_forward, force_right, -force_forward, -force_right]
            for box_prim in boxes:
                body_path = PhysicsSchemaTools.sdfPathToInt(box_prim.GetPath())
                for force in chain(directional_forces, directional_forces):
                    box_position, _ = xform_utils.get_world_pose(box_prim)
                    box_position = carb.Float3(*box_position.numpy())
                    physx_simulation_interface.apply_force_at_pos(
                        stage_id, body_path, carb.Float3(*force), box_position
                    )
                    await app_utils.update_app_async(steps=10)

            for box_prim in boxes:
                body_path = PhysicsSchemaTools.sdfPathToInt(box_prim.GetPath())
                box_location, _ = xform_utils.get_world_pose(box_prim)
                box_location = box_location.numpy()
                force_to_center = (pallet_center.numpy() - box_location) * strength * strength_center_multiplier
                physx_simulation_interface.apply_force_at_pos(
                    stage_id,
                    body_path,
                    carb.Float3(*force_to_center),
                    carb.Float3(*box_location),
                )
            await app_utils.update_app_async(steps=20)
            app_utils.pause()

        async def stack_boxes_on_pallet_async(
            pallet_prim: Any,
            walls_root: str,
            boxes_urls_and_weights: list[tuple[str, float]],
            num_boxes: int,
            drop_height: float = 1.5,
            drop_margin: float = 0.2,
            gen: Any | None = None,
        ) -> list[Any]:
            pallet_path = pallet_prim.GetPath()
            print(f"[BoxStacking] Running scenario for pallet {pallet_path} with {num_boxes} boxes..")
            bbox_cache = bounds_utils.create_bbox_cache()

            physics_material = RigidBodyMaterial(
                f"{pallet_path}/Looks/PhysicsMaterial",
                static_frictions=[0.01],
                dynamic_frictions=[0.01],
                restitutions=[0.0],
            )
            GeomPrim(f"{pallet_path}/.*").apply_physics_materials(physics_material, weaker_than_descendants=[True])

            place_collision_walls(
                pallet_prim, walls_root, bbox_cache, height=drop_height + drop_margin, material=physics_material
            )

            box_urls, box_weights = zip(*boxes_urls_and_weights)
            box_weights_arr = np.asarray(box_weights, dtype=float)
            box_weights_arr /= box_weights_arr.sum()
            rand_boxes_urls = gen.choice(box_urls, size=num_boxes, p=box_weights_arr)
            boxes = [create_asset(box_url, f"{pallet_path}_Boxes/Box_{i}") for i, box_url in enumerate(rand_boxes_urls)]
            boxes.sort(
                key=lambda box: bounds_utils.compute_bound_volume(box, bbox_cache=bbox_cache, space="local"),
                reverse=True,
            )

            spawn_midpoint, spawn_size = bounds_utils.compute_bound_range(
                pallet_prim, bbox_cache=bbox_cache, space="untransformed"
            )
            pallet_width, pallet_depth, pallet_height = spawn_size
            spawn_center = spawn_midpoint + np.array([0.0, 0.0, pallet_height / 2 + drop_height])
            spawn_width, spawn_depth = pallet_width / 2 - drop_margin, pallet_depth / 2 - drop_margin

            pallet_position, pallet_orientation = xform_utils.get_world_pose(pallet_prim)

            for box_prim in boxes:
                local_loc = spawn_center + np.array(
                    [gen.uniform(-spawn_width, spawn_width), gen.uniform(-spawn_depth, spawn_depth), 0.0]
                )
                angles = [
                    gen.choice([180, 90, 0, -90, -180]) + gen.uniform(-3, 3),
                    gen.choice([180, 90, 0, -90, -180]) + gen.uniform(-3, 3),
                    gen.choice([180, 90, 0, -90, -180]) + gen.uniform(-3, 3),
                ]
                local_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                for axis_index, angle in enumerate(angles):
                    axis_angles = [0.0, 0.0, 0.0]
                    axis_angles[axis_index] = angle
                    axis_quat = (
                        transform_utils.euler_angles_to_quaternion(axis_angles, degrees=True).numpy().astype(np.float32)
                    )
                    local_orientation = transform_utils.quaternion_multiplication(local_orientation, axis_quat).numpy()
                world_loc = transform_utils.transform_local_to_world(
                    local_loc, pallet_position, pallet_orientation
                ).numpy()
                world_orientation = transform_utils.quaternion_multiplication(
                    pallet_orientation, local_orientation
                ).numpy()
                box_path = str(box_prim.GetPath())

                XformPrim(
                    box_path,
                    positions=tuple(world_loc),
                    orientations=world_orientation,
                    reset_xform_op_properties=True,
                )
                box_geom = GeomPrim(f"{box_path}/.*", apply_collision_apis=True)
                box_geom.set_collision_approximations("convexHull")
                add_rigid_body_dynamics(box_prim, angular_damping=0.9)
                box_geom.apply_physics_materials(physics_material, weaker_than_descendants=[True])
                await app_utils.update_app_async()

                app_utils.play()
                await app_utils.update_app_async(steps=20)
                app_utils.pause()

            await apply_forces_async(boxes, pallet_prim)

            # Flush the velocity changes while simulation is still enabled. Writing velocities after disabling a body
            # causes PhysX to reject the operation, and a persistent RigidPrim view would be invalidated by the next batch.
            zero_velocities = np.zeros((len(boxes), 3), dtype=np.float32)
            rep.functional.modify.attribute(boxes, "physics:velocity", zero_velocities)
            rep.functional.modify.attribute(boxes, "physics:angularVelocity", zero_velocities)
            omni.physx.get_physx_simulation_interface().flush_changes()

            rep.functional.modify.attribute(boxes, "physics:rigidBodyEnabled", False)
            omni.physx.get_physx_simulation_interface().flush_changes()

            physics_material.set_friction_coefficients(static_frictions=[0.9], dynamic_frictions=[0.9])

            return boxes

        async def run_box_stacking_scenarios_async(
            num_pallets: int,
            env_url: str | None = None,
            write_data: bool = True,
            rng: Any | None = None,
        ) -> None:
            if rng is None:
                rep.set_global_seed(SEED)
                rng = rep.rng.ReplicatorRNG(seed=SEED)
            gen = rng.generator

            assets_root_path = await get_assets_root_path_async()

            pallets_urls_and_weights = [
                (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd", 0.25),
                (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_02.usd", 0.75),
            ]
            boxes_urls_and_weights = [
                (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd", 0.02),
                (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01.usd", 0.06),
                (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd", 0.12),
                (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd", 0.80),
            ]

            scene_root = "/World"
            if env_url is not None:
                env_path = env_url if env_url.startswith("omniverse://") else assets_root_path + env_url
                success, _stage = await stage_utils.open_stage_async(env_path)
                if not success or _stage is None:
                    carb.log_error(f"[BoxStacking] Failed to open stage: {env_path}")
                    return
                await app_utils.update_app_async()
                resolved_root = resolve_scene_root_path(_stage)
                if resolved_root is None:
                    carb.log_error(
                        "[BoxStacking] Could not find a valid scene root in the opened environment "
                        "(expected /World or /Root, or a default prim on the stage)."
                    )
                    return
                scene_root = resolved_root
            else:
                await stage_utils.create_new_stage_async(template="empty")
                rep.functional.create.scope(name="Lights", parent="/World")
                rep.functional.create.distant_light(
                    intensity=400, parent="/World/Lights", name="DistantLight", rotation=(0, 60, 0)
                )
                rep.functional.create.dome_light(intensity=500, parent="/World/Lights", name="DomeLight")
                ground_plane = rep.functional.create.plane(parent=scene_root, name="GroundPlane", scale=(10, 10, 1))
                rep.functional.physics.apply_collider(ground_plane)
                await app_utils.update_app_async()

            pallets = []
            pallets_urls, pallets_weights = zip(*pallets_urls_and_weights)
            pallet_weights_arr = np.asarray(pallets_weights, dtype=float)
            pallet_weights_arr /= pallet_weights_arr.sum()
            rand_pallet_urls = gen.choice(pallets_urls, size=num_pallets, p=pallet_weights_arr)
            custom_pallet_locations = [
                (-9.3, 5.3, 1.3),
                (-9.3, 7.3, 1.3),
                (-9.3, -0.6, 1.3),
            ]
            gen.shuffle(custom_pallet_locations)
            for i, pallet_url in enumerate(rand_pallet_urls):
                if env_url is not None:
                    if i % 2 == 0 and custom_pallet_locations:
                        rand_loc = custom_pallet_locations.pop()
                    else:
                        rand_loc = (
                            -6.5 + gen.uniform(-0.2, 0.2),
                            i * 1.75 + gen.uniform(0, 0.2),
                            gen.uniform(0, 0.2),
                        )
                else:
                    rand_loc = (
                        i * 1.5 + gen.uniform(0, 0.2),
                        gen.uniform(-0.2, 0.2),
                        0.0,
                    )
                rand_rot = (0, 0, gen.choice([180, 90, 0, -90, -180]) + gen.uniform(-15, 15))
                pallet_prim = create_asset_with_colliders(
                    pallet_url,
                    f"{scene_root}/Pallet_{i}",
                    translations=rand_loc,
                    orientations=transform_utils.euler_angles_to_quaternion(rand_rot, degrees=True).numpy(),
                )
                pallets.append(pallet_prim)

            walls_root = f"{scene_root}/_CollisionWalls"
            stage_utils.define_prim(walls_root, "Xform")

            total_boxes = []
            for pallet in pallets:
                drop_height = 1.0 if env_url is not None else 1.5
                rand_num_boxes = int(gen.integers(8, 16) if env_url is not None else gen.integers(12, 21))
                stacked_boxes = await stack_boxes_on_pallet_async(
                    pallet,
                    walls_root,
                    boxes_urls_and_weights,
                    num_boxes=rand_num_boxes,
                    drop_height=drop_height,
                    gen=gen,
                )
                total_boxes.extend(stacked_boxes)

            if stage_utils.get_current_stage(backend="usd").GetPrimAtPath(walls_root).IsValid():
                stage_utils.delete_prim(walls_root)

            rep.functional.modify.attribute(total_boxes, "physics:rigidBodyEnabled", True)
            omni.physx.get_physx_simulation_interface().flush_changes()
            app_utils.play()
            await app_utils.update_app_async(steps=200)
            app_utils.pause()

            if write_data:
                out_dir = os.path.join(os.getcwd(), "_out_box_stacking")
                print(f"Writing data to {out_dir}..")
                backend = rep.backends.get("DiskBackend")
                backend.initialize(output_dir=out_dir)
                writer = rep.WriterRegistry.get("BasicWriter")
                writer.initialize(backend=backend, rgb=True)
                cam = rep.functional.create.camera(position=(5, -5, 2), look_at=(0, 0, 0), name="PalletCamera")
                rp = rep.create.render_product(cam, resolution=(512, 512))
                writer.attach(rp)

                await rep.orchestrator.step_async(rt_subframes=8)

                await rep.orchestrator.wait_until_complete_async()
                writer.detach()
                rp.destroy()

        async def run_example_async(
            num_pallets: int, env_url: str | None, write_data: bool, rng: Any | None = None
        ) -> None:
            await run_box_stacking_scenarios_async(
                num_pallets=num_pallets, env_url=env_url, write_data=write_data, rng=rng
            )

        # Run the test
        out_dir = os.path.join(os.getcwd(), "_out_box_stacking")
        golden_dir = os.path.join(self.GOLDEN_ROOT, "_out_box_stacking_env_none")

        test_num_pallets = 1
        test_env_url = None
        test_write_data = True
        expected_png_count = 1  # one orchestrator step, independent of num_pallets
        await run_example_async(test_num_pallets, test_env_url, test_write_data)

        folder_contents_success = validate_folder_contents(
            path=out_dir, expected_counts={"png": expected_png_count}, recursive=True
        )
        self.assertTrue(folder_contents_success, f"Output directory contents validation failed for {out_dir}")
        rgb_result = compare_images_in_directories(
            golden_dir=golden_dir,
            test_dir=out_dir,
            path_pattern=r"^rgb_.*\.png$",
            allclose_rtol=None,
            allclose_atol=None,
            mean_tolerance=self.RGB_MEAN_DIFF_TOLERANCE,
            print_all_stats=False,
        )
        self.assertTrue(
            rgb_result["all_passed"],
            f"RGB image comparison failed (tol={self.RGB_MEAN_DIFF_TOLERANCE}). "
            f"Golden dir: {golden_dir}, output dir: {out_dir}",
        )

    async def test_sdg_randomizer_simready_assets(self) -> None:
        """Run the SimReady assets snippet and validate BasicWriter RGB output."""
        import carb.settings
        import isaacsim.core.experimental.utils.bounds as bounds_utils
        import numpy as np
        import omni.replicator.core as rep
        from isaacsim.core.experimental.utils.semantics import upgrade_prim_semantics_to_labels
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Sdf, Usd, UsdPhysics

        if not app_utils.is_extension_enabled("omni.simready.explorer"):
            app_utils.enable_extension("omni.simready.explorer")
        import omni.simready.explorer as sre

        out_dir = tempfile.mkdtemp(prefix="test_simready_assets_")
        print(f"Output directory: {out_dir}")

        golden_dir = os.path.join(self.GOLDEN_ROOT, "_out_simready_assets")

        NUM_SCENARIOS = 5
        SEED = 34

        def enable_simready_explorer() -> None:
            if sre.get_instance().browser_model is None:
                import omni.kit.actions.core as actions

                actions.execute_action("omni.simready.explorer", "toggle_window")

        def set_prim_variants(prim: Usd.Prim, variants: dict[str, str]) -> None:
            vsets = prim.GetVariantSets()
            for name, value in variants.items():
                vset = vsets.GetVariantSet(name)
                if vset:
                    vset.SetVariantSelection(value)

        async def search_assets_async() -> tuple[list[Any], list[Any], list[Any]]:
            tables = await sre.find_assets(["table", "furniture"])
            plates = await sre.find_assets(["plate"])
            bowls = await sre.find_assets(["bowl"])
            dishes = plates + bowls
            fruits = await sre.find_assets(["fruit"])
            vegetables = await sre.find_assets(["vegetable"])
            items = fruits + vegetables
            return tables, dishes, items

        async def run_simready_randomization_async(
            stage: Any,
            camera_prim: Any,
            render_product: Any,
            tables: list[Any],
            dishes: list[Any],
            items: list[Any],
            rng: Any,
        ) -> None:
            root_layer = stage.GetRootLayer()
            variation_layer = Sdf.Layer.CreateAnonymous("variation")
            root_layer.subLayerPaths.insert(0, variation_layer.identifier)
            stage.SetEditTarget(variation_layer)

            variants = {"PhysicsVariant": "RigidBody"}
            rep.functional.create.scope(name="Assets")

            table_asset = tables[rng.integers(len(tables))]
            table_prim = rep.functional.create.reference(
                usd_path=table_asset.main_url, parent="/Assets", name=table_asset.name
            )
            set_prim_variants(table_prim, variants)
            upgrade_prim_semantics_to_labels(table_prim)
            await app_utils.update_app_async()
            UsdPhysics.RigidBodyAPI(table_prim).GetRigidBodyEnabledAttr().Set(False)

            bbox_cache = bounds_utils.create_bbox_cache()
            table_aabb = bounds_utils.compute_aabb(table_prim, bbox_cache=bbox_cache, space="world")
            table_extent = table_aabb[3:] - table_aabb[:3]

            dish_asset = dishes[rng.integers(len(dishes))]
            dish_prim = rep.functional.create.reference(
                usd_path=dish_asset.main_url, parent="/Assets", name=dish_asset.name
            )
            set_prim_variants(dish_prim, variants)
            upgrade_prim_semantics_to_labels(dish_prim)
            await app_utils.update_app_async()
            dish_aabb = bounds_utils.compute_aabb(dish_prim, bbox_cache=bbox_cache, space="world")
            dish_extent = dish_aabb[3:] - dish_aabb[:3]

            center_region_scale = 0.75
            dish_range_x = max(0, (table_extent[0] - dish_extent[0]) / 2 * center_region_scale)
            dish_range_y = max(0, (table_extent[1] - dish_extent[1]) / 2 * center_region_scale)
            dish_position = (
                rng.uniform(-dish_range_x, dish_range_x) if dish_range_x > 0 else 0,
                rng.uniform(-dish_range_y, dish_range_y) if dish_range_y > 0 else 0,
                table_extent[2] + dish_extent[2] / 2,
            )
            rep.functional.modify.pose(dish_prim, position_value=dish_position)

            num_items = rng.integers(2, 5)
            item_prims = []
            for _ in range(num_items):
                item_asset = items[rng.integers(len(items))]
                item_prim = rep.functional.create.reference(
                    usd_path=item_asset.main_url, parent="/Assets", name=item_asset.name
                )
                set_prim_variants(item_prim, variants)
                upgrade_prim_semantics_to_labels(item_prim)
                item_prims.append(item_prim)
                await app_utils.update_app_async()

            stack_height = dish_position[2]
            item_scatter_radius = max(0, dish_extent[0] / 4)
            for item_prim in item_prims:
                item_aabb = bounds_utils.compute_aabb(item_prim, bbox_cache=bbox_cache, space="world")
                item_extent = item_aabb[3:] - item_aabb[:3]
                scatter_x = rng.uniform(-item_scatter_radius, item_scatter_radius) if item_scatter_radius > 0 else 0
                scatter_y = rng.uniform(-item_scatter_radius, item_scatter_radius) if item_scatter_radius > 0 else 0
                item_position = (
                    dish_position[0] + scatter_x,
                    dish_position[1] + scatter_y,
                    stack_height + item_extent[2] / 2,
                )
                rep.functional.modify.pose(item_prim, position_value=item_position)
                stack_height += item_extent[2]

            SimulationManager.invalidate_physics()
            SimulationManager.initialize_physics()
            SimulationManager.step(steps=25)
            stage.SetEditTarget(root_layer)

            camera_position = (
                dish_position[0] + rng.uniform(-0.5, 0.5),
                dish_position[1] + rng.uniform(-0.5, 0.5),
                dish_position[2] + 1.5 + rng.uniform(-0.5, 0.5),
            )
            rep.functional.modify.pose(
                camera_prim, position_value=camera_position, look_at_value=dish_prim, look_at_up_axis=(0, 0, 1)
            )
            render_product.hydra_texture.set_updates_enabled(True)
            await rep.orchestrator.step_async(delta_time=0.0, rt_subframes=16)
            render_product.hydra_texture.set_updates_enabled(False)

            variation_layer.Clear()
            root_layer.subLayerPaths.remove(variation_layer.identifier)

        async def run_simready_randomizations_async(num_scenarios: int) -> None:
            stage = await stage_utils.create_new_stage_async()
            rng = np.random.default_rng(SEED)
            rep.set_global_seed(SEED)
            rep.orchestrator.set_capture_on_play(False)
            SimulationManager.set_physics_dt(1.0 / 60.0)
            carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

            rep.functional.create.xform(name="World")
            rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
            rep.functional.create.distant_light(
                intensity=2500, parent="/World", name="DistantLight", rotation=(-75, 0, 0)
            )

            enable_simready_explorer()
            tables, dishes, items = await search_assets_async()

            backend = rep.backends.get("DiskBackend")
            backend.initialize(output_dir=out_dir)
            writer = rep.writers.get("BasicWriter")
            writer.initialize(backend=backend, rgb=True)
            camera_prim = rep.functional.create.camera(
                position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera"
            )
            rp = rep.create.render_product(camera_prim, (512, 512))
            rp.hydra_texture.set_updates_enabled(False)
            writer.attach(rp)

            for _ in range(num_scenarios):
                await run_simready_randomization_async(stage, camera_prim, rp, tables, dishes, items, rng)

            await rep.orchestrator.wait_until_complete_async()
            writer.detach()
            rp.destroy()

        # Run the test
        test_num_scenarios = 2
        await run_simready_randomizations_async(test_num_scenarios)

        folder_contents_success = validate_folder_contents(
            path=out_dir, expected_counts={"png": test_num_scenarios}, recursive=True
        )
        self.assertTrue(folder_contents_success, f"Output directory contents validation failed for {out_dir}")
        mean_tolerance = 7.5
        rgb_result = compare_images_in_directories(
            golden_dir=golden_dir,
            test_dir=out_dir,
            path_pattern=r"^rgb_.*\.png$",
            allclose_rtol=None,
            allclose_atol=None,
            mean_tolerance=mean_tolerance,
            print_all_stats=False,
        )
        self.assertTrue(
            rgb_result["all_passed"],
            f"RGB image comparison failed (tol={mean_tolerance}). " f"Golden dir: {golden_dir}, output dir: {out_dir}",
        )

    async def test_sdg_randomizer_object_reconstruction_assets(self) -> None:
        """Drop a 3D object-reconstructed asset in sequential waves and validate the captured output."""
        import random

        import carb.settings
        import numpy as np
        import omni.replicator.core as rep
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.storage.native import get_assets_root_path_async

        out_dir = tempfile.mkdtemp(prefix="test_object_reconstruction_assets_drop_")
        print(f"Output directory: {out_dir}")

        golden_dir = os.path.join(self.GOLDEN_ROOT, "_out_object_reconstruction_assets_drop")

        # Asset produced by the NVIDIA 3D Object Reconstruction workflow (https://github.com/NVIDIA/3DObjectReconstruction)
        TOASTER_PASTRY_USD = "/Isaac/Samples/Replicator/3DObjectReconstruction/toaster_pastry.usd"
        NUM_ASSETS = 24  # Total instances spawned per scenario; any remainder is released in the final wave
        NUM_WAVES = (
            2  # Number of sequential waves per scenario, each settled wave is captured before the next is released
        )
        NUM_SCENARIOS = (
            2  # Number of times the wave sequence is reset and repeated; total captures = NUM_SCENARIOS * NUM_WAVES
        )
        BASE_DROP_HEIGHT = 1.0  # Spawn height for the first wave of a scenario (m)
        HEIGHT_STEP_PER_WAVE = 0.3  # Extra spawn height per wave, giving clearance above the growing pile (m)
        WITHIN_WAVE_HEIGHT_JITTER = 0.05  # Per-asset height variance within a wave to avoid exact overlaps (m)
        SPAWN_XY_JITTER = 0.35  # Random horizontal offset +/- (m)
        FALL_SPEED_THRESHOLD = 0.3  # Linear speed above which a wave is considered actively falling (m/s)
        SETTLE_SPEED_THRESHOLD = 0.02  # Linear speed below which a wave is considered settled (m/s)
        RNG_SEED = 17  # Reproducible randomization seed
        MAX_STEPS_PER_WAVE = 50  # Maximum simulation steps to wait for a wave to settle

        async def run_example_async(num_assets: int, num_waves: int, num_scenarios: int) -> None:
            if num_waves <= 0:
                raise ValueError("num_waves must be positive.")
            if num_assets < num_waves:
                raise ValueError("num_assets must be at least num_waves.")

            assets_per_wave, remainder = divmod(num_assets, num_waves)
            if remainder:
                print(
                    f"[SDG] Asset count is not divisible by wave count; last wave takes the remainder of {remainder} "
                    f"(size {assets_per_wave + remainder})."
                )

            await stage_utils.create_new_stage_async()
            assets_root_path = await get_assets_root_path_async()
            rng = random.Random(RNG_SEED)

            # Disable capture on play, frames will be captured manually
            rep.orchestrator.set_capture_on_play(False)

            # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto))
            carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

            rep.functional.create.xform(name="World")
            rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
            rep.functional.create.scope(name="Looks", parent="/World")

            # Using a cube to avoid collision tunneling: a zero-thickness plane can let fast-falling assets pass through.
            floor = rep.functional.create.cube(parent="/World", name="Floor", scale=(5, 5, 0.05))
            rep.functional.physics.apply_collider(floor)
            floor_material = rep.functional.create.material(
                mdl="OmniPBR.mdl",
                diffuse_color_constant=(0.6, 0.6, 0.6),
                bind_prims=floor,
                parent="/World/Looks",
                name="FloorMaterial",
            )

            # Batch-create every instance in a single call instead of looping over individual create.reference calls.
            all_prims = rep.functional.create_batch.reference(
                usd_path=assets_root_path + TOASTER_PASTRY_USD,
                semantics={"class": "toaster_pastry"},
                parent="/World",
                name="ToasterPastry",
                count=num_assets,
            )
            # Every asset starts out kinematic: it ignores gravity and can be freely repositioned until its wave releases
            # it by disabling kinematic mode. A single persistent view covers every asset across all waves and scenarios.
            rep.functional.physics.apply_rigid_body(all_prims, approximation="convexHull", kinematicEnabled=True)
            all_rigid_prims = RigidPrim([prim.GetPath().pathString for prim in all_prims])

            # Replicator setup, render product is disabled by default and enabled only at capture time
            camera = rep.functional.create.camera(position=(1.5, 1.5, 1.5), look_at=(0, 0, 0), parent="/World")
            render_product = rep.create.render_product(camera, (640, 480))
            # Enable render product updates only at capture time
            render_product.hydra_texture.set_updates_enabled(False)
            backend = rep.backends.get("DiskBackend")
            backend.initialize(output_dir=out_dir)
            writer = rep.writers.get("BasicWriter")
            writer.initialize(
                backend=backend,
                rgb=True,
                distance_to_camera=True,
                colorize_depth=True,
                semantic_segmentation=True,
                colorize_semantic_segmentation=True,
            )
            writer.attach(render_product)

            # Start the simulation
            print("[SDG] Starting simulation")
            app_utils.play()

            for scenario_idx in range(num_scenarios):
                # Reset every asset for a new scenario: freeze it back to kinematic (no-op the first time) and stagger
                # each wave's spawn height for clearance above the pile the previous scenario left behind.
                rep.functional.modify.attribute(all_prims, "physics:kinematicEnabled", True)
                positions, rotations = [], []
                for i in range(num_assets):
                    wave_idx = min(i // assets_per_wave, num_waves - 1)
                    x = rng.uniform(-SPAWN_XY_JITTER, SPAWN_XY_JITTER)
                    y = rng.uniform(-SPAWN_XY_JITTER, SPAWN_XY_JITTER)
                    z = BASE_DROP_HEIGHT + wave_idx * HEIGHT_STEP_PER_WAVE + rng.uniform(0, WITHIN_WAVE_HEIGHT_JITTER)
                    positions.append((x, y, z))
                    rotations.append((rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 360)))
                rep.functional.modify.pose(all_prims, position_value=positions, rotation_value=rotations)
                # Let the kinematic targets sync to their new poses before releasing any wave, otherwise the released
                # wave inherits a large implied velocity from the pose jump instead of starting at rest.
                await app_utils.update_app_async()

                # Drop one wave at a time: release it, wait for it to settle onto the (growing) pile, then capture
                for wave_idx in range(num_waves):
                    wave_end = num_assets if wave_idx == num_waves - 1 else (wave_idx + 1) * assets_per_wave
                    wave_slice = slice(wave_idx * assets_per_wave, wave_end)
                    wave_prims = all_prims[wave_slice]
                    wave_size = wave_slice.stop - wave_slice.start

                    # Release this wave from kinematic mode so gravity takes over
                    rep.functional.modify.attribute(wave_prims, "physics:kinematicEnabled", False)
                    # Zero out the implied teleport velocity only after disabling kinematic mode, since PhysX
                    # rejects velocity writes on kinematic bodies.
                    all_rigid_prims.set_velocities(
                        linear_velocities=np.zeros((wave_size, 3)),
                        angular_velocities=np.zeros((wave_size, 3)),
                        indices=list(range(wave_slice.start, wave_slice.stop)),
                    )

                    # Step the simulation until the wave has fallen and settled
                    has_fallen = False
                    for _ in range(MAX_STEPS_PER_WAVE):
                        await app_utils.update_app_async()
                        linear_velocities, _ = all_rigid_prims.get_velocities()
                        max_speed = float(np.linalg.norm(linear_velocities.numpy()[wave_slice], axis=1).max())
                        if not has_fallen:
                            # Wait until the wave has started moving before checking for a settled state
                            if max_speed > FALL_SPEED_THRESHOLD:
                                has_fallen = True
                            continue
                        if max_speed < SETTLE_SPEED_THRESHOLD:
                            break

                    # Randomize the floor color and capture a frame of the settled wave
                    color = (rng.random(), rng.random(), rng.random())
                    rep.functional.modify.attribute(floor_material, "inputs:diffuse_color_constant", color)
                    print(
                        f"[SDG]  Scenario {scenario_idx + 1}/{num_scenarios} - Wave {wave_idx + 1}/{num_waves} settled, "
                        f"floor color -> {tuple(round(c, 2) for c in color)}"
                    )
                    render_product.hydra_texture.set_updates_enabled(True)
                    print(
                        f"[SDG]  Capturing frame {scenario_idx * num_waves + wave_idx + 1}/{num_scenarios * num_waves}"
                    )
                    await rep.orchestrator.step_async(delta_time=0.0, pause_timeline=False, rt_subframes=16)
                    render_product.hydra_texture.set_updates_enabled(False)

            # Pause the simulation and clean up resources
            total_captures = num_scenarios * num_waves
            print(f"[SDG] Simulation complete. {total_captures} frames saved to {out_dir}")
            app_utils.pause()
            await rep.orchestrator.wait_until_complete_async()
            writer.detach()
            render_product.destroy()

        # Run the test
        await run_example_async(NUM_ASSETS, NUM_WAVES, NUM_SCENARIOS)

        total_captures = NUM_SCENARIOS * NUM_WAVES
        expected_pngs = total_captures * 3  # rgb + colorized depth + colorized semantic segmentation per capture
        expected_json = total_captures  # semantic segmentation label json per capture
        folder_contents_success = validate_folder_contents(
            path=out_dir,
            recursive=True,
            expected_counts={"png": expected_pngs, "json": expected_json},
            fail_on_empty_extensions={"png", "json"},
        )
        self.assertTrue(folder_contents_success, f"Output directory contents validation failed for {out_dir}")

        # Compare every output image type against golden data, each with its own tolerance.
        comparisons = (
            ("RGB", r"^rgb_.*\.png$", 7.5),
            ("depth", r"^distance_to_camera_.*\.png$", 1.5),
            ("semantic segmentation", r"^semantic_segmentation_.*\.png$", 4.5),
        )
        for label, path_pattern, mean_diff_tolerance in comparisons:
            image_result = compare_images_in_directories(
                golden_dir=golden_dir,
                test_dir=out_dir,
                path_pattern=path_pattern,
                allclose_rtol=None,
                allclose_atol=None,
                mean_tolerance=mean_diff_tolerance,
                print_all_stats=False,
            )
            self.assertTrue(
                image_result["all_passed"],
                f"{label} image comparison failed (tol={mean_diff_tolerance}). "
                f"Golden dir: {golden_dir}, output dir: {out_dir}",
            )
