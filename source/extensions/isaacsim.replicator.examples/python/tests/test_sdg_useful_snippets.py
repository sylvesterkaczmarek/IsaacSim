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

"""Verify Replicator snippets for multi-camera, simulation, event, and writer capture."""

import tempfile
from typing import Any

import carb.settings
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit
import omni.usd
from isaacsim.test.utils.file_validation import validate_folder_contents


class TestSDGUsefulSnippets(omni.kit.test.AsyncTestCase):
    """Runs representative SDG snippets and checks the files each capture path writes."""

    async def setUp(self) -> None:
        """Create a clean stage and preserve the DLSS setting used by capture snippets."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()
        self.original_dlss_exec_mode = carb.settings.get_settings().get("rtx/post/dlss/execMode")

    async def tearDown(self) -> Any:
        """Close the stage, wait for pending loads, and restore the DLSS setting.

        Returns:
            None.
        """
        stage_utils.close_stage()
        await app_utils.update_app_async()
        # In some cases the test will end before the asset is loaded, in this case wait for assets to load
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await app_utils.update_app_async()
        carb.settings.get_settings().set("rtx/post/dlss/execMode", self.original_dlss_exec_mode)

    async def test_sdg_snippet_multi_camera(self) -> Any:
        """Capture three camera views through both direct RGB annotators and a custom writer.

        Returns:
            None.
        """
        import os

        import carb.settings
        import omni.replicator.core as rep
        import omni.usd
        from omni.replicator.core import Writer
        from omni.replicator.core.backends import DiskBackend
        from omni.replicator.core.functional import write_image

        NUM_FRAMES = 5
        mc_test_root = tempfile.mkdtemp(prefix="test_mc_")
        print(f"Test output root: {mc_test_root}")

        # Randomize cube color every frame using a graph-based replicator randomizer
        def cube_color_randomizer() -> None:
            cube_prims = rep.get.prims(path_pattern="Cube")
            with cube_prims:
                rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
            return cube_prims.node

        # Example of custom writer class to access the annotator data
        class MyWriter(Writer):
            def __init__(self, rgb: bool = True) -> None:
                # Organize data from render product perspective (legacy, annotator, renderProduct)
                self.data_structure = "renderProduct"
                self.annotators = []
                self._frame_id = 0
                if rgb:
                    # Create a new rgb annotator and add it to the writer's list of annotators
                    self.annotators.append(rep.annotators.get("rgb"))
                # Create writer output directory and initialize DiskBackend
                writer_dir = os.path.join(mc_test_root, "writer")
                print(f"Writing writer data to {writer_dir}")
                self.backend = DiskBackend(output_dir=writer_dir, overwrite=True)

            def write(self, data: Any) -> None:
                if "renderProducts" in data:
                    for rp_name, rp_data in data["renderProducts"].items():
                        if "rgb" in rp_data:
                            file_path = f"{rp_name}_frame_{self._frame_id}.png"
                            self.backend.schedule(write_image, data=rp_data["rgb"]["data"], path=file_path)
                self._frame_id += 1

        rep.WriterRegistry.register(MyWriter)

        # Create a new stage
        omni.usd.get_context().new_stage()

        # Set global random seed for the replicator randomizer
        rep.set_global_seed(11)

        # Disable capture on play to capture data manually using step
        rep.orchestrator.set_capture_on_play(False)

        # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

        # Setup stage
        rep.functional.create.xform(name="World")
        rep.functional.create.dome_light(intensity=900, parent="/World", name="DomeLight")
        cube = rep.functional.create.cube(parent="/World", name="Cube", semantics={"class": "my_cube"})

        # Register the graph-based cube color randomizer to trigger on every frame
        rep.randomizer.register(cube_color_randomizer)
        with rep.trigger.on_frame():
            rep.randomizer.cube_color_randomizer()

        # Create cameras
        cam_top = rep.functional.create.camera(position=(0, 0, 5), look_at=(0, 0, 0), parent="/World", name="CamTop")
        cam_side = rep.functional.create.camera(position=(2, 2, 0), look_at=(0, 0, 0), parent="/World", name="CamSide")
        cam_persp = rep.functional.create.camera(
            position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="CamPersp"
        )

        # Create the render products
        rp_top = rep.create.render_product(cam_top, resolution=(320, 320), name="RpTop")
        rp_side = rep.create.render_product(cam_side, resolution=(640, 640), name="RpSide")
        rp_persp = rep.create.render_product(cam_persp, resolution=(1024, 1024), name="RpPersp")

        # Example of accessing the data through a custom writer
        writer = rep.WriterRegistry.get("MyWriter")
        writer.initialize(rgb=True)
        writer.attach([rp_top, rp_side, rp_persp])

        # Example of accessing the data directly through annotators
        rgb_annotators = []
        for rp in [rp_top, rp_side, rp_persp]:
            # Create a new rgb annotator for each render product
            rgb = rep.annotators.get("rgb")
            # Attach the annotator to the render product
            rgb.attach(rp)
            rgb_annotators.append(rgb)

        # Create annotator output directory
        output_dir_annot = os.path.join(mc_test_root, "annot")
        print(f"Writing annotator data to {output_dir_annot}")
        os.makedirs(output_dir_annot)

        async def run_example_async() -> None:
            """Step color-randomized captures and write each render product through both paths."""
            for i in range(NUM_FRAMES):
                print(f"Step {i}")
                # The step function triggers registered graph-based randomizers, collects data from annotators,
                # and invokes the write function of attached writers with the annotator data
                await rep.orchestrator.step_async(rt_subframes=32)
                for j, rgb_annot in enumerate(rgb_annotators):
                    file_path = os.path.join(output_dir_annot, f"rp{j}_step_{i}.png")
                    write_image(path=file_path, data=rgb_annot.get_data())

            # Wait for the data to be written and release resources
            await rep.orchestrator.wait_until_complete_async()
            writer.detach()
            for annot in rgb_annotators:
                annot.detach()
            for rp in [rp_top, rp_side, rp_persp]:
                rp.destroy()

        # asyncio.ensure_future(run_example_async())
        await run_example_async()

        # Validate the output directory contents
        annot_output_dir = os.path.join(mc_test_root, "annot")
        folder_contents_success_annot = validate_folder_contents(
            path=annot_output_dir, expected_counts={"png": NUM_FRAMES * 3}
        )
        self.assertTrue(
            folder_contents_success_annot, f"Output directory contents validation failed for {annot_output_dir}"
        )
        writer_output_dir = os.path.join(mc_test_root, "writer")
        folder_contents_success_writer = validate_folder_contents(
            path=writer_output_dir, expected_counts={"png": NUM_FRAMES * 3}
        )
        self.assertTrue(
            folder_contents_success_writer, f"Output directory contents validation failed for {writer_output_dir}"
        )

    async def test_sdg_snippet_simulation_get_data(self) -> None:
        """Drop cubes with SimulationManager and capture writer and annotator data when each stops."""
        import os

        import carb.settings
        import numpy as np
        import omni
        import omni.replicator.core as rep
        from isaacsim.core.experimental.objects import GroundPlane
        from isaacsim.core.simulation_manager import SimulationManager
        from omni.replicator.core.functional import write_image, write_json
        from pxr import UsdPhysics

        # Util function to save semantic segmentation annotator data
        def write_sem_data(sem_data: Any, file_path: Any) -> None:
            id_to_labels = sem_data["info"]["idToLabels"]
            write_json(path=file_path + ".json", data=id_to_labels)
            sem_image_data = sem_data["data"]
            write_image(path=file_path + ".png", data=sem_image_data)

        # Create a new stage
        omni.usd.get_context().new_stage()

        # Setting capture on play to False will prevent the replicator from capturing data each frame
        rep.orchestrator.set_capture_on_play(False)

        # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

        # Add a dome light and a ground plane
        rep.functional.create.xform(name="World")
        rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
        ground_plane = GroundPlane("/World/GroundPlane")
        rep.functional.modify.semantics(ground_plane.prims, {"class": "ground_plane"}, mode="add")

        # Create a camera and render product to collect the data from
        rep.functional.create.xform(name="World")
        cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
        rp = rep.create.render_product(cam, resolution=(512, 512), name="MyRenderProduct")

        # Set the output directory for the data
        out_dir = tempfile.mkdtemp(prefix="test_sim_event_")
        writer_dir = os.path.join(out_dir, "writer")
        annotator_dir = os.path.join(out_dir, "annotator")

        os.makedirs(writer_dir)
        os.makedirs(annotator_dir)

        print(f"Outputting data to {out_dir}..")
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=writer_dir)

        # Example of using a writer to save the data
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True, semantic_segmentation=True, colorize_semantic_segmentation=True)
        writer.attach(rp)

        # Example of accesing the data directly from annotators
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach(rp)
        sem_annot = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": True})
        sem_annot.attach(rp)

        # Initialize the simulation manager
        SimulationManager.initialize_physics()

        async def run_example_async() -> None:
            """Spawn rigid cubes, wait for low velocity, and write RGB plus semantic data."""
            # Spawn and drop a few cubes, capture data when they stop moving
            for i in range(5):
                cube = rep.functional.create.cube(name=f"Cuboid_{i}", parent="/World")
                rep.functional.modify.position(cube, (0, 0, 10 + i))
                rep.functional.modify.semantics(cube, {"class": "cuboid"}, mode="add")
                rep.functional.physics.apply_rigid_body(cube, with_collider=True)
                physics_rigid_body_api = UsdPhysics.RigidBodyAPI(cube)

                for s in range(500):
                    SimulationManager.step()
                    linear_velocity = physics_rigid_body_api.GetVelocityAttr().Get()
                    speed = np.linalg.norm(linear_velocity)

                    if speed < 0.1:
                        print(f"Cube_{i} stopped moving after {s} simulation steps, writing data..")
                        # Tigger the writer and update the annotators with new data
                        await rep.orchestrator.step_async(rt_subframes=4, delta_time=0.0, pause_timeline=False)
                        rgb_path = os.path.join(annotator_dir, f"Cube_{i}_step_{s}_rgb.png")
                        sem_path = os.path.join(annotator_dir, f"Cube_{i}_step_{s}_sem")
                        write_image(path=rgb_path, data=rgb_annot.get_data())
                        write_sem_data(sem_annot.get_data(), sem_path)
                        break

            # Wait for the data to be written to disk and clean up resources
            await rep.orchestrator.wait_until_complete_async()
            rgb_annot.detach()
            sem_annot.detach()
            writer.detach()
            rp.destroy()

        # asyncio.ensure_future(run_example_async())
        await run_example_async()

        # Validate the output directory contents
        folder_contents_success_annot = validate_folder_contents(
            path=annotator_dir, expected_counts={"png": 10, "json": 5}
        )
        folder_contents_success_writer = validate_folder_contents(
            path=writer_dir, expected_counts={"png": 10, "json": 5}
        )
        self.assertTrue(
            folder_contents_success_annot, f"Output directory contents validation failed for {annotator_dir}"
        )
        self.assertTrue(folder_contents_success_writer, f"Output directory contents validation failed for {writer_dir}")

    async def test_sdg_snippet_custom_event_and_write(self) -> None:
        """Trigger named graph randomizers and direct USD pose edits between BasicWriter captures."""
        import carb.settings
        import omni.replicator.core as rep
        import omni.usd

        omni.usd.get_context().new_stage()

        # Set global random seed for the replicator randomizer to ensure reproducibility
        rep.set_global_seed(11)

        # Setting capture on play to False will prevent the replicator from capturing data each frame
        rep.orchestrator.set_capture_on_play(False)

        # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

        rep.functional.create.xform(name="World")
        rep.functional.create.distant_light(intensity=4000, rotation=(315, 0, 0), parent="/World", name="DistantLight")
        small_cube = rep.functional.create.cube(scale=0.75, position=(-1.5, 1.5, 0), parent="/World", name="SmallCube")
        large_cube = rep.functional.create.cube(scale=1.25, position=(1.5, -1.5, 0), parent="/World", name="LargeCube")

        # Graph-based randomizations triggered on custom events
        with rep.trigger.on_custom_event(event_name="randomize_small_cube"):
            small_cube_node = rep.get.prim_at_path(small_cube.GetPath())
            with small_cube_node:
                rep.randomizer.rotation()

        with rep.trigger.on_custom_event(event_name="randomize_large_cube"):
            large_cube_node = rep.get.prim_at_path(large_cube.GetPath())
            with large_cube_node:
                rep.randomizer.rotation()

        # Use the disk backend to write the data to disk
        out_dir = tempfile.mkdtemp(prefix="test_custom_event_")
        print(f"Writing data to {out_dir}")
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=out_dir)

        cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
        rp = rep.create.render_product(cam, (512, 512))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        writer.attach(rp)

        async def run_example_async() -> None:
            """Capture original, event-randomized, and directly moved cube states."""
            print(f"Capturing at original positions")
            await rep.orchestrator.step_async(rt_subframes=8)

            print("Randomizing small cube rotation (graph-based) and capturing...")
            rep.utils.send_og_event(event_name="randomize_small_cube")
            await rep.orchestrator.step_async(rt_subframes=8)

            print("Moving small cube position (USD API) and capturing...")
            small_cube.GetAttribute("xformOp:translate").Set((-1.5, 1.5, -2))
            await rep.orchestrator.step_async(rt_subframes=8)

            print("Randomizing large cube rotation (graph-based) and capturing...")
            rep.utils.send_og_event(event_name="randomize_large_cube")
            await rep.orchestrator.step_async(rt_subframes=8)

            print("Moving large cube position (USD API) and capturing...")
            large_cube.GetAttribute("xformOp:translate").Set((1.5, -1.5, 2))
            await rep.orchestrator.step_async(rt_subframes=8)

            # Wait until all the data is saved to disk and cleanup writer and render product
            await rep.orchestrator.wait_until_complete_async()
            writer.detach()
            rp.destroy()

        # asyncio.ensure_future(run_example_async())
        await run_example_async()

        # Validate the output directory contents
        folder_contents_success = validate_folder_contents(path=out_dir, expected_counts={"png": 5})
        self.assertTrue(folder_contents_success, f"Output directory contents validation failed for {out_dir}")
