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

"""Verify Replicator motion-blur capture snippets for RT and path tracing render modes."""

import os
import tempfile

import carb.settings
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit
import omni.usd
from isaacsim.test.utils.file_validation import get_folder_file_summary, validate_folder_contents
from isaacsim.test.utils.image_comparison import compare_images_in_directories


class TestSDGMotionBlur(omni.kit.test.AsyncTestCase):
    """Runs motion-blur SDG snippets and checks the files each capture path writes."""

    GOLDEN_MOTION_BLUR_ROOT = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data", "golden", "_out_motion_blur"
    )
    RGB_MEAN_DIFF_TOLERANCE = 5

    async def setUp(self) -> None:
        """Create a clean stage and preserve the DLSS setting used by capture snippets."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()
        self.original_dlss_exec_mode = carb.settings.get_settings().get("rtx/post/dlss/execMode")

    async def tearDown(self) -> None:
        """Close the stage, wait for pending loads, and restore the DLSS setting."""
        stage_utils.close_stage()
        await app_utils.update_app_async()
        # In some cases the test will end before the asset is loaded, in this case wait for assets to load
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await app_utils.update_app_async()
        carb.settings.get_settings().set("rtx/post/dlss/execMode", self.original_dlss_exec_mode)

    async def _run_motion_blur_raytracing_async(
        self,
        delta_times: list[float | None],
        num_frames: int = 3,
        max_blur_diameter_fraction: float = 0.02,
        exposure_fraction: float = 1.0,
        num_samples: int = 8,
    ) -> None:
        """Run the RTX Real-Time motion blur snippet.

        Args:
            delta_times: Frame delta times in seconds, with ``None`` selecting the stage time-code rate.
            num_frames: Number of frames to capture for each delta time.
            max_blur_diameter_fraction: Maximum blur diameter relative to the largest screen dimension.
            exposure_fraction: Exposure duration as a fraction of one frame.
            num_samples: Number of motion-blur filter samples.
        """
        import os

        import carb.settings
        import isaacsim.core.experimental.utils.stage as stage_utils
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import PhysicsScene
        from isaacsim.storage.native import get_assets_root_path_async

        # Paths to the animated and physics-ready assets
        PHYSICS_ASSET_URL = "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
        ANIM_ASSET_URL = "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd"

        # -z velocities and start locations of the animated (left side) and physics (right side) assets (stage units/s)
        ASSET_VELOCITIES = [0, 5, 10]
        ASSET_X_MIRRORED_LOCATIONS = [(0.5, 0, 0.3), (0.3, 0, 0.3), (0.1, 0, 0.3)]

        # Used to calculate how many frames to animate the assets to maintain the same velocity as the physics assets
        ANIMATION_DURATION = 10

        # Number of frames to capture for each scenario
        # NUM_FRAMES = 3
        NUM_FRAMES = num_frames

        # # Configuration for RTX Real-Time motion blur examples
        # DELTA_TIMES = [None, 1 / 30, 1 / 60, 1 / 240]
        # MAX_BLUR_DIAMETER_FRACTION = 0.02
        # EXPOSURE_FRACTION = 1.0
        # NUM_SAMPLES = 8

        async def setup_stage_async() -> None:
            """Create a new USD stage with animated and physics-enabled assets with synchronized motion."""
            await stage_utils.create_new_stage_async()
            settings = carb.settings.get_settings()
            # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
            settings.set("rtx/post/dlss/execMode", 2)

            # Capture data only on request
            rep.orchestrator.set_capture_on_play(False)

            timeline = omni.timeline.get_timeline_interface()
            timeline.set_end_time(ANIMATION_DURATION)
            timeline.commit()

            # Create lights
            rep.functional.create.xform(name="World")
            rep.functional.create.dome_light(intensity=100, parent="/World", name="DomeLight")
            rep.functional.create.distant_light(
                intensity=2500, rotation=(315, 0, 0), parent="/World", name="DistantLight"
            )

            # Setup the physics assets with gravity disabled and the requested velocity
            assets_root_path = await get_assets_root_path_async()
            physics_asset_url = assets_root_path + PHYSICS_ASSET_URL
            for location, velocity in zip(ASSET_X_MIRRORED_LOCATIONS, ASSET_VELOCITIES):
                prim = rep.functional.create.reference(
                    usd_path=physics_asset_url,
                    parent="/World",
                    name=f"physics_asset_{int(abs(velocity))}",
                    position=location,
                )
                rigid_prim = RigidPrim(str(prim.GetPrimPath()))
                rigid_prim.set_enabled_gravities([False])
                rigid_prim.set_velocities(linear_velocities=[(0, 0, -velocity)], angular_velocities=[(0, 0, 0)])

            # Setup animated assets maintaining the same velocity as the physics assets
            anim_asset_url = assets_root_path + ANIM_ASSET_URL
            for location, velocity in zip(ASSET_X_MIRRORED_LOCATIONS, ASSET_VELOCITIES):
                start_location = (-location[0], location[1], location[2])
                prim = rep.functional.create.reference(
                    usd_path=anim_asset_url,
                    parent="/World",
                    name=f"anim_asset_{int(abs(velocity))}",
                    position=start_location,
                )
                animation_distance = velocity * ANIMATION_DURATION
                end_location = (start_location[0], start_location[1], start_location[2] - animation_distance)
                end_keyframe_time = timeline.get_time_codes_per_seconds() * ANIMATION_DURATION
                # Timesampled keyframe (animated) translation
                prim.GetAttribute("xformOp:translate").Set(start_location, time=0)
                prim.GetAttribute("xformOp:translate").Set(end_location, time=end_keyframe_time)

        async def run_motion_blur_example_rt_async(
            num_frames: int = NUM_FRAMES,
            delta_time: float | None = None,
            max_blur_diameter_fraction: float = 0.02,
            exposure_fraction: float = 1.0,
            num_samples: int = 8,
        ) -> None:
            """Capture motion blur frames using RTX Real-Time rendering.

            Args:
                num_frames: Number of frames to capture.
                delta_time: Frame delta time in seconds, or ``None`` to use the stage time-code rate.
                max_blur_diameter_fraction: Maximum blur diameter relative to the largest screen dimension.
                exposure_fraction: Exposure duration as a fraction of one frame.
                num_samples: Number of motion-blur filter samples.
            """
            await setup_stage_async()
            stage = omni.usd.get_context().get_stage()
            settings = carb.settings.get_settings()

            # Enable motion blur capture
            settings.set("/omni/replicator/captureMotionBlur", True)

            # Set RTX Real-Time (raytracing) motion blur settings
            print("[MotionBlur] Setting RealTimePathTracing render mode motion blur settings")
            settings.set("/rtx/rendermode", "RealTimePathTracing")
            # 0: Disabled, 1: TAA, 2: FXAA, 3: DLSS, 4:RTXAA
            settings.set("/rtx/post/aa/op", 2)
            # (float): The fraction of the largest screen dimension to use as the maximum motion blur diameter.
            settings.set("/rtx/post/motionblur/maxBlurDiameterFraction", max_blur_diameter_fraction)
            # (float): Exposure time fraction in frames (1.0 = one frame duration) to sample.
            settings.set("/rtx/post/motionblur/exposureFraction", exposure_fraction)
            # (int): Number of samples to use in the filter. A higher number improves quality at the cost of performance.
            settings.set("/rtx/post/motionblur/numSamples", num_samples)

            # Setup backend
            delta_time_str = "None" if delta_time is None else f"{delta_time:.4f}"
            output_directory = os.path.join(os.getcwd(), f"_out_motion_blur_func_dt_{delta_time_str}_rt")
            print(f"[MotionBlur] Output directory: {output_directory}")
            backend = rep.backends.get("DiskBackend")
            backend.initialize(output_dir=output_directory)

            # Setup writer and render product
            camera = rep.functional.create.camera(
                position=(0, 1.5, 0), look_at=(0, 0, 0), parent="/World", name="MotionBlurCam"
            )
            render_product = rep.create.render_product(camera, (1280, 720))
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(backend=backend, rgb=True)
            writer.attach(render_product)

            # Run a few updates to make sure all materials are fully loaded for capture
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            rep.functional.physics.create_physics_scene(path="/PhysicsScene")
            physics_scene = PhysicsScene("/PhysicsScene")

            stage_time_codes_per_second = stage.GetTimeCodesPerSecond()
            target_physics_fps = stage_time_codes_per_second if delta_time is None else 1 / delta_time

            target_physics_dt = 1.0 / target_physics_fps
            original_physics_dt = physics_scene.get_dt()

            if target_physics_dt < original_physics_dt:
                print(
                    f"[MotionBlur] Changing physics FPS from {1.0 / original_physics_dt:.0f} to {target_physics_fps:.0f}"
                )
                physics_scene.set_dt(target_physics_dt)

            # Start the timeline
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            timeline.commit()

            for i in range(num_frames):
                print(f"[MotionBlur] \tCapturing frame {i}")
                await rep.orchestrator.step_async(delta_time=delta_time)

            if target_physics_dt < original_physics_dt:
                print(
                    f"[MotionBlur] Restoring physics FPS from {target_physics_fps:.0f} to {1.0 / original_physics_dt:.0f}"
                )
                physics_scene.set_dt(original_physics_dt)

            # Wait until the data is fully written
            await rep.orchestrator.wait_until_complete_async()

            # Stop the timeline
            timeline.stop()
            timeline.commit()

            # Cleanup
            writer.detach()
            render_product.destroy()

        for delta_time in delta_times:
            await run_motion_blur_example_rt_async(
                num_frames=NUM_FRAMES,
                delta_time=delta_time,
                max_blur_diameter_fraction=max_blur_diameter_fraction,
                exposure_fraction=exposure_fraction,
                num_samples=num_samples,
            )

    async def _run_motion_blur_pathtracing_async(
        self,
        delta_times: list[float | None],
        samples_per_pixel: list[int],
        motion_blur_subsamples: list[int],
        num_frames: int = 3,
    ) -> None:
        """Run the Path Tracing motion blur snippet.

        Args:
            delta_times: Frame delta times in seconds, with ``None`` selecting the stage time-code rate.
            samples_per_pixel: Path-tracing sample counts to capture.
            motion_blur_subsamples: Motion-blur subframe counts to capture.
            num_frames: Number of frames to capture for each configuration.
        """
        import os

        import carb.settings
        import isaacsim.core.experimental.utils.stage as stage_utils
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import PhysicsScene
        from isaacsim.storage.native import get_assets_root_path_async

        # Paths to the animated and physics-ready assets
        PHYSICS_ASSET_URL = "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
        ANIM_ASSET_URL = "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd"

        # -z velocities and start locations of the animated (left side) and physics (right side) assets (stage units/s)
        ASSET_VELOCITIES = [0, 5, 10]
        ASSET_X_MIRRORED_LOCATIONS = [(0.5, 0, 0.3), (0.3, 0, 0.3), (0.1, 0, 0.3)]

        # Used to calculate how many frames to animate the assets to maintain the same velocity as the physics assets
        ANIMATION_DURATION = 10

        # Number of frames to capture for each scenario
        # NUM_FRAMES = 3
        NUM_FRAMES = num_frames

        # # Configuration for Path Tracing motion blur examples
        # DELTA_TIMES = [None, 1 / 30, 1 / 60, 1 / 240]
        # SAMPLES_PER_PIXEL = [32, 128]
        # MOTION_BLUR_SUBSAMPLES = [4, 16]

        async def setup_stage_async() -> None:
            """Create a new USD stage with animated and physics-enabled assets with synchronized motion."""
            await stage_utils.create_new_stage_async()
            settings = carb.settings.get_settings()
            # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
            settings.set("rtx/post/dlss/execMode", 2)

            # Capture data only on request
            rep.orchestrator.set_capture_on_play(False)

            timeline = omni.timeline.get_timeline_interface()
            timeline.set_end_time(ANIMATION_DURATION)
            timeline.commit()

            # Create lights
            rep.functional.create.xform(name="World")
            rep.functional.create.dome_light(intensity=100, parent="/World", name="DomeLight")
            rep.functional.create.distant_light(
                intensity=2500, rotation=(315, 0, 0), parent="/World", name="DistantLight"
            )

            # Setup the physics assets with gravity disabled and the requested velocity
            assets_root_path = await get_assets_root_path_async()
            physics_asset_url = assets_root_path + PHYSICS_ASSET_URL
            for location, velocity in zip(ASSET_X_MIRRORED_LOCATIONS, ASSET_VELOCITIES):
                prim = rep.functional.create.reference(
                    usd_path=physics_asset_url,
                    parent="/World",
                    name=f"physics_asset_{int(abs(velocity))}",
                    position=location,
                )
                rigid_prim = RigidPrim(str(prim.GetPrimPath()))
                rigid_prim.set_enabled_gravities([False])
                rigid_prim.set_velocities(linear_velocities=[(0, 0, -velocity)], angular_velocities=[(0, 0, 0)])

            # Setup animated assets maintaining the same velocity as the physics assets
            anim_asset_url = assets_root_path + ANIM_ASSET_URL
            for location, velocity in zip(ASSET_X_MIRRORED_LOCATIONS, ASSET_VELOCITIES):
                start_location = (-location[0], location[1], location[2])
                prim = rep.functional.create.reference(
                    usd_path=anim_asset_url,
                    parent="/World",
                    name=f"anim_asset_{int(abs(velocity))}",
                    position=start_location,
                )
                animation_distance = velocity * ANIMATION_DURATION
                end_location = (start_location[0], start_location[1], start_location[2] - animation_distance)
                end_keyframe_time = timeline.get_time_codes_per_seconds() * ANIMATION_DURATION
                # Timesampled keyframe (animated) translation
                prim.GetAttribute("xformOp:translate").Set(start_location, time=0)
                prim.GetAttribute("xformOp:translate").Set(end_location, time=end_keyframe_time)

        async def run_motion_blur_example_pt_async(
            num_frames: int = NUM_FRAMES,
            delta_time: float | None = None,
            motion_blur_subsamples: int = 8,
            samples_per_pixel: int = 64,
        ) -> None:
            """Capture motion blur frames using Path Tracing rendering.

            Args:
                num_frames: Number of frames to capture.
                delta_time: Frame delta time in seconds, or ``None`` to use the stage time-code rate.
                motion_blur_subsamples: Number of motion-blur subframes to render.
                samples_per_pixel: Number of path-tracing samples per pixel.
            """
            await setup_stage_async()
            stage = omni.usd.get_context().get_stage()
            settings = carb.settings.get_settings()

            # Enable motion blur capture
            settings.set("/omni/replicator/captureMotionBlur", True)

            # Set Path Tracing motion blur settings
            print("[MotionBlur] Setting PathTracing render mode motion blur settings")
            settings.set("/rtx/rendermode", "PathTracing")
            # (int): Total number of samples for each rendered pixel, per frame.
            settings.set("/rtx/pathtracing/spp", samples_per_pixel)
            # (int): Maximum number of samples to accumulate per pixel. When this count is reached the rendering stops until a scene or setting change is detected, restarting the rendering process. Set to 0 to remove this limit.
            settings.set("/rtx/pathtracing/totalSpp", samples_per_pixel)
            settings.set("/rtx/pathtracing/optixDenoiser/enabled", 0)
            # Number of sub samples to render if in PathTracing render mode and motion blur is enabled.
            settings.set("/omni/replicator/pathTracedMotionBlurSubSamples", motion_blur_subsamples)

            # Setup backend
            delta_time_str = "None" if delta_time is None else f"{delta_time:.4f}"
            mode_str = f"pt_subsamples_{motion_blur_subsamples}_spp_{samples_per_pixel}"
            output_directory = os.path.join(os.getcwd(), f"_out_motion_blur_func_dt_{delta_time_str}_{mode_str}")
            print(f"[MotionBlur] Output directory: {output_directory}")
            backend = rep.backends.get("DiskBackend")
            backend.initialize(output_dir=output_directory)

            # Setup writer and render product
            camera = rep.functional.create.camera(
                position=(0, 1.5, 0), look_at=(0, 0, 0), parent="/World", name="MotionBlurCam"
            )
            render_product = rep.create.render_product(camera, (1280, 720))
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(backend=backend, rgb=True)
            writer.attach(render_product)

            # Run a few updates to make sure all materials are fully loaded for capture
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            rep.functional.physics.create_physics_scene(path="/PhysicsScene")
            physics_scene = PhysicsScene("/PhysicsScene")

            # Path tracing renders multiple subframes per frame, so the physics FPS is scaled accordingly
            stage_time_codes_per_second = stage.GetTimeCodesPerSecond()
            target_physics_fps = stage_time_codes_per_second if delta_time is None else 1 / delta_time
            target_physics_fps *= motion_blur_subsamples

            target_physics_dt = 1.0 / target_physics_fps
            original_physics_dt = physics_scene.get_dt()

            if target_physics_dt < original_physics_dt:
                print(
                    f"[MotionBlur] Changing physics FPS from {1.0 / original_physics_dt:.0f} to {target_physics_fps:.0f}"
                )
                physics_scene.set_dt(target_physics_dt)

            # Start the timeline
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            timeline.commit()

            for i in range(num_frames):
                print(f"[MotionBlur] \tCapturing frame {i}")
                await rep.orchestrator.step_async(delta_time=delta_time)

            if target_physics_dt < original_physics_dt:
                print(
                    f"[MotionBlur] Restoring physics FPS from {target_physics_fps:.0f} to {1.0 / original_physics_dt:.0f}"
                )
                physics_scene.set_dt(original_physics_dt)

            # Switch back to the raytracing render mode
            print("[MotionBlur] Restoring render mode to RealTimePathTracing")
            settings.set("/rtx/rendermode", "RealTimePathTracing")

            # Wait until the data is fully written
            await rep.orchestrator.wait_until_complete_async()

            # Stop the timeline
            timeline.stop()
            timeline.commit()

            # Cleanup
            writer.detach()
            render_product.destroy()

        for delta_time in delta_times:
            for motion_blur_subsample in motion_blur_subsamples:
                for samples_per_pixel_value in samples_per_pixel:
                    await run_motion_blur_example_pt_async(
                        num_frames=NUM_FRAMES,
                        delta_time=delta_time,
                        motion_blur_subsamples=motion_blur_subsample,
                        samples_per_pixel=samples_per_pixel_value,
                    )

    def _validate_against_golden(self, out_dir: str, golden_dir: str, num_frames: int) -> None:
        """Validate captured output file counts and compare RGB images against golden data.

        Args:
            out_dir: Directory containing captured output.
            golden_dir: Directory containing golden images.
            num_frames: Expected number of captured frames.
        """
        folder_contents_success = validate_folder_contents(path=out_dir, expected_counts={"png": num_frames})
        summary = get_folder_file_summary(out_dir)
        print(f"[MotionBlur] Validating {out_dir}: expected png={num_frames}, found {summary}")
        self.assertTrue(
            folder_contents_success,
            f"Output directory contents validation failed for {out_dir}: expected png={num_frames}, found {summary}",
        )

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

    async def test_sdg_snippet_motion_blur_rt_delta_none(self) -> None:
        """Capture short motion-blur sequences with RealTimePathTracing and default delta time."""
        motion_blur_output_root = tempfile.mkdtemp(prefix="test_motion_blur_rt_delta_none_")
        print(f"Test output root: {motion_blur_output_root}")

        num_frames = 3
        delta_times = [None]

        original_cwd = os.getcwd()
        try:
            os.chdir(motion_blur_output_root)
            await self._run_motion_blur_raytracing_async(delta_times=delta_times, num_frames=num_frames)
        finally:
            os.chdir(original_cwd)

        out_dir = os.path.join(motion_blur_output_root, "_out_motion_blur_func_dt_None_rt")
        golden_dir = os.path.join(self.GOLDEN_MOTION_BLUR_ROOT, "_out_motion_blur_func_dt_None_rt")
        self._validate_against_golden(out_dir, golden_dir, num_frames)

    async def test_sdg_snippet_motion_blur_pt_delta_none(self) -> None:
        """Capture short motion-blur sequences with PathTracing and default delta time."""
        motion_blur_output_root = tempfile.mkdtemp(prefix="test_motion_blur_pt_delta_none_")
        print(f"Test output root: {motion_blur_output_root}")

        num_frames = 3
        delta_times = [None]
        samples_per_pixel = [32]
        motion_blur_subsamples = [4]

        original_cwd = os.getcwd()
        try:
            os.chdir(motion_blur_output_root)
            await self._run_motion_blur_pathtracing_async(
                delta_times=delta_times,
                samples_per_pixel=samples_per_pixel,
                motion_blur_subsamples=motion_blur_subsamples,
                num_frames=num_frames,
            )
        finally:
            os.chdir(original_cwd)

        out_dir = os.path.join(motion_blur_output_root, "_out_motion_blur_func_dt_None_pt_subsamples_4_spp_32")
        golden_dir = os.path.join(self.GOLDEN_MOTION_BLUR_ROOT, "_out_motion_blur_func_dt_None_pt_subsamples_4_spp_32")
        self._validate_against_golden(out_dir, golden_dir, num_frames)

    async def test_sdg_snippet_motion_blur_rt_delta_240(self) -> None:
        """Capture short motion-blur sequences with RealTimePathTracing and 1/240 delta time."""
        motion_blur_output_root = tempfile.mkdtemp(prefix="test_motion_blur_rt_delta_240_")
        print(f"Test output root: {motion_blur_output_root}")

        num_frames = 3
        delta_times = [1 / 240]

        original_cwd = os.getcwd()
        try:
            os.chdir(motion_blur_output_root)
            await self._run_motion_blur_raytracing_async(delta_times=delta_times, num_frames=num_frames)
        finally:
            os.chdir(original_cwd)

        out_dir = os.path.join(motion_blur_output_root, "_out_motion_blur_func_dt_0.0042_rt")
        golden_dir = os.path.join(self.GOLDEN_MOTION_BLUR_ROOT, "_out_motion_blur_func_dt_0.0042_rt")
        self._validate_against_golden(out_dir, golden_dir, num_frames)

    async def test_sdg_snippet_motion_blur_pt_delta_240(self) -> None:
        """Capture short motion-blur sequences with PathTracing and 1/240 delta time."""
        motion_blur_output_root = tempfile.mkdtemp(prefix="test_motion_blur_pt_delta_240_")
        print(f"Test output root: {motion_blur_output_root}")

        num_frames = 3
        delta_times = [1 / 240]
        samples_per_pixel = [32]
        motion_blur_subsamples = [4]

        original_cwd = os.getcwd()
        try:
            os.chdir(motion_blur_output_root)
            await self._run_motion_blur_pathtracing_async(
                delta_times=delta_times,
                samples_per_pixel=samples_per_pixel,
                motion_blur_subsamples=motion_blur_subsamples,
                num_frames=num_frames,
            )
        finally:
            os.chdir(original_cwd)

        out_dir = os.path.join(motion_blur_output_root, "_out_motion_blur_func_dt_0.0042_pt_subsamples_4_spp_32")
        golden_dir = os.path.join(
            self.GOLDEN_MOTION_BLUR_ROOT, "_out_motion_blur_func_dt_0.0042_pt_subsamples_4_spp_32"
        )
        self._validate_against_golden(out_dir, golden_dir, num_frames)
