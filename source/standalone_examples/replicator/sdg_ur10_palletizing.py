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

"""Generate synthetic data from the UR10 palletizing simulation."""

from __future__ import annotations

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import asyncio
import json
import os
import random

import carb
import carb.settings
import omni
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.robot_motion.examples", True)

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.bounds as bounds_utils
import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.replicator.core as rep
from isaacsim.core.experimental.utils.semantics import upgrade_prim_semantics_to_labels
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from isaacsim.robot_motion.examples.manipulation.ur10_palletizing import (
    BinSpawner,
    BinStackingContext,
    PalletizerController,
    Ur10Assets,
    build_scene,
    configure_conveyor,
)
from isaacsim.storage.native import get_assets_root_path
from omni.physx import get_physx_scene_query_interface
from omni.replicator.core.functional import write_image
from pxr import Sdf

DEFAULT_NUM_CAPTURES = 4
DEFAULT_BIN_FLIP_FRAMES = 2
DEFAULT_PALLET_FRAMES = 2
MAX_BINS = 36

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-captures",
    type=int,
    default=DEFAULT_NUM_CAPTURES,
    help="Number of bins to capture (1-36).",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Validate captured output files against expected counts and exit.",
)
args, _ = parser.parse_known_args()


class Ur10PalletizingSimulation:
    ENV_PATH = "/World/Ur10Table"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.robot = None
        self.spawner: BinSpawner | None = None
        self.context: BinStackingContext | None = None
        self.controller: PalletizerController | None = None
        self._physics_callback_id: int | None = None
        self._initialized = False
        self._needs_reset = True

    async def load_async(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)
        await omni.usd.get_context().new_stage_async()

        SimulationManager.setup_simulation(device="cpu")
        assets = Ur10Assets()
        self.robot = build_scene(self.ENV_PATH, assets)
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()
        configure_conveyor(self.ENV_PATH)
        self.robot.setup()
        await app_utils.update_app_async()
        self.spawner = BinSpawner(self.ENV_PATH, assets)
        self.context = BinStackingContext(self.robot, self.spawner)
        self.controller = PalletizerController()
        self._initialized = False
        self._needs_reset = True

    def start(self) -> None:
        self._remove_callback()
        self._physics_callback_id = SimulationManager.register_callback(
            self._on_physics_step, event=SimulationEvent.PHYSICS_POST_STEP
        )
        app_utils.play()

    def clear(self) -> None:
        self._remove_callback()

    async def wait_for_bin_async(self, bin_name: str, max_frames: int = 240) -> None:
        bin_path = f"{self.ENV_PATH}/bins/{bin_name}"
        for _ in range(max_frames):
            if prim_utils.is_prim_valid(bin_path):
                _, size = bounds_utils.compute_bound_range(bin_path, space="local")
                if all(v > 0.0 for v in size):
                    return
            await app_utils.update_app_async()
        raise RuntimeError(f"Timed out waiting for {bin_name}.")

    def _remove_callback(self) -> None:
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None

    def _on_physics_step(self, dt: float, context: object) -> None:
        if self.robot is None or self.spawner is None or self.context is None or self.controller is None:
            return
        if self.controller.state == "done":
            app_utils.pause()
            return

        if not self._initialized:
            self.robot.initialize()
            self._initialized = True
        if self._needs_reset:
            self.robot.reset()
            self.spawner.reset()
            self.context.reset()
            self.controller.reset()
            self._needs_reset = False

        self.spawner.step()
        observation = self.context.read(SimulationManager.get_simulation_time())
        command = self.controller.step(observation)
        self.context.apply(command, dt)


class PalletizingSDGDemo:
    BINS_FOLDER_PATH = "/World/Ur10Table/bins"
    FLIP_HELPER_PATH = "/World/Ur10Table/pallet_holder"
    PALLET_PRIM_MESH_PATH = "/World/Ur10Table/pallet/Xform/Mesh_015"

    def __init__(self, output_dir: str | None = None) -> None:
        self._bin_counter = 0
        self._num_captures = MAX_BINS
        self._bin_flip_frames = DEFAULT_BIN_FLIP_FRAMES
        self._pallet_frames = DEFAULT_PALLET_FRAMES
        self._stage = None
        self._active_bin = None
        self._stage_event_sub = None
        self._in_running_state = False
        self._bin_flip_scenario_done = False
        self._timeline_sub = None
        self._timeline_stop_sub = None
        self._overlap_extent = None
        self._error = None
        self._rng = rep.rng.ReplicatorRNG(seed=42)
        self._variation_layer = None
        self._prev_edit_target = None
        default_out = os.path.join(os.getcwd(), "_out_palletizing_sdg_demo")
        self._output_dir = output_dir if output_dir is not None else default_out
        print(f"[PalletizingSDGDemo] Output directory: {self._output_dir}")

    def start(self, num_captures, bin_flip_frames, pallet_frames):
        self._num_captures = num_captures if 1 <= num_captures <= MAX_BINS else MAX_BINS
        self._bin_flip_frames = bin_flip_frames
        self._pallet_frames = pallet_frames
        if self._init():
            self._start()

    def is_running(self):
        if self._error is not None:
            raise self._error
        return self._in_running_state

    def _set_active_bin(self, bin_index: int) -> bool:
        bin_path = f"{self.BINS_FOLDER_PATH}/bin_{bin_index}"
        if not prim_utils.is_prim_valid(bin_path):
            self._active_bin = None
            self._overlap_extent = None
            return False
        _, size = bounds_utils.compute_bound_range(bin_path, space="local")
        half_ext = size * 0.5
        if not all(v > 0.0 for v in half_ext):
            self._active_bin = None
            self._overlap_extent = None
            return False
        self._active_bin = prim_utils.get_prim_at_path(bin_path)
        upgrade_prim_semantics_to_labels(self._active_bin, include_descendants=True)
        self._overlap_extent = carb.Float3(half_ext[0], half_ext[1], half_ext[2] * 1.1)
        return True

    def _init(self):
        self._stage = omni.usd.get_context().get_stage()
        if not self._set_active_bin(self._bin_counter):
            print("[PalletizingSDGDemo] Could not find bin. Start the palletizing simulation first.")
            return False

        if not app_utils.is_playing():
            print("[PalletizingSDGDemo] Please start the palletizing demo first.")
            return False

        upgrade_prim_semantics_to_labels("/World", include_descendants=True)
        rep.orchestrator.set_capture_on_play(False)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
        if self._stage.GetPrimAtPath("/Replicator"):
            omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])
        return True

    def _start(self):
        self._error = None
        self._timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
            on_event=self._on_timeline_event,
            observer_name="ur10_palletizing_sdg.PalletizingSDGDemo._on_timeline_event",
        )
        self._timeline_stop_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_STOP,
            on_event=self._on_timeline_stop_event,
            observer_name="ur10_palletizing_sdg.PalletizingSDGDemo._on_timeline_stop_event",
        )
        self._stage_event_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.CLOSING),
            on_event=self._on_stage_closing_event,
            observer_name="ur10_palletizing_sdg.PalletizingSDGDemo._on_stage_closing_event",
        )
        self._in_running_state = True
        print("[PalletizingSDGDemo] Starting the palletizing SDG demo.")

    def clear(self):
        self._unsubscribe_timeline_ticks()
        if self._timeline_stop_sub:
            self._timeline_stop_sub.reset()
            self._timeline_stop_sub = None
        if self._stage_event_sub:
            self._stage_event_sub.reset()
            self._stage_event_sub = None
        self._in_running_state = False
        self._bin_counter = 0
        self._active_bin = None
        self._bin_flip_scenario_done = False
        self._overlap_extent = None
        self._pop_variation_layer()
        if self._stage and self._stage.GetPrimAtPath("/Replicator"):
            omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])

    def _on_stage_closing_event(self, e: carb.eventdispatcher.Event):
        self.clear()

    def _on_timeline_stop_event(self, e: carb.eventdispatcher.Event):
        if not self._in_running_state:
            return
        print("[PalletizingSDGDemo] Timeline stopped. Cleaning up the SDG pipeline.")
        self.clear()

    def _on_timeline_event(self, e: carb.eventdispatcher.Event):
        self._check_bin_overlaps()

    def _check_bin_overlaps(self):
        if not self._active_bin and not self._set_active_bin(self._bin_counter):
            return
        bin_pose = omni.usd.get_world_transform_matrix(self._active_bin)
        origin = bin_pose.ExtractTranslation()
        quat_gf = bin_pose.ExtractRotation().GetQuaternion()
        get_physx_scene_query_interface().overlap_box(
            carb.Float3(self._overlap_extent),
            carb.Float3(origin[0], origin[1], origin[2]),
            carb.Float4(
                quat_gf.GetImaginary()[0],
                quat_gf.GetImaginary()[1],
                quat_gf.GetImaginary()[2],
                quat_gf.GetReal(),
            ),
            self._on_overlap_hit,
            False,
        )

    def _unsubscribe_timeline_ticks(self) -> None:
        if self._timeline_sub:
            self._timeline_sub.reset()
            self._timeline_sub = None

    def _on_overlap_hit(self, hit):
        if hit.rigid_body == str(self._active_bin.GetPrimPath()):
            return True

        if not self._bin_flip_scenario_done and hit.rigid_body.startswith(self.FLIP_HELPER_PATH):
            self._unsubscribe_timeline_ticks()
            asyncio.ensure_future(self._start_sdg_after_overlap(self._run_bin_flip_scenario))
            return False

        is_pallet_hit = hit.rigid_body.startswith(self.PALLET_PRIM_MESH_PATH)
        is_other_bin_hit = hit.rigid_body.startswith(f"{self.BINS_FOLDER_PATH}/bin_")
        if is_pallet_hit or is_other_bin_hit:
            self._unsubscribe_timeline_ticks()
            asyncio.ensure_future(self._start_sdg_after_overlap(self._run_pallet_scenario))
            return False
        return True

    async def _start_sdg_after_overlap(self, scenario) -> None:
        name = "bin flip" if scenario == self._run_bin_flip_scenario else "pallet"
        print(f"[PalletizingSDGDemo] Triggered {name} capture for bin {self._bin_counter}.")
        wait_retract_arm = 30
        print(f"[PalletizingSDGDemo] Waiting {wait_retract_arm} frames for the arm to retract.")
        await app_utils.update_app_async(steps=wait_retract_arm)
        if not self._in_running_state:
            return
        app_utils.pause()
        await scenario()

    def _resume_overlap_monitoring(self):
        self._timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
            on_event=self._on_timeline_event,
            observer_name="ur10_palletizing_sdg.PalletizingSDGDemo._on_timeline_event",
        )
        app_utils.play()

    def _switch_to_pathtracing(self, spp=32, total_spp=32):
        carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
        carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
        carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", total_spp)

    def _switch_to_realtime_pathtracing(self):
        carb.settings.get_settings().set("/rtx/rendermode", "RealTimePathTracing")

    def _push_variation_layer(self) -> None:
        if self._variation_layer is not None or self._stage is None:
            return
        root_layer = self._stage.GetRootLayer()
        self._prev_edit_target = self._stage.GetEditTarget()
        self._variation_layer = Sdf.Layer.CreateAnonymous("sdg_variation")
        root_layer.subLayerPaths.insert(0, self._variation_layer.identifier)
        self._stage.SetEditTarget(self._variation_layer)

    def _pop_variation_layer(self) -> None:
        if self._variation_layer is None or self._stage is None:
            return
        root_layer = self._stage.GetRootLayer()
        if self._prev_edit_target is not None:
            self._stage.SetEditTarget(self._prev_edit_target)
        self._variation_layer.Clear()
        if self._variation_layer.identifier in root_layer.subLayerPaths:
            root_layer.subLayerPaths.remove(self._variation_layer.identifier)
        self._variation_layer = None
        self._prev_edit_target = None

    async def _run_bin_flip_scenario(self):
        rgb_annot = None
        instance_segmentation_annot = None
        rp = None
        completed = False
        try:
            await app_utils.update_app_async()
            print(f"[PalletizingSDGDemo] Running bin flip scenario for bin {self._bin_counter}.")

            self._switch_to_pathtracing(spp=16, total_spp=32)
            await app_utils.update_app_async()
            self._push_variation_layer()
            cam, lights = self._setup_bin_flip_randomization()

            rgb_annot = rep.annotators.get("rgb")
            instance_segmentation_annot = rep.annotators.get("instance_segmentation", init_params={"colorize": True})
            rp = rep.create.render_product(cam, (512, 512))
            rgb_annot.attach(rp)
            instance_segmentation_annot.attach(rp)
            out_dir = os.path.join(self._output_dir, f"annot_bin_{self._bin_counter}")
            os.makedirs(out_dir, exist_ok=True)

            print(f"[PalletizingSDGDemo] Capturing bin flip data for bin {self._bin_counter}.")
            for i in range(self._bin_flip_frames):
                self._randomize_bin_flip_frame(cam, lights, i)
                await rep.orchestrator.step_async(rt_subframes=16, delta_time=0.0)
                write_image(path=os.path.join(out_dir, f"rgb_{i}.png"), data=rgb_annot.get_data())
                instance_data = instance_segmentation_annot.get_data()
                write_image(path=os.path.join(out_dir, f"instance_segmentation_{i}.png"), data=instance_data["data"])
                with open(os.path.join(out_dir, f"instance_segmentation_info_{i}.json"), "w") as f:
                    json.dump(instance_data["info"], f, indent=4)

            await rep.orchestrator.wait_until_complete_async()
            self._bin_flip_scenario_done = True
            completed = True
        except Exception as exc:
            self._error = exc
            return
        finally:
            if rgb_annot is not None:
                rgb_annot.detach()
            if instance_segmentation_annot is not None:
                instance_segmentation_annot.detach()
            if rp is not None:
                rp.destroy()
            if self._stage.GetPrimAtPath("/Replicator"):
                omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])
            self._pop_variation_layer()
            self._switch_to_realtime_pathtracing()
            if not completed:
                self.clear()
        if completed and self._in_running_state:
            await app_utils.update_app_async(steps=3)
            self._resume_overlap_monitoring()

    def _setup_bin_flip_randomization(self):
        lights = rep.functional.create_batch.sphere_light(
            count=3,
            parent="/World",
            name="SdgFlipLight",
            enable_color_temperature=True,
            radius=0.5,
        )
        cam = rep.functional.create.camera(parent="/World", name="SdgFlipCam")
        return cam, lights

    def _randomize_bin_flip_frame(self, cam, lights, frame_index):
        gen = self._rng.generator
        color_palette = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        # Sampled from the viewport at a few viewpoints around the bin.
        camera_positions = [(1.96, 0.72, -0.34), (1.48, 0.70, 0.90), (0.79, -0.86, 0.12), (-0.49, 1.47, 0.58)]
        for light in lights:
            rep.functional.modify.pose(
                light,
                position_value=(
                    float(gen.uniform(0.25, 1.0)),
                    float(gen.uniform(0.25, 1.0)),
                    float(gen.uniform(0.5, 0.75)),
                ),
                scale_value=float(gen.uniform(0.5, 0.8)),
            )
            rep.functional.modify.attribute(light, "inputs:colorTemperature", float(gen.normal(6500, 2000)))
            rep.functional.modify.attribute(light, "inputs:intensity", float(gen.normal(45000, 15000)))
            color = color_palette[int(gen.integers(0, len(color_palette)))]
            rep.functional.modify.attribute(light, "inputs:color", color)
        rep.functional.modify.pose(
            cam,
            position_value=camera_positions[frame_index % len(camera_positions)],
            look_at_value=(0.78, 0.72, -0.1),
        )

    async def _run_pallet_scenario(self):
        writer = None
        rp = None
        completed = False
        try:
            await app_utils.update_app_async()
            print(f"[PalletizingSDGDemo] Running pallet scenario for bin {self._bin_counter}.")
            self._push_variation_layer()
            cam, materials, pallet_mat, texture_paths, pallet_loc = self._setup_bin_and_pallet_randomization()
            out_dir = os.path.join(self._output_dir, f"writer_bin_{self._bin_counter}", "")
            backend = rep.backends.get("DiskBackend")
            backend.initialize(output_dir=out_dir)
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(
                backend=backend, rgb=True, instance_segmentation=True, colorize_instance_segmentation=True
            )
            rp = rep.create.render_product(cam, (512, 512))
            writer.attach(rp)

            print(f"[PalletizingSDGDemo] Capturing pallet data for bin {self._bin_counter}.")
            for i in range(self._pallet_frames):
                self._randomize_bin_and_pallet_frame(cam, materials, pallet_mat, texture_paths, pallet_loc, i)
                await rep.orchestrator.step_async(rt_subframes=16, delta_time=0.0)
            await rep.orchestrator.wait_until_complete_async()

            self._next_bin()
            completed = True
        except Exception as exc:
            self._error = exc
            return
        finally:
            if writer is not None:
                writer.detach()
            if rp is not None:
                rp.destroy()
            if self._stage.GetPrimAtPath("/Replicator"):
                omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])
            self._pop_variation_layer()
            if not completed:
                self.clear()
        if completed and self._in_running_state:
            await app_utils.update_app_async(steps=3)
            self._resume_overlap_monitoring()

    def _setup_bin_and_pallet_randomization(self):
        bin_prims = [
            prim_utils.get_prim_at_path(f"{self.BINS_FOLDER_PATH}/bin_{i}/Visuals/FOF_Mesh_Magenta_Box")
            for i in range(self._bin_counter + 1)
        ]
        pallet = prim_utils.get_prim_at_path(self.PALLET_PRIM_MESH_PATH)
        pallet_loc = tuple(omni.usd.get_world_transform_matrix(pallet).ExtractTranslation())
        materials = rep.functional.create_batch.material(
            mdl="OmniPBR.mdl", count=len(bin_prims), parent="/World", name="SdgBinMat"
        )
        for bin_prim, mat in zip(bin_prims, materials):
            rep.functional.modify.material(bin_prim, mat)
        pallet_mat = rep.functional.create.material(mdl="OmniPBR.mdl", parent="/World", name="SdgPalletMat")
        rep.functional.modify.material(pallet, pallet_mat)
        assets_root_path = get_assets_root_path()
        texture_paths = [
            assets_root_path + "/NVIDIA/Materials/Base/Wood/Oak/Oak_BaseColor.png",
            assets_root_path + "/NVIDIA/Materials/Base/Wood/Ash/Ash_BaseColor.png",
            assets_root_path + "/NVIDIA/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
            assets_root_path + "/NVIDIA/Materials/Base/Wood/Timber/Timber_BaseColor.png",
        ]
        cam = rep.functional.create.camera(parent="/World", name="SdgPalletCam")
        return cam, materials, pallet_mat, texture_paths, pallet_loc

    def _randomize_bin_and_pallet_frame(self, cam, materials, pallet_mat, texture_paths, pallet_loc, frame_index):
        gen = self._rng.generator
        for mat in materials:
            color = tuple(gen.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
            rep.functional.modify.attribute(mat, "inputs:diffuse_color_constant", color)
            roughness = float(gen.choice([0.1, 0.9]))
            rep.functional.modify.attribute(mat, "inputs:reflection_roughness_constant", roughness)
        if frame_index % 2 == 0:
            texture = texture_paths[int(gen.integers(0, len(texture_paths)))]
            rep.functional.modify.attribute(pallet_mat, "inputs:diffuse_texture", texture)
            rep.functional.modify.attribute(pallet_mat, "inputs:project_uvw", True)
            rep.functional.modify.attribute(pallet_mat, "inputs:texture_rotate", float(gen.uniform(80, 95)))
            position = tuple(gen.uniform((0.0, -2.0, 1.0), (2.0, 1.0, 2.0)))
            rep.functional.modify.pose(cam, position_value=position, look_at_value=pallet_loc)

    def _next_bin(self):
        self._bin_counter += 1
        if self._bin_counter >= self._num_captures:
            self.clear()
            print("[PalletizingSDGDemo] Palletizing SDG demo finished.")
            return False
        self._active_bin = None
        print(f"[PalletizingSDGDemo] Moving to bin {self._bin_counter}.")
        self._bin_flip_scenario_done = False
        return True


async def run_example_async(num_captures, bin_flip_frames, pallet_frames, output_dir=None) -> None:
    rep.set_global_seed(42)
    palletizing = Ur10PalletizingSimulation(seed=42)
    await palletizing.load_async()
    palletizing.start()
    await palletizing.wait_for_bin_async("bin_0")

    print(f"[PalletizingSDGDemo] Starting SDG pipeline with {num_captures} bins to capture.")
    sdg_demo = PalletizingSDGDemo(output_dir=output_dir)
    try:
        sdg_demo.start(num_captures, bin_flip_frames, pallet_frames)
        while sdg_demo.is_running():
            await app_utils.update_app_async()
    finally:
        sdg_demo.clear()
        palletizing.clear()
        app_utils.pause()


out_dir = os.path.join(os.getcwd(), "_out_palletizing_sdg_demo")
simulation_app.run_coroutine(
    run_example_async(
        num_captures=args.num_captures,
        bin_flip_frames=DEFAULT_BIN_FLIP_FRAMES,
        pallet_frames=DEFAULT_PALLET_FRAMES,
        output_dir=out_dir,
    )
)

# <start-sdg-ur10-palletizing-test>
if args.test:
    import sys

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.test.utils")
    from isaacsim.test.utils.file_validation import validate_folder_contents

    if not os.path.isdir(out_dir):
        print(f"[PalletizingSDGDemo][Test][FAIL] Output directory is missing: {out_dir}")
        sys.exit(1)
    missing_writers = [
        f"writer_bin_{i}"
        for i in range(args.num_captures)
        if not os.path.isdir(os.path.join(out_dir, f"writer_bin_{i}"))
    ]
    if missing_writers:
        print(f"[PalletizingSDGDemo][Test][FAIL] Missing pallet writer folders in {out_dir}: {missing_writers}")
        sys.exit(1)
    # Flip is path-dependent; only annotator folders that exist for captured bins count.
    num_flips = sum(os.path.isdir(os.path.join(out_dir, f"annot_bin_{i}")) for i in range(args.num_captures))
    # Bin flip (annotators): 2 PNGs + 1 JSON per frame.
    # Pallet (BasicWriter): 2 PNGs + 2 JSONs (mapping + semantics mapping) per frame.
    expected_pngs = (num_flips * DEFAULT_BIN_FLIP_FRAMES * 2) + (args.num_captures * DEFAULT_PALLET_FRAMES * 2)
    expected_jsons = (num_flips * DEFAULT_BIN_FLIP_FRAMES) + (args.num_captures * DEFAULT_PALLET_FRAMES * 2)
    expected_counts = {"png": expected_pngs, "json": expected_jsons}
    ok = validate_folder_contents(
        path=out_dir,
        recursive=True,
        expected_counts=expected_counts,
        fail_on_empty_files=True,
    )
    if not ok:
        print(f"[PalletizingSDGDemo][Test][FAIL] Output validation failed for {out_dir}")
        sys.exit(1)
    print(
        f"[PalletizingSDGDemo][Test][PASS] Output validation succeeded for {out_dir} "
        f"({expected_pngs} pngs, {expected_jsons} jsons, {num_flips}/{args.num_captures} flipped bins)"
    )
# <end-sdg-ur10-palletizing-test>

simulation_app.close()
