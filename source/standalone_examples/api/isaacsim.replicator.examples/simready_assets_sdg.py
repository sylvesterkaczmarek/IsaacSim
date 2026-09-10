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

"""Demonstrate synthetic data generation using SimReady assets with randomized scenes."""

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import argparse
import asyncio
import os
import sys
import time
from typing import Any

import carb.settings
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.bounds as bounds_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.replicator.core as rep
from isaacsim.core.experimental.utils.semantics import upgrade_prim_semantics_to_labels
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Sdf, Usd, UsdPhysics

NUM_SCENARIOS = 5
SEED = 34

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-scenarios", type=int, default=NUM_SCENARIOS, help="Number of randomization scenarios to create"
)
args, _ = parser.parse_known_args()
num_scenarios = args.num_scenarios

if not app_utils.is_extension_enabled("omni.simready.explorer"):
    app_utils.enable_extension("omni.simready.explorer")
import omni.simready.explorer as sre


def enable_simready_explorer() -> None:
    """Enable the SimReady Explorer window if not already open."""
    if sre.get_instance().browser_model is None:
        import omni.kit.actions.core as actions

        actions.execute_action("omni.simready.explorer", "toggle_window")


def set_prim_variants(prim: Usd.Prim, variants: dict[str, str]) -> None:
    """Set variant selections on a prim from a dictionary of variant set names to values.

    Args:
        prim: USD prim containing the variant sets to update.
        variants: Desired selection keyed by variant-set name.
    """
    vsets = prim.GetVariantSets()
    for name, value in variants.items():
        vset = vsets.GetVariantSet(name)
        if vset:
            vset.SetVariantSelection(value)


async def search_assets_async() -> tuple[list, list, list]:
    """Search for SimReady assets (tables, dishes, items) asynchronously.

    Returns:
        Separate table, dish, and food-item search result lists.
    """
    print(f"[SDG] Searching for SimReady assets...")
    start_time = time.time()
    tables = await sre.find_assets(["table", "furniture"])
    print(f"[SDG]   - Found {len(tables)} tables ({time.time() - start_time:.2f}s)")
    start_time = time.time()
    plates = await sre.find_assets(["plate"])
    print(f"[SDG]   - Found {len(plates)} plates ({time.time() - start_time:.2f}s)")
    start_time = time.time()
    bowls = await sre.find_assets(["bowl"])
    print(f"[SDG]   - Found {len(bowls)} bowls ({time.time() - start_time:.2f}s)")
    dishes = plates + bowls
    start_time = time.time()
    fruits = await sre.find_assets(["fruit"])
    print(f"[SDG]   - Found {len(fruits)} fruits ({time.time() - start_time:.2f}s)")
    start_time = time.time()
    vegetables = await sre.find_assets(["vegetable"])
    print(f"[SDG]   - Found {len(vegetables)} vegetables ({time.time() - start_time:.2f}s)")
    items = fruits + vegetables
    return tables, dishes, items


def search_assets() -> tuple[list, list, list]:
    """Run SimReady asset search while pumping the app (required in standalone SimulationApp).

    Returns:
        Separate table, dish, and food-item search result lists.
    """
    search_task = asyncio.ensure_future(search_assets_async())
    while not search_task.done():
        app_utils.update_app()
    return search_task.result()


def run_simready_randomization(
    stage: Usd.Stage,
    camera_prim: Usd.Prim,
    render_product: Any,
    tables: list,
    dishes: list,
    items: list,
    rng: np.random.Generator | None = None,
) -> None:
    """Arrange SimReady tableware, settle it with physics, and capture an RGB frame.

    Args:
        stage: Stage on which to author the temporary randomized asset layer.
        camera_prim: Camera to position above the selected dish.
        render_product: Render product to enable while capturing the scenario.
        tables: SimReady table search results from which to select one asset.
        dishes: SimReady dish search results from which to select one asset.
        items: SimReady food-item search results from which to select two to four assets.
        rng: NumPy random generator used for asset and pose selection, or None to create an unseeded generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    print(f"[SDG]   Creating anonymous variation layer for the randomizations...")
    root_layer = stage.GetRootLayer()
    variation_layer = Sdf.Layer.CreateAnonymous("variation")
    root_layer.subLayerPaths.insert(0, variation_layer.identifier)
    stage.SetEditTarget(variation_layer)

    variants = {"PhysicsVariant": "RigidBody"}
    rep.functional.create.scope(name="Assets")

    print(f"[SDG]   Loading assets...")
    table_asset = tables[rng.integers(len(tables))]
    start_time = time.time()
    table_prim = rep.functional.create.reference(usd_path=table_asset.main_url, parent="/Assets", name=table_asset.name)
    set_prim_variants(table_prim, variants)
    upgrade_prim_semantics_to_labels(table_prim)
    print(f"[SDG]     - Table: '{table_asset.name}' ({time.time() - start_time:.2f}s)")
    app_utils.update_app()

    UsdPhysics.RigidBodyAPI(table_prim).GetRigidBodyEnabledAttr().Set(False)

    bbox_cache = bounds_utils.create_bbox_cache()
    table_aabb = bounds_utils.compute_aabb(table_prim, bbox_cache=bbox_cache, space="world")
    table_extent = table_aabb[3:] - table_aabb[:3]

    dish_asset = dishes[rng.integers(len(dishes))]
    start_time = time.time()
    dish_prim = rep.functional.create.reference(usd_path=dish_asset.main_url, parent="/Assets", name=dish_asset.name)
    set_prim_variants(dish_prim, variants)
    upgrade_prim_semantics_to_labels(dish_prim)
    print(f"[SDG]     - Dish: '{dish_asset.name}' ({time.time() - start_time:.2f}s)")
    app_utils.update_app()

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
        start_time = time.time()
        item_prim = rep.functional.create.reference(
            usd_path=item_asset.main_url, parent="/Assets", name=item_asset.name
        )
        set_prim_variants(item_prim, variants)
        upgrade_prim_semantics_to_labels(item_prim)
        print(f"[SDG]     - Item: '{item_asset.name}' ({time.time() - start_time:.2f}s)")
        item_prims.append(item_prim)
        app_utils.update_app()

    print(f"[SDG]   Positioning assets on table...")
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

    num_sim_steps = 25
    print(f"[SDG]   Running physics simulation ({num_sim_steps} steps)...")
    SimulationManager.invalidate_physics()
    SimulationManager.initialize_physics()
    SimulationManager.step(steps=num_sim_steps)

    print(f"[SDG]   Setting edit target to root layer...")
    stage.SetEditTarget(root_layer)

    print(f"[SDG]   Positioning camera and capturing frame...")
    camera_position = (
        dish_position[0] + rng.uniform(-0.5, 0.5),
        dish_position[1] + rng.uniform(-0.5, 0.5),
        dish_position[2] + 1.5 + rng.uniform(-0.5, 0.5),
    )
    rep.functional.modify.pose(
        camera_prim, position_value=camera_position, look_at_value=dish_prim, look_at_up_axis=(0, 0, 1)
    )
    render_product.hydra_texture.set_updates_enabled(True)
    rep.orchestrator.step(delta_time=0.0, rt_subframes=16)
    render_product.hydra_texture.set_updates_enabled(False)

    print(f"[SDG]   Removing temp variation layer...")
    variation_layer.Clear()
    root_layer.subLayerPaths.remove(variation_layer.identifier)


def run_simready_randomizations(num_scenarios: int, output_dir: str | None = None) -> None:
    """Run multiple SimReady randomization scenarios and capture the results.

    Args:
        num_scenarios: Number of independently randomized scenes to capture.
        output_dir: Directory for captured RGB images, or None to use the example output directory.
    """
    print(f"[SDG] Initializing scene...")
    stage_utils.create_new_stage()
    stage = stage_utils.get_current_stage(backend="usd")

    rng = np.random.default_rng(SEED)
    rep.set_global_seed(SEED)

    rep.orchestrator.set_capture_on_play(False)
    SimulationManager.set_physics_dt(1.0 / 60.0)

    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    print(f"[SDG] Setting up lighting...")
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
    rep.functional.create.distant_light(intensity=2500, parent="/World", name="DistantLight", rotation=(-75, 0, 0))

    enable_simready_explorer()

    tables, dishes, items = search_assets()

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "_out_simready_assets")
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=output_dir)
    writer = rep.writers.get("BasicWriter")
    print(f"[SDG] Initializing writer, output directory: {output_dir}...")
    writer.initialize(backend=backend, rgb=True)

    print(f"[SDG] Creating camera and render product...")
    camera_prim = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
    rp = rep.create.render_product(camera_prim, (512, 512))
    rp.hydra_texture.set_updates_enabled(False)
    writer.attach(rp)

    for i in range(num_scenarios):
        print(f"[SDG] Scenario {i + 1}/{num_scenarios}")
        run_simready_randomization(
            stage=stage, camera_prim=camera_prim, render_product=rp, tables=tables, dishes=dishes, items=items, rng=rng
        )

    print("[SDG] Wait for the data to be written and cleanup render products...")
    rep.orchestrator.wait_until_complete()
    writer.detach()
    rp.destroy()


print(f"[SDG] Starting SDG pipeline with {num_scenarios} scenarios...")
run_simready_randomizations(num_scenarios)

# <start-simready-assets-sdg-test>
test_parser = argparse.ArgumentParser()
test_parser.add_argument(
    "--test",
    action="store_true",
    help="Validate captured output files against expected counts and exit.",
)
test_args, _ = test_parser.parse_known_args()

if test_args.test:
    import sys

    import omni.kit.app
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.test.utils")
    enable_extension("isaacsim.replicator.examples")
    from isaacsim.test.utils.file_validation import get_folder_file_summary, validate_folder_contents
    from isaacsim.test.utils.image_comparison import compare_images_in_directories

    rgb_mean_diff_tolerance = 7.5
    replicator_examples_ext_path = (
        omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module("isaacsim.replicator.examples")
    )
    golden_dir = os.path.join(
        replicator_examples_ext_path,
        "isaacsim",
        "replicator",
        "examples",
        "tests",
        "data",
        "golden",
        "_out_simready_assets",
    )
    out_dir = os.path.join(os.getcwd(), "_out_simready_assets")
    ok = validate_folder_contents(
        path=out_dir,
        recursive=True,
        expected_counts={"png": num_scenarios},
        fail_on_empty_files=True,
    )
    if not ok:
        summary = get_folder_file_summary(out_dir, recursive=True)
        print(
            f"[SDG][Test][FAIL] Output validation failed for {out_dir}: "
            f"expected png={num_scenarios}, found {summary}"
        )
        sys.exit(1)

    rgb_result = compare_images_in_directories(
        golden_dir=golden_dir,
        test_dir=out_dir,
        path_pattern=r"^rgb_.*\.png$",
        allclose_rtol=None,
        allclose_atol=None,
        mean_tolerance=rgb_mean_diff_tolerance,
        print_all_stats=False,
    )
    if not rgb_result["all_passed"]:
        print(
            f"[SDG][Test][FAIL] RGB image comparison failed (tol={rgb_mean_diff_tolerance}). "
            f"Golden dir: {golden_dir}, output dir: {out_dir}"
        )
        sys.exit(1)
    print(f"[SDG][Test][PASS] Output validation succeeded for {out_dir}")
# <end-simready-assets-sdg-test>

simulation_app.close()
