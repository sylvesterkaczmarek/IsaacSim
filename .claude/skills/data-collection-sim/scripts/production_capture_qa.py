#!/usr/bin/env python3
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

"""Production headless Replicator capture with QA validation thresholds.

Runs a full SDG pipeline (RGB, depth, segmentation, bbox) then validates
every frame against configurable quality thresholds. Produces a JSON report
with per-frame metrics and an overall PASS/FAIL verdict for CI sign-off.

Usage:
    $ISAAC_SIM_DIR/python.sh production_capture_qa.py --config prod.yaml
    $ISAAC_SIM_DIR/python.sh production_capture_qa.py --num-frames 50 --output-dir /data/sdg_run
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml
from isaacsim import SimulationApp

# --- Default configuration ---------------------------------------------------

DEFAULT_CONFIG = {
    "headless": True,
    "renderer": "RealTimePathTracing",
    "resolution": [1280, 720],
    "rt_subframes": 32,
    "num_frames": 50,
    "seed": 42,
    "env_url": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "output_dir": "_out_production_qa",
    "annotations": {
        "rgb": True,
        "bounding_box_2d_tight": True,
        "semantic_segmentation": True,
        "distance_to_image_plane": True,
        "bounding_box_3d": True,
    },
    "objects": [
        {"url": "/Isaac/Props/Forklift/forklift.usd", "label": "forklift", "count": 1},
        {"url": "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd", "label": "pallet", "count": 3},
        {"url": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd", "label": "cardbox", "count": 6},
        {"url": "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd", "label": "traffic_cone", "count": 4},
    ],
    "camera": {
        "focal_length": 24.0,
        "focus_distance": 400.0,
        "clipping_range": [0.1, 10000000.0],
        "position_min": [-15, -5, 1.5],
        "position_max": [5, 10, 4.0],
    },
    # QA validation thresholds
    "qa_thresholds": {
        "min_mean_rgb": 30,
        "max_mean_rgb": 245,
        "min_rgb_std": 10,
        "min_depth_valid_ratio": 0.5,
        "max_depth_nan_ratio": 0.05,
        "min_segmentation_classes": 1,
        "min_bbox_area_px": 100,
        "min_frames_with_detections_ratio": 0.8,
        "max_frame_failure_ratio": 0.05,
    },
}

# --- CLI ----------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Production SDG capture with QA validation")
parser.add_argument("--config", type=str, default=None, help="YAML config (overrides defaults)")
parser.add_argument("--num-frames", type=int, default=None)
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--skip-capture", action="store_true", help="Only run validation on existing output")
parser.add_argument("--report-path", type=str, default=None, help="Path for JSON QA report")
cli_args, _ = parser.parse_known_args()

config = dict(DEFAULT_CONFIG)
if cli_args.config and os.path.isfile(cli_args.config):
    with open(cli_args.config) as f:
        user_cfg = yaml.safe_load(f) or {}
        if "qa_thresholds" in user_cfg:
            config["qa_thresholds"].update(user_cfg.pop("qa_thresholds"))
        config.update(user_cfg)
if cli_args.num_frames is not None:
    config["num_frames"] = cli_args.num_frames
if cli_args.output_dir is not None:
    config["output_dir"] = cli_args.output_dir
if cli_args.seed is not None:
    config["seed"] = cli_args.seed

output_dir = config["output_dir"]
if not os.path.isabs(output_dir):
    output_dir = os.path.join(os.getcwd(), output_dir)
config["output_dir"] = output_dir

report_path = cli_args.report_path or os.path.join(output_dir, "qa_report.json")


# --- Capture ------------------------------------------------------------------


def run_capture(cfg):
    """Execute headless Replicator capture."""
    import math

    simulation_app = SimulationApp({"renderer": cfg["renderer"], "headless": cfg["headless"]})

    import carb.settings
    import isaacsim.core.experimental.utils.stage as stage_utils
    import omni.replicator.core as rep
    from isaacsim.core.experimental.utils.transform import euler_angles_to_quaternion
    from isaacsim.storage.native import get_assets_root_path

    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
    rep.set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    assets_root = get_assets_root_path()
    if not assets_root:
        print("[QA-CAPTURE] ERROR: Cannot resolve assets root path", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)

    env_url = assets_root + cfg["env_url"]
    print(f"[QA-CAPTURE] Loading stage: {env_url}")
    stage_utils.open_stage(env_url)
    stage = stage_utils.get_current_stage()

    stage.DefinePrim("/SDG", "Scope")

    spawned = {}
    for obj_cfg in cfg["objects"]:
        label = obj_cfg["label"]
        spawned[label] = []
        for i in range(obj_cfg["count"]):
            prim_path = f"/SDG/{label}_{i}"
            stage_utils.add_reference_to_stage(
                usd_path=assets_root + obj_cfg["url"],
                path=prim_path,
            )
            p = stage.GetPrimAtPath(prim_path)
            from pxr import Gf, UsdGeom

            xf = UsdGeom.Xformable(p)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(rng.uniform(-15, 5), rng.uniform(-5, 10), 0))
            quat = euler_angles_to_quaternion([0, 0, rng.uniform(0, 2 * math.pi)]).numpy()
            xf.AddOrientOp().Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))
            rep.functional.modify.semantics(p, {"class": label}, mode="add")
            spawned[label].append(p)

    cam_cfg = cfg["camera"]
    cam = rep.functional.create.camera(
        focal_length=cam_cfg["focal_length"],
        focus_distance=cam_cfg["focus_distance"],
        clipping_range=tuple(cam_cfg["clipping_range"]),
        name="DataCam",
        parent="/SDG",
    )

    resolution = tuple(cfg["resolution"])
    rp = rep.create.render_product(cam, resolution, name="main_view")

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=cfg["output_dir"], **cfg["annotations"])
    writer.attach(rp)

    num_frames = cfg["num_frames"]
    rt_subframes = cfg["rt_subframes"]
    pos_min = np.array(cam_cfg["position_min"])
    pos_max = np.array(cam_cfg["position_max"])
    all_prims = [p for group in spawned.values() for p in group]

    print(f"[QA-CAPTURE] Generating {num_frames} frames (rt_subframes={rt_subframes})")
    t0 = time.time()

    try:
        for i in range(num_frames):
            target = all_prims[rng.integers(0, len(all_prims))]
            rep.functional.modify.pose(
                cam,
                position_value=rng.uniform(pos_min, pos_max).tolist(),
                look_at_value=target,
                look_at_up_axis=(0, 0, 1),
            )
            if i % 5 == 0:
                for label, obj_prims in spawned.items():
                    for op in obj_prims:
                        rep.functional.modify.pose(
                            op,
                            position_value=(rng.uniform(-15, 5), rng.uniform(-5, 10), 0),
                            rotation_value=(0.0, 0.0, math.degrees(rng.uniform(0, 2 * math.pi))),
                        )
            rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed
                print(f"[QA-CAPTURE] Frame {i + 1}/{num_frames} ({fps:.2f} fps)")

        rep.orchestrator.wait_until_complete()
    finally:
        writer.detach()
        rp.destroy()

    elapsed = time.time() - t0
    print(f"[QA-CAPTURE] Done. {num_frames} frames in {elapsed:.1f}s ({num_frames / elapsed:.2f} fps)")
    simulation_app.close()


# --- Validation ---------------------------------------------------------------


def validate_output(output_dir, cfg):
    """Validate captured output against QA thresholds. Returns structured report."""
    from pathlib import Path

    thresholds = cfg["qa_thresholds"]
    num_expected = cfg["num_frames"]

    report = {
        "output_dir": output_dir,
        "config": {k: v for k, v in cfg.items() if k != "qa_thresholds"},
        "thresholds": thresholds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "frame_metrics": [],
        "summary": {},
        "verdict": "PENDING",
        "failures": [],
    }

    out_path = Path(output_dir)
    if not out_path.exists():
        report["verdict"] = "FAIL"
        report["failures"].append("Output directory does not exist")
        return report

    rgb_dir = out_path / "rgb"
    depth_dir = out_path / "distance_to_image_plane"
    seg_dir = out_path / "semantic_segmentation"
    bbox_dir = out_path / "bounding_box_2d_tight"

    rgb_files = sorted(rgb_dir.glob("*.png")) if rgb_dir.exists() else []
    depth_files = sorted(depth_dir.glob("*.npy")) if depth_dir.exists() else []
    seg_files = sorted(seg_dir.glob("*.png")) if seg_dir.exists() else []
    bbox_files = sorted(bbox_dir.glob("*.npy")) if bbox_dir.exists() else []

    # Frame count check
    rgb_count = len(rgb_files)
    if rgb_count < num_expected:
        report["failures"].append(f"Frame count {rgb_count} < expected {num_expected}")

    frames_with_detections = 0
    frame_failures = 0

    for idx in range(rgb_count):
        frame_report = {"frame": idx, "checks": {}, "passed": True}

        # RGB quality
        if idx < len(rgb_files):
            from PIL import Image

            img = np.array(Image.open(rgb_files[idx]))
            mean_rgb = float(img[:, :, :3].mean())
            std_rgb = float(img[:, :, :3].std())
            frame_report["checks"]["mean_rgb"] = mean_rgb
            frame_report["checks"]["std_rgb"] = std_rgb

            if mean_rgb < thresholds["min_mean_rgb"]:
                frame_report["checks"]["rgb_fail"] = "too_dark"
                frame_report["passed"] = False
            elif mean_rgb > thresholds["max_mean_rgb"]:
                frame_report["checks"]["rgb_fail"] = "too_bright"
                frame_report["passed"] = False
            if std_rgb < thresholds["min_rgb_std"]:
                frame_report["checks"]["rgb_fail"] = "low_variance"
                frame_report["passed"] = False

        # Depth quality
        if idx < len(depth_files):
            depth = np.load(depth_files[idx])
            nan_ratio = float(np.isnan(depth).sum() / depth.size)
            valid_ratio = float(np.isfinite(depth).sum() / depth.size)
            frame_report["checks"]["depth_nan_ratio"] = nan_ratio
            frame_report["checks"]["depth_valid_ratio"] = valid_ratio

            if nan_ratio > thresholds["max_depth_nan_ratio"]:
                frame_report["checks"]["depth_fail"] = "too_many_nans"
                frame_report["passed"] = False
            if valid_ratio < thresholds["min_depth_valid_ratio"]:
                frame_report["checks"]["depth_fail"] = "insufficient_valid_pixels"
                frame_report["passed"] = False

        # Segmentation quality
        if idx < len(seg_files):
            from PIL import Image

            seg = np.array(Image.open(seg_files[idx]))
            unique_classes = len(np.unique(seg)) - 1  # subtract background
            frame_report["checks"]["segmentation_classes"] = unique_classes

            if unique_classes < thresholds["min_segmentation_classes"]:
                frame_report["checks"]["seg_fail"] = "no_labeled_objects"
                frame_report["passed"] = False

        # Bounding box quality
        if idx < len(bbox_files):
            bboxes = np.load(bbox_files[idx], allow_pickle=True)
            if hasattr(bboxes, "item"):
                bboxes = bboxes.item()
            has_detection = False
            if isinstance(bboxes, dict) and "data" in bboxes:
                bbox_data = bboxes["data"]
                if len(bbox_data) > 0:
                    has_detection = True
                    areas = []
                    for box in bbox_data:
                        if hasattr(box, "__len__") and len(box) >= 4:
                            w = abs(box[2] - box[0])
                            h = abs(box[3] - box[1])
                            areas.append(w * h)
                    if areas:
                        frame_report["checks"]["min_bbox_area"] = float(min(areas))
                        frame_report["checks"]["max_bbox_area"] = float(max(areas))
                        frame_report["checks"]["num_detections"] = len(areas)
            elif isinstance(bboxes, np.ndarray) and bboxes.size > 0:
                has_detection = True

            if has_detection:
                frames_with_detections += 1

        if not frame_report["passed"]:
            frame_failures += 1
        report["frame_metrics"].append(frame_report)

    # Summary statistics
    detection_ratio = frames_with_detections / max(rgb_count, 1)
    failure_ratio = frame_failures / max(rgb_count, 1)

    report["summary"] = {
        "total_frames": rgb_count,
        "expected_frames": num_expected,
        "frames_passed": rgb_count - frame_failures,
        "frames_failed": frame_failures,
        "failure_ratio": failure_ratio,
        "frames_with_detections": frames_with_detections,
        "detection_ratio": detection_ratio,
        "depth_files": len(depth_files),
        "segmentation_files": len(seg_files),
        "bbox_files": len(bbox_files),
    }

    # Overall verdict
    if detection_ratio < thresholds["min_frames_with_detections_ratio"]:
        report["failures"].append(
            f"Detection ratio {detection_ratio:.2f} < threshold " f"{thresholds['min_frames_with_detections_ratio']}"
        )
    if failure_ratio > thresholds["max_frame_failure_ratio"]:
        report["failures"].append(
            f"Frame failure ratio {failure_ratio:.2f} > threshold " f"{thresholds['max_frame_failure_ratio']}"
        )

    report["verdict"] = "FAIL" if report["failures"] else "PASS"
    return report


# --- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    if not cli_args.skip_capture:
        run_capture(config)

    print(f"\n[QA-VALIDATE] Running validation on: {output_dir}")
    qa_report = validate_output(output_dir, config)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(qa_report, f, indent=2, default=str)

    print(f"[QA-VALIDATE] Report written to: {report_path}")
    print(f"[QA-VALIDATE] Frames: {qa_report['summary'].get('total_frames', 0)}/{config['num_frames']}")
    print(f"[QA-VALIDATE] Passed: {qa_report['summary'].get('frames_passed', 0)}")
    print(f"[QA-VALIDATE] Failed: {qa_report['summary'].get('frames_failed', 0)}")
    print(f"[QA-VALIDATE] Detection ratio: {qa_report['summary'].get('detection_ratio', 0):.2%}")
    print(f"\n{'=' * 40}")
    print(f"  VERDICT: {qa_report['verdict']}")
    print(f"{'=' * 40}")

    if qa_report["failures"]:
        print("\nFailures:")
        for fail in qa_report["failures"]:
            print(f"  - {fail}")

    sys.exit(0 if qa_report["verdict"] == "PASS" else 1)
