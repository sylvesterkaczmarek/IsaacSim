# SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Verify ROS 2 bounding-box publishing.

Covers semantic detections, empty semantics, timestamp selection,
viewpoint-sensitive 2D projection, occlusion, and golden CSV comparisons.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import omni.graph.core as og
import omni.kit.app
import omni.kit.commands
import omni.kit.viewport.utility
import rclpy
import usdrt.Sdf
from isaacsim.core.experimental.objects import Capsule, Cone, Cube, Cylinder, Sphere
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.experimental.utils.semantics import add_labels
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.ros2.core.impl.ros2_image_test_utils import ros2_image_to_buffer
from isaacsim.ros2.core.impl.ros2_test_case import ROS2TestCase
from isaacsim.test.utils.image_io import save_rgb_image
from PIL import Image, ImageDraw
from pxr import Sdf
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, Detection3DArray

from .common import get_qos_profile

# Goldens are bbox CSVs next to this file (not reference renders). Flip UPDATE_GOLDEN_BBOX_CSV locally to refresh;
# SAVE_GOLDEN_IMAGES writes a PNG overlay (magenta = CSV golden, green = live) when true.
_BBOX_GOLDEN_TWO_CUBES = "golden_bbox_two_labeled_cubes_grid_env_1280x720"
_BBOX_GOLDEN_MIXED_PRIMITIVES = "golden_bbox_mixed_sphere_cylinder_cone_capsule_1280x720"
_BBOX_GOLDEN_TWO_CUBES_OCCLUSION_TIGHT = "golden_bbox_two_cubes_occlusion_1280x720_tight"
_BBOX_GOLDEN_TWO_CUBES_OCCLUSION_LOOSE = "golden_bbox_two_cubes_occlusion_1280x720_loose"

UPDATE_GOLDEN_BBOX_CSV = False
SAVE_GOLDEN_IMAGES = False

_GOLDEN_BBOX_PIXEL_DELTA = 3.0
_GOLDEN_BBOX_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "golden", "bbox")
_DEBUG_BBOX_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "debug", "bbox")


def _semantic_id_to_label(semantic_labels_json: str) -> dict[str, str]:
    """Map published semantic IDs to their class label.

    Args:
        semantic_labels_json: JSON payload from a ``semanticLabelsTopicName`` topic.

    Returns:
        Class label for each semantic ID, both as strings.
    """
    mapping: dict[str, str] = {}
    for semantic_id, entry in json.loads(semantic_labels_json).items():
        label = entry.get("class") if isinstance(entry, dict) else entry
        if label:
            mapping[str(semantic_id)] = str(label)
    return mapping


def _detections_by_label(detections: Any, id_to_label: dict[str, str]) -> dict[str, Any]:
    """Index detections by class label rather than by their published semantic ID.

    Args:
        detections: Detections from a ``Detection2DArray`` or ``Detection3DArray`` message.
        id_to_label: Class label for each semantic ID, from :func:`_semantic_id_to_label`.

    Returns:
        Detection for each class label.

    Raises:
        KeyError: If a detection carries a semantic ID that the labels topic did not publish.
    """
    by_label: dict[str, Any] = {}
    for detection in _BBoxSet.sort_detections(detections):
        if not detection.results:
            continue
        semantic_id = str(detection.results[0].hypothesis.class_id)
        if semantic_id not in id_to_label:
            raise KeyError(f"semantic id {semantic_id!r} is missing from the published labels {id_to_label!r}")
        by_label[id_to_label[semantic_id]] = detection
    return by_label


class _BBoxSet:
    """Semantic label to pixel bounding boxes, with CSV I/O and overlay drawing.

    Boxes are keyed by class label rather than by the published semantic ID, because
    semantic IDs are assigned per session and are not stable across runs or renderer
    versions.

    Args:
        boxes: Bounding boxes keyed by class label.
    """

    def __init__(self, boxes: dict[str, tuple[float, float, float, float]]) -> None:
        self._boxes = dict(boxes)

    @staticmethod
    def _sort_key(label: str) -> tuple:
        return (0, int(label)) if label.isdigit() else (1, label)

    @staticmethod
    def sort_detections(detections: Any) -> Any:
        return sorted(detections, key=lambda d: str(d.results[0].hypothesis.class_id) if d.results else "999")

    @classmethod
    def from_detections(cls, detections: Any, id_to_label: dict[str, str]) -> "_BBoxSet":
        boxes: dict[str, tuple[float, float, float, float]] = {}
        for label, detection in _detections_by_label(detections, id_to_label).items():
            bb = detection.bbox
            cx, cy = float(bb.center.position.x), float(bb.center.position.y)
            sx, sy = float(bb.size_x), float(bb.size_y)
            boxes[label] = (cx - 0.5 * sx, cy - 0.5 * sy, cx + 0.5 * sx, cy + 0.5 * sy)
        return cls(boxes)

    @classmethod
    def from_csv(cls, path: str) -> "_BBoxSet":
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        return cls({r["label"].strip(): tuple(float(r[k]) for k in ("x0", "y0", "x1", "y1")) for r in rows})

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["label", "x0", "y0", "x1", "y1"])
            for label in self:
                w.writerow([label, *self._boxes[label]])

    def sorted_xyxy(self) -> list[tuple[float, float, float, float]]:
        return [self._boxes[k] for k in self]

    def labels(self) -> set[str]:
        return set(self._boxes)

    def __getitem__(self, label: str) -> tuple[float, float, float, float]:
        return self._boxes[label]

    def __iter__(self) -> None:
        return iter(sorted(self._boxes, key=self._sort_key))

    def __len__(self) -> int:
        return len(self._boxes)

    def overlay(self, rgb: np.ndarray, color: str = "green") -> np.ndarray:
        return self.draw_on_rgb(rgb, [(color, self)])

    @staticmethod
    def draw_on_rgb(rgb: np.ndarray, layers: list[tuple[str, "_BBoxSet"]]) -> np.ndarray:
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        img = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(img)
        for color, bset in layers:
            for box in bset.sorted_xyxy():
                x0, y0, x1, y1 = (int(round(v)) for v in box)
                draw.rectangle([x0, y0, x1, y1], outline=color, width=1)
        return np.asarray(img, dtype=np.uint8)

    @classmethod
    def save_debug_overlay(cls, rgb: np.ndarray, golden: "_BBoxSet", live: "_BBoxSet", stem: str) -> None:
        os.makedirs(_DEBUG_BBOX_DIR, exist_ok=True)
        name = f"{stem}_bbox_overlay.png"
        img = cls.draw_on_rgb(rgb, [("#ff00ff", golden), ("#00ff00", live)])
        save_rgb_image(img, _DEBUG_BBOX_DIR, name)
        print(f"DEBUG: wrote {os.path.join(_DEBUG_BBOX_DIR, name)}", flush=True)


class TestROS2BboxPublishing(ROS2TestCase):
    """ROS 2 bounding box topics from OmniGraph ROS2CameraHelper nodes."""

    async def setUp(self) -> None:
        """Set the viewport resolution used by bounding-box projection tests."""
        await super().setUp()
        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        viewport_api.set_texture_resolution((1280, 720))
        await omni.kit.app.get_app().next_update_async()

    async def test_bbox(self) -> Any:
        """Geometry for 2D tight/loose and 3D bbox (explicit render product).

        cube_1 partially occludes cube_2, so 2D tight areas are smaller than 2D loose for label "cube1".

        Returns:
            None.
        """
        opened, _ = await stage_utils.open_stage_async(
            self._assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
        )
        self.assertTrue(opened, "Failed to open grid environment stage")

        cube_1 = Cube("/cube_1", positions=[0, -4, 0.5], scales=[1.55, 0.4, 1.0], sizes=1.0)
        cube_2 = Cube("/cube_2", positions=[1.45, -1.9, 0.52], scales=[0.55, 0.55, 0.55], sizes=1.0)
        add_labels(cube_1.prims[0], labels=["Cube0"], taxonomy="class")
        add_labels(cube_2.prims[0], labels=["Cube1"], taxonomy="class")
        cube_3 = Cube("/cube_3", positions=[100, 0, 0], scales=[1, 1, 3], sizes=1.0)
        cube_4 = Cube("/cube_4", positions=[2.4, -0.3, 0.5], scales=[1, 1, 3], sizes=1.0)
        add_labels(cube_3.prims[0], labels=["Cube2"], taxonomy="class")
        add_labels(cube_4.prims[0], labels=["Cube3"], taxonomy="class")
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[0, -6, 0.5], target=[0, 0, 0.5])

        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("Bbox2dTightPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("Bbox2dLoosePublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("Bbox3dPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path("/OmniverseKit_Persp")]),
                        ("CreateRenderProduct.inputs:height", 480),
                        ("CreateRenderProduct.inputs:width", 640),
                        ("Bbox2dTightPublish.inputs:topicName", "bbox_2d_tight"),
                        ("Bbox2dTightPublish.inputs:type", "bbox_2d_tight"),
                        ("Bbox2dTightPublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox2dTightPublish.inputs:enableSemanticLabels", True),
                        ("Bbox2dTightPublish.inputs:semanticLabelsTopicName", "semantic_labels_tight"),
                        ("Bbox2dLoosePublish.inputs:topicName", "bbox_2d_loose"),
                        ("Bbox2dLoosePublish.inputs:type", "bbox_2d_loose"),
                        ("Bbox2dLoosePublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox2dLoosePublish.inputs:enableSemanticLabels", True),
                        ("Bbox2dLoosePublish.inputs:semanticLabelsTopicName", "semantic_labels_loose"),
                        ("Bbox3dPublish.inputs:topicName", "bbox_3d"),
                        ("Bbox3dPublish.inputs:type", "bbox_3d"),
                        ("Bbox3dPublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox3dPublish.inputs:enableSemanticLabels", True),
                        ("Bbox3dPublish.inputs:semanticLabelsTopicName", "semantic_labels_3d"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "Bbox2dTightPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "Bbox2dLoosePublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "Bbox3dPublish.inputs:execIn"),
                        (
                            "CreateRenderProduct.outputs:renderProductPath",
                            "Bbox2dTightPublish.inputs:renderProductPath",
                        ),
                        (
                            "CreateRenderProduct.outputs:renderProductPath",
                            "Bbox2dLoosePublish.inputs:renderProductPath",
                        ),
                        ("CreateRenderProduct.outputs:renderProductPath", "Bbox3dPublish.inputs:renderProductPath"),
                    ],
                },
            )
        except Exception as e:
            print(e)

        await omni.kit.app.get_app().next_update_async()

        received = dict.fromkeys(
            (
                "bbox_2d_tight",
                "bbox_2d_loose",
                "bbox_3d",
                "semantic_data_3d",
                "semantic_data_tight",
                "semantic_data_loose",
            )
        )
        _set = received.__setitem__

        node = self.create_node("bbox_tester")
        qos = get_qos_profile()
        self.create_subscription(node, Detection2DArray, "bbox_2d_tight", lambda d: _set("bbox_2d_tight", d), qos)
        self.create_subscription(node, Detection2DArray, "bbox_2d_loose", lambda d: _set("bbox_2d_loose", d), qos)
        self.create_subscription(node, Detection3DArray, "bbox_3d", lambda d: _set("bbox_3d", d), qos)
        self.create_subscription(node, String, "semantic_labels_3d", lambda d: _set("semantic_data_3d", d), qos)
        self.create_subscription(node, String, "semantic_labels_tight", lambda d: _set("semantic_data_tight", d), qos)
        self.create_subscription(node, String, "semantic_labels_loose", lambda d: _set("semantic_data_loose", d), qos)

        def spin() -> None:
            rclpy.spin_once(node, timeout_sec=0.01)

        await omni.kit.app.get_app().next_update_async()

        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        await self.simulate_until_condition(
            lambda: all(message is not None for message in received.values()),
            max_frames=600,
            per_frame_callback=spin,
        )

        bbox_2d_tight = received["bbox_2d_tight"]
        bbox_2d_loose = received["bbox_2d_loose"]
        bbox_3d = received["bbox_3d"]

        self.assertIsNotNone(bbox_2d_tight)
        self.assertIsNotNone(bbox_2d_loose)
        self.assertIsNotNone(bbox_3d)

        # Semantic IDs are assigned per session and are not even consistent between topics, so every
        # detection is resolved to its class label through the labels topic published alongside it.
        # cube_3 sits outside the frustum, so it contributes no label.
        in_frustum_labels = {"cube0", "cube1", "cube3"}

        # Size and center of each labelled cube in world space, keyed by published class label.
        expected_3d = {
            "cube0": ((1.55, 0.4, 1.0), (0.0, -4.0, 0.5)),
            "cube1": ((0.55, 0.55, 0.55), (1.45, -1.9, 0.52)),
            "cube3": ((1.0, 1.0, 3.0), (2.4, -0.3, 0.5)),
        }
        dets_3d = _detections_by_label(bbox_3d.detections, _semantic_id_to_label(received["semantic_data_3d"].data))
        self.assertEqual(set(dets_3d), in_frustum_labels, msg="3D detections keyed by class label")
        for label, (size, center) in expected_3d.items():
            with self.subTest(kind="bbox_3d", label=label):
                bbox = dets_3d[label].bbox
                for axis, expected in zip("xyz", size):
                    self.assertAlmostEqual(float(getattr(bbox.size, axis)), expected, places=5, msg=f"size {axis}")
                for axis, expected in zip("xyz", center):
                    self.assertAlmostEqual(
                        float(getattr(bbox.center.position, axis)), expected, places=5, msg=f"center {axis}"
                    )

        tight_dets = _detections_by_label(
            bbox_2d_tight.detections, _semantic_id_to_label(received["semantic_data_tight"].data)
        )
        loose_dets = _detections_by_label(
            bbox_2d_loose.detections, _semantic_id_to_label(received["semantic_data_loose"].data)
        )
        self.assertEqual(set(tight_dets), in_frustum_labels, msg="expected three in-frustum class detections (tight)")
        self.assertEqual(set(loose_dets), in_frustum_labels, msg="expected three in-frustum class detections (loose)")

        def _area(detection: Any) -> float:
            return float(detection.bbox.size_x) * float(detection.bbox.size_y)

        for label in sorted(in_frustum_labels):
            with self.subTest(kind="pairing", label=label):
                self.assertLessEqual(
                    _area(tight_dets[label]),
                    _area(loose_dets[label]) + 1.0,
                    msg=f"{label}: tight area should not exceed loose",
                )
        self.assertLess(
            _area(tight_dets["cube1"]),
            0.92 * _area(loose_dets["cube1"]),
            msg="partially occluded cube_2: tight box should be clearly smaller than loose",
        )
        # cube0 (cube_1) is unobstructed; tight and loose boxes match.
        with self.subTest(kind="unoccluded_tight_matches_loose", label="cube0"):
            tight, loose = tight_dets["cube0"].bbox, loose_dets["cube0"].bbox
            self.assertAlmostEqual(float(tight.size_x), float(loose.size_x), delta=1.5)
            self.assertAlmostEqual(float(tight.size_y), float(loose.size_y), delta=1.5)
            self.assertAlmostEqual(float(tight.center.position.x), float(loose.center.position.x), delta=1.5)
            self.assertAlmostEqual(float(tight.center.position.y), float(loose.center.position.y), delta=1.5)

    async def test_empty_semantics(self) -> None:
        """Verifies empty semantic labels don't cause a crash."""
        cube_3 = Cube("/cube_3", positions=[100, 0, 0], scales=[1, 1, 3], sizes=1.0)
        add_labels(cube_3.prims[0], labels=["Cube2"], taxonomy="class")
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[0, -6, 0.5], target=[0, 0, 0.5])

        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        render_product_path = viewport_api.get_render_product_path()

        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("Bbox3dPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("Bbox3dPublish.inputs:renderProductPath", render_product_path),
                        ("Bbox3dPublish.inputs:topicName", "bbox_3d"),
                        ("Bbox3dPublish.inputs:type", "bbox_3d"),
                        ("Bbox3dPublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox3dPublish.inputs:enableSemanticLabels", True),
                        ("Bbox3dPublish.inputs:semanticLabelsTopicName", "semantic_labels"),
                    ],
                    og.Controller.Keys.CONNECT: [("OnPlaybackTick.outputs:tick", "Bbox3dPublish.inputs:execIn")],
                },
            )
        except Exception as e:
            print(e)

        await omni.kit.app.get_app().next_update_async()

        received = {"semantic": None}
        _set = received.__setitem__
        node = self.create_node("bbox_empty_semantics_tester")
        self.create_subscription(node, Detection3DArray, "bbox_3d", lambda _: None, get_qos_profile())
        self.create_subscription(node, String, "semantic_labels", lambda d: _set("semantic", d), get_qos_profile())

        def spin() -> None:
            rclpy.spin_once(node, timeout_sec=0.01)

        await self.simulate_until_condition(lambda: False, max_frames=30)

        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        await self.simulate_until_condition(
            lambda: received["semantic"] is not None,
            max_frames=600,
            per_frame_callback=spin,
        )

        self.assertIsNotNone(received["semantic"])
        semantic_dict = json.loads(received["semantic"].data)
        self.assertIn("time_stamp", semantic_dict)
        self.assertNotIn("0", semantic_dict)

    async def test_bbox_helpers_use_system_time_with_render_product(self) -> None:
        """Bbox-only graph on CreateRenderProduct: stamps use wall time when useSystemTime is set."""
        scene_path = "/Isaac/Environments/Grid/default_environment.usd"
        opened, _ = await stage_utils.open_stage_async(self._assets_root_path + scene_path)
        self.assertTrue(opened, "Failed to open grid environment stage")

        cube_1 = Cube("/cube_1", positions=[0, 0, 0], scales=[1.5, 1, 1], sizes=1.0)
        add_labels(cube_1.prims[0], labels=["Cube0"], taxonomy="class")

        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("Bbox2dTightPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("Bbox2dLoosePublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("Bbox3dPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path("/OmniverseKit_Persp")]),
                        ("CreateRenderProduct.inputs:height", 600),
                        ("CreateRenderProduct.inputs:width", 800),
                        ("Bbox2dTightPublish.inputs:topicName", "bbox_2d_tight"),
                        ("Bbox2dTightPublish.inputs:type", "bbox_2d_tight"),
                        ("Bbox2dTightPublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox2dLoosePublish.inputs:topicName", "bbox_2d_loose"),
                        ("Bbox2dLoosePublish.inputs:type", "bbox_2d_loose"),
                        ("Bbox2dLoosePublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox3dPublish.inputs:topicName", "bbox_3d"),
                        ("Bbox3dPublish.inputs:type", "bbox_3d"),
                        ("Bbox3dPublish.inputs:resetSimulationTimeOnStop", True),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "Bbox2dTightPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "Bbox2dLoosePublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "Bbox3dPublish.inputs:execIn"),
                        (
                            "CreateRenderProduct.outputs:renderProductPath",
                            "Bbox2dTightPublish.inputs:renderProductPath",
                        ),
                        (
                            "CreateRenderProduct.outputs:renderProductPath",
                            "Bbox2dLoosePublish.inputs:renderProductPath",
                        ),
                        ("CreateRenderProduct.outputs:renderProductPath", "Bbox3dPublish.inputs:renderProductPath"),
                    ],
                },
            )
        except Exception as e:
            print(e)

        await omni.kit.app.get_app().next_update_async()

        received = dict.fromkeys(("tight", "loose", "3d"))
        _set = received.__setitem__
        node = self.create_node("bbox_system_time_tester")
        self.create_subscription(node, Detection2DArray, "bbox_2d_tight", lambda d: _set("tight", d), get_qos_profile())
        self.create_subscription(node, Detection2DArray, "bbox_2d_loose", lambda d: _set("loose", d), get_qos_profile())
        self.create_subscription(node, Detection3DArray, "bbox_3d", lambda d: _set("3d", d), get_qos_profile())

        await omni.kit.app.get_app().next_update_async()
        omni.kit.commands.execute(
            "ChangeProperty", prop_path=Sdf.Path("/OmniverseKit_Persp.horizontalAperture"), value=6.0, prev=0
        )

        og.Controller.attribute("/ActionGraph/Bbox2dTightPublish.inputs:useSystemTime").set(True)
        og.Controller.attribute("/ActionGraph/Bbox2dLoosePublish.inputs:useSystemTime").set(True)
        og.Controller.attribute("/ActionGraph/Bbox3dPublish.inputs:useSystemTime").set(True)

        await omni.kit.app.get_app().next_update_async()
        system_time = int(time.time())

        def spin() -> None:
            rclpy.spin_once(node, timeout_sec=0.1)

        self._timeline.play()
        await self.simulate_until_condition(
            lambda: all(v is not None for v in received.values()),
            max_frames=600,
            per_frame_callback=spin,
        )

        self.assertGreaterEqual(received["tight"].header.stamp.sec, system_time)
        self.assertGreaterEqual(received["loose"].header.stamp.sec, system_time)
        self.assertGreaterEqual(received["3d"].header.stamp.sec, system_time)

    async def test_bbox_2d_tight_projection_differs_across_viewpoints(self) -> None:
        """2D tight boxes change in image space when the perspective camera orbits the scene."""
        cube_1 = Cube("/cube_1", positions=[2, 0, 0], scales=[1.5, 1, 1], sizes=1.0)
        cube_2 = Cube("/cube_2", positions=[-1.5, 0, 0], scales=[1, 2, 1], sizes=1.0)
        add_labels(cube_1.prims[0], labels=["Cube0"], taxonomy="class")
        add_labels(cube_2.prims[0], labels=["Cube1"], taxonomy="class")

        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        render_product_path = viewport_api.get_render_product_path()

        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("Bbox2dTightPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("Bbox2dTightPublish.inputs:renderProductPath", render_product_path),
                        ("Bbox2dTightPublish.inputs:topicName", "bbox_2d_tight"),
                        ("Bbox2dTightPublish.inputs:type", "bbox_2d_tight"),
                        ("Bbox2dTightPublish.inputs:resetSimulationTimeOnStop", True),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "Bbox2dTightPublish.inputs:execIn"),
                    ],
                },
            )
        except Exception as e:
            print(e)

        await omni.kit.app.get_app().next_update_async()

        received = {"bbox": None}
        node = self.create_node("bbox_multi_view_tester")
        self.create_subscription(
            node,
            Detection2DArray,
            "bbox_2d_tight",
            lambda d: received.__setitem__("bbox", d),
            get_qos_profile(),
        )

        def spin() -> None:
            rclpy.spin_once(node, timeout_sec=0.01)

        eyes = [
            [0, -6, 0.5],
            [6, 0, 0.5],
            [-5.5, -3.0, 2.0],
        ]
        fingerprints = []
        target = [0, 0, 0.5]

        for eye in eyes:
            self._timeline.stop()
            await omni.kit.app.get_app().next_update_async()
            ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=eye, target=target)
            await self.simulate_until_condition(lambda: False, max_frames=5)
            # Drain queued Detection2DArray from the previous camera pose; otherwise the first
            # message after play() can be stale and fingerprints repeat across viewpoints.
            for _ in range(128):
                spin()
            received["bbox"] = None
            self._timeline.play()
            await omni.kit.app.get_app().next_update_async()
            ok = await self.simulate_until_condition(
                lambda: received["bbox"] is not None and len(received["bbox"].detections) >= 1,
                max_frames=600,
                per_frame_callback=spin,
            )
            self.assertTrue(ok, f"Timed out waiting for 2D bbox at eye={eye}")
            dets = _BBoxSet.sort_detections(received["bbox"].detections)
            fingerprints.append(
                tuple(
                    (
                        str(d.results[0].hypothesis.class_id) if d.results else "?",
                        round(float(d.bbox.center.position.x), 2),
                        round(float(d.bbox.center.position.y), 2),
                        round(float(d.bbox.size_x), 2),
                        round(float(d.bbox.size_y), 2),
                    )
                    for d in dets
                )
            )

        self._timeline.stop()
        self.assertEqual(len(fingerprints), len(eyes))
        for i in range(len(eyes)):
            for j in range(i + 1, len(eyes)):
                self.assertNotEqual(
                    fingerprints[i],
                    fingerprints[j],
                    msg=f"poses {eyes[i]} and {eyes[j]} produced identical 2D tight fingerprints {fingerprints[i]!r}",
                )

    def _assert_live_matches_golden(self, *, golden: _BBoxSet, live: _BBoxSet, stem: str, rgb: np.ndarray) -> None:
        if SAVE_GOLDEN_IMAGES:
            _BBoxSet.save_debug_overlay(rgb, golden, live, stem)
        self.assertEqual(golden.labels(), live.labels(), msg=f"{stem}: label mismatch")
        for label in golden:
            with self.subTest(stem=stem, label=label):
                for axis, (g, lv) in enumerate(zip(golden[label], live[label])):
                    self.assertAlmostEqual(
                        g, lv, delta=_GOLDEN_BBOX_PIXEL_DELTA, msg=f"{stem} label={label} axis={axis}"
                    )

    async def _assert_2d_tight_boxes_match_golden_csv(
        self,
        *,
        golden_stem: str,
        expected_num_detections: int,
        setup_labeled_geometry: Callable[[], None],
    ) -> None:
        csv_path = os.path.join(_GOLDEN_BBOX_DIR, f"{golden_stem}.csv")
        rgb_topic = f"{golden_stem}_rgb"
        bbox_topic = f"{golden_stem}_bbox"
        labels_topic = f"{golden_stem}_labels"
        if not UPDATE_GOLDEN_BBOX_CSV:
            self.assertTrue(os.path.isfile(csv_path), f"Missing {csv_path}. Set UPDATE_GOLDEN_BBOX_CSV=True.")

        scene_path = "/Isaac/Environments/Grid/default_environment.usd"
        opened, _ = await stage_utils.open_stage_async(self._assets_root_path + scene_path)
        self.assertTrue(opened, "Failed to open grid environment stage")
        await omni.kit.app.get_app().next_update_async()

        setup_labeled_geometry()
        await omni.kit.app.get_app().next_update_async()

        graph_path = f"/ActionGraph_{golden_stem}"
        try:
            og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("Bbox2dTightPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path("/OmniverseKit_Persp")]),
                        ("CreateRenderProduct.inputs:height", 720),
                        ("CreateRenderProduct.inputs:width", 1280),
                        ("RGBPublish.inputs:topicName", rgb_topic),
                        ("RGBPublish.inputs:type", "rgb"),
                        ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox2dTightPublish.inputs:topicName", bbox_topic),
                        ("Bbox2dTightPublish.inputs:type", "bbox_2d_tight"),
                        ("Bbox2dTightPublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox2dTightPublish.inputs:enableSemanticLabels", True),
                        ("Bbox2dTightPublish.inputs:semanticLabelsTopicName", labels_topic),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "Bbox2dTightPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                        (
                            "CreateRenderProduct.outputs:renderProductPath",
                            "Bbox2dTightPublish.inputs:renderProductPath",
                        ),
                    ],
                },
            )
        except Exception as e:
            print(e)

        await omni.kit.app.get_app().next_update_async()

        received = {"rgb": None, "bbox": None, "labels": None}
        _set = received.__setitem__
        node = self.create_node(f"node_{golden_stem}")
        self.create_subscription(node, RosImage, rgb_topic, lambda d: _set("rgb", d), get_qos_profile())
        self.create_subscription(node, Detection2DArray, bbox_topic, lambda d: _set("bbox", d), get_qos_profile())
        self.create_subscription(node, String, labels_topic, lambda d: _set("labels", d), get_qos_profile())

        def spin() -> None:
            rclpy.spin_once(node, timeout_sec=0.1)

        self._timeline.play()
        await self.simulate_until_condition(
            lambda: all(v is not None for v in received.values())
            and len(received["bbox"].detections) >= expected_num_detections,
            max_frames=600,
            per_frame_callback=spin,
        )
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        det_msg = f"{golden_stem}: expected {expected_num_detections} 2D tight detections"
        self.assertEqual(len(received["bbox"].detections), expected_num_detections, det_msg)
        rgb_array = ros2_image_to_buffer(
            received["rgb"],
            normalize_color_order=True,
            squeeze_singleton_channel=True,
            copy=True,
        )
        live = _BBoxSet.from_detections(received["bbox"].detections, _semantic_id_to_label(received["labels"].data))

        if UPDATE_GOLDEN_BBOX_CSV:
            os.makedirs(_GOLDEN_BBOX_DIR, exist_ok=True)
            live.to_csv(csv_path)
            print(f"UPDATE_GOLDEN_BBOX_CSV: wrote {csv_path}", flush=True)
            return

        self._assert_live_matches_golden(
            golden=_BBoxSet.from_csv(csv_path),
            live=live,
            stem=golden_stem,
            rgb=rgb_array,
        )

    async def test_two_cube_bbox_pipeline_rgb_prerendered(self) -> None:
        """2D tight boxes (pixel xyxy) must match golden CSV within a few pixels."""

        def setup() -> None:
            c1 = Cube("/cube_1", positions=[2, 0, 0], scales=[1.5, 1, 1], sizes=1.0)
            c2 = Cube("/cube_2", positions=[-1.5, 0, 0], scales=[1, 2, 1], sizes=1.0)
            add_labels(c1.prims[0], labels=["Cube0"], taxonomy="class")
            add_labels(c2.prims[0], labels=["Cube1"], taxonomy="class")
            ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[0, -6, 0.5], target=[0, 0, 0.5])

        await self._assert_2d_tight_boxes_match_golden_csv(
            golden_stem=_BBOX_GOLDEN_TWO_CUBES,
            expected_num_detections=2,
            setup_labeled_geometry=setup,
        )

    async def test_mixed_primitives_bbox_pipeline_rgb_prerendered(self) -> None:
        """Mixed primitives scene: same as two-cube test with four tight boxes."""

        def setup() -> None:
            s = Sphere(
                "/bbox_golden/mixed_sphere",
                positions=[-2.3, 0.35, 0.48],
                radii=0.52,
                colors=np.array([0.92, 0.22, 0.18]),
            )
            cy = Cylinder(
                "/bbox_golden/mixed_cylinder",
                positions=[0.0, 0.0, 0.62],
                radii=0.36,
                heights=1.05,
                colors=np.array([0.2, 0.82, 0.32]),
            )
            co = Cone(
                "/bbox_golden/mixed_cone",
                positions=[2.15, -0.25, 0.52],
                radii=0.41,
                heights=0.95,
                colors=np.array([0.22, 0.38, 0.92]),
            )
            ca = Capsule(
                "/bbox_golden/mixed_capsule",
                positions=[-0.4, 1.55, 0.62],
                radii=0.26,
                heights=0.9,
                colors=np.array([0.92, 0.86, 0.2]),
            )
            add_labels(s.prims[0], labels=["Mix0"], taxonomy="class")
            add_labels(cy.prims[0], labels=["Mix1"], taxonomy="class")
            add_labels(co.prims[0], labels=["Mix2"], taxonomy="class")
            add_labels(ca.prims[0], labels=["Mix3"], taxonomy="class")
            ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[0, -7.25, 1.15], target=[0, 0, 0.55])

        await self._assert_2d_tight_boxes_match_golden_csv(
            golden_stem=_BBOX_GOLDEN_MIXED_PRIMITIVES,
            expected_num_detections=4,
            setup_labeled_geometry=setup,
        )

    async def test_two_cube_occlusion_bbox_pipeline_rgb_prerendered_tight_loose(self) -> None:
        """Partial occlusion: tight and loose 2D boxes each match their CSV; green-only overlays must differ."""
        tight_csv = os.path.join(_GOLDEN_BBOX_DIR, f"{_BBOX_GOLDEN_TWO_CUBES_OCCLUSION_TIGHT}.csv")
        loose_csv = os.path.join(_GOLDEN_BBOX_DIR, f"{_BBOX_GOLDEN_TWO_CUBES_OCCLUSION_LOOSE}.csv")
        if not UPDATE_GOLDEN_BBOX_CSV:
            self.assertTrue(os.path.isfile(tight_csv), f"Missing {tight_csv}. Set UPDATE_GOLDEN_BBOX_CSV=True.")
            self.assertTrue(os.path.isfile(loose_csv), f"Missing {loose_csv}. Set UPDATE_GOLDEN_BBOX_CSV=True.")

        scene_path = "/Isaac/Environments/Grid/default_environment.usd"
        opened, _ = await stage_utils.open_stage_async(self._assets_root_path + scene_path)
        self.assertTrue(opened, "Failed to open grid environment stage")
        await omni.kit.app.get_app().next_update_async()

        c1 = Cube("/cube_1", positions=[0, -4, 0.5], scales=[1.55, 0.4, 1.0], sizes=1.0)
        c2 = Cube("/cube_2", positions=[1.45, -1.9, 0.52], scales=[0.55, 0.55, 0.55], sizes=1.0)
        add_labels(c1.prims[0], labels=["Cube0"], taxonomy="class")
        add_labels(c2.prims[0], labels=["Cube1"], taxonomy="class")
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[0, -6, 0.5], target=[0, 0, 0.5])
        await omni.kit.app.get_app().next_update_async()

        rgb_topic = "bbox_occlusion_rgb"
        tight_topic = "bbox_occlusion_tight"
        loose_topic = "bbox_occlusion_loose"
        tight_labels_topic = "bbox_occlusion_tight_labels"
        loose_labels_topic = "bbox_occlusion_loose_labels"
        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph_occlusion_tight_loose", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("Bbox2dTightPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("Bbox2dLoosePublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path("/OmniverseKit_Persp")]),
                        ("CreateRenderProduct.inputs:height", 720),
                        ("CreateRenderProduct.inputs:width", 1280),
                        ("RGBPublish.inputs:topicName", rgb_topic),
                        ("RGBPublish.inputs:type", "rgb"),
                        ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox2dTightPublish.inputs:topicName", tight_topic),
                        ("Bbox2dTightPublish.inputs:type", "bbox_2d_tight"),
                        ("Bbox2dTightPublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox2dTightPublish.inputs:enableSemanticLabels", True),
                        ("Bbox2dTightPublish.inputs:semanticLabelsTopicName", tight_labels_topic),
                        ("Bbox2dLoosePublish.inputs:topicName", loose_topic),
                        ("Bbox2dLoosePublish.inputs:type", "bbox_2d_loose"),
                        ("Bbox2dLoosePublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox2dLoosePublish.inputs:enableSemanticLabels", True),
                        ("Bbox2dLoosePublish.inputs:semanticLabelsTopicName", loose_labels_topic),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "Bbox2dTightPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "Bbox2dLoosePublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                        (
                            "CreateRenderProduct.outputs:renderProductPath",
                            "Bbox2dTightPublish.inputs:renderProductPath",
                        ),
                        (
                            "CreateRenderProduct.outputs:renderProductPath",
                            "Bbox2dLoosePublish.inputs:renderProductPath",
                        ),
                    ],
                },
            )
        except Exception as e:
            print(e)

        await omni.kit.app.get_app().next_update_async()

        received = dict.fromkeys(("rgb", "tight", "loose", "tight_labels", "loose_labels"))
        _set = received.__setitem__
        node = self.create_node("bbox_occlusion_tester")
        self.create_subscription(node, RosImage, rgb_topic, lambda d: _set("rgb", d), get_qos_profile())
        self.create_subscription(node, Detection2DArray, tight_topic, lambda d: _set("tight", d), get_qos_profile())
        self.create_subscription(node, Detection2DArray, loose_topic, lambda d: _set("loose", d), get_qos_profile())
        self.create_subscription(node, String, tight_labels_topic, lambda d: _set("tight_labels", d), get_qos_profile())
        self.create_subscription(node, String, loose_labels_topic, lambda d: _set("loose_labels", d), get_qos_profile())

        def spin() -> None:
            rclpy.spin_once(node, timeout_sec=0.1)

        self._timeline.play()
        await self.simulate_until_condition(
            lambda: all(v is not None for v in received.values())
            and len(received["tight"].detections) >= 2
            and len(received["loose"].detections) >= 2,
            max_frames=600,
            per_frame_callback=spin,
        )
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        self.assertEqual(len(received["tight"].detections), 2, "occlusion: expected 2 tight detections")
        self.assertEqual(len(received["loose"].detections), 2, "occlusion: expected 2 loose detections")

        rgb_array = ros2_image_to_buffer(
            received["rgb"],
            normalize_color_order=True,
            squeeze_singleton_channel=True,
            copy=True,
        )
        tight_live = _BBoxSet.from_detections(
            received["tight"].detections, _semantic_id_to_label(received["tight_labels"].data)
        )
        loose_live = _BBoxSet.from_detections(
            received["loose"].detections, _semantic_id_to_label(received["loose_labels"].data)
        )
        self.assertFalse(
            np.array_equal(tight_live.overlay(rgb_array), loose_live.overlay(rgb_array)),
            "occlusion scene: tight and loose overlays must differ so both goldens are meaningful",
        )

        if UPDATE_GOLDEN_BBOX_CSV:
            os.makedirs(_GOLDEN_BBOX_DIR, exist_ok=True)
            tight_live.to_csv(tight_csv)
            loose_live.to_csv(loose_csv)
            print(f"UPDATE_GOLDEN_BBOX_CSV: wrote {tight_csv} and {loose_csv}", flush=True)
            return

        with self.subTest(overlay="tight"):
            self._assert_live_matches_golden(
                golden=_BBoxSet.from_csv(tight_csv),
                live=tight_live,
                stem=_BBOX_GOLDEN_TWO_CUBES_OCCLUSION_TIGHT,
                rgb=rgb_array,
            )
        with self.subTest(overlay="loose"):
            self._assert_live_matches_golden(
                golden=_BBoxSet.from_csv(loose_csv),
                live=loose_live,
                stem=_BBOX_GOLDEN_TWO_CUBES_OCCLUSION_LOOSE,
                rgb=rgb_array,
            )
