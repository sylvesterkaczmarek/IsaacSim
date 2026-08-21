# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ROS 2 PointCloud2 shape and density metadata."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import omni.graph.core as og
import omni.kit.app
import rclpy
from isaacsim.ros2.core.impl.ros2_test_case import ROS2TestCase
from sensor_msgs.msg import PointCloud2


class TestROS2PointCloudShape(ROS2TestCase):
    """Verify flat defaults and explicit organized PointCloud2 output."""

    async def _publish_point_cloud(self, *, organized: bool) -> PointCloud2:
        topic = f"point_cloud_shape_{uuid4().hex}"
        node = self.create_node(f"subscriber_{uuid4().hex}")
        received = None

        def callback(message: PointCloud2) -> None:
            nonlocal received
            received = message

        self.create_subscription(node, PointCloud2, topic, callback)

        # Keep the NumPy allocation alive while the C++ publisher reads dataPtr.
        points = np.arange(18, dtype=np.float32).reshape(6, 3)
        self._point_cloud_shape_test_points = points

        set_values = [
            ("Publisher.inputs:topicName", topic),
            ("Publisher.inputs:frameId", "map"),
            ("Publisher.inputs:dataPtr", int(points.ctypes.data)),
            ("Publisher.inputs:bufferSize", points.nbytes),
            ("Publisher.inputs:cudaDeviceIndex", -1),
        ]
        if organized:
            set_values.extend(
                [
                    ("Publisher.inputs:height", 2),
                    ("Publisher.inputs:width", 3),
                    ("Publisher.inputs:isDense", False),
                ]
            )

        graph_path = f"/PointCloudShape_{uuid4().hex}"
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("Publisher", "isaacsim.ros2.bridge.ROS2PublishPointCloud"),
                ],
                og.Controller.Keys.SET_VALUES: set_values,
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "Publisher.inputs:execIn"),
                ],
            },
        )

        self._timeline.play()

        def spin_ros() -> None:
            rclpy.spin_once(node, timeout_sec=0.01)

        await self.simulate_until_condition(lambda: received is not None, max_frames=120, per_frame_callback=spin_ros)
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        self.assertIsNotNone(received, "PointCloud2 message was not received")
        return received

    async def test_default_point_cloud_remains_flat_and_dense(self) -> None:
        """Existing callers keep the historical 1 x N dense layout."""
        message = await self._publish_point_cloud(organized=False)

        self.assertEqual(message.height, 1)
        self.assertEqual(message.width, 6)
        self.assertTrue(message.is_dense)
        self.assertEqual(message.row_step, message.point_step * 6)
        self.assertEqual(len(message.data), message.row_step)

    async def test_explicit_organized_point_cloud_shape_and_density(self) -> None:
        """Publisher can preserve a two-dimensional point layout and holes."""
        message = await self._publish_point_cloud(organized=True)

        self.assertEqual(message.height, 2)
        self.assertEqual(message.width, 3)
        self.assertFalse(message.is_dense)
        self.assertEqual(message.row_step, message.point_step * 3)
        self.assertEqual(len(message.data), message.row_step * 2)
