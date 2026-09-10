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

"""Verify ROS 2 OmniGraph helper APIs exposed by isaacsim.ros2.nodes."""

from __future__ import annotations

from typing import Any

import omni.graph.core as og
import omni.kit.app
import omni.usd
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.ros2.core.impl.ros2_test_case import ROS2TestCase
from isaacsim.ros2.nodes import (
    RGB_COMPRESSION_OPTIONS,
    Ros2CameraGraphConfig,
    Ros2ClockGraphConfig,
    Ros2GenericPublisherGraphConfig,
    Ros2JointStatesGraphConfig,
    Ros2OdometryGraphConfig,
    Ros2RtxLidarGraphConfig,
    Ros2RtxRadarGraphConfig,
    Ros2TfGraphConfig,
    create_ros2_camera_graph,
    create_ros2_clock_graph,
    create_ros2_generic_publisher_graph,
    create_ros2_joint_states_graph,
    create_ros2_odometry_graph,
    create_ros2_rtx_lidar_graph,
    create_ros2_rtx_radar_graph,
    create_ros2_tf_graph,
)
from pxr import Gf, Sdf, UsdGeom, UsdRender


class TestRos2GraphApi(ROS2TestCase):
    """Verify ROS 2 graph helper creation from the nodes extension."""

    async def setUp(self) -> None:
        """Create a fresh stage for each graph helper test."""
        await super().setUp()
        await stage_utils.create_new_stage_async()
        await omni.kit.app.get_app().next_update_async()

    def _get_nodes_by_type(self, graph_path: str) -> dict[str, list[Any]]:
        """Collect graph nodes keyed by OmniGraph node type."""
        graph = og.get_graph_by_path(graph_path)
        self.assertIsNotNone(graph, f"Graph {graph_path} was not created")

        nodes_by_type: dict[str, list[Any]] = {}
        for node in graph.get_nodes():
            nodes_by_type.setdefault(node.get_type_name(), []).append(node)
        return nodes_by_type

    def _assert_usd_connection(self, source_attr_path: str, target_attr_path: str) -> None:
        """Assert that a USD-authored OmniGraph connection exists."""
        stage = omni.usd.get_context().get_stage()
        target_prim_path, target_attr_name = target_attr_path.split(".", 1)
        target_attr = stage.GetPrimAtPath(target_prim_path).GetAttribute(target_attr_name)
        self.assertTrue(target_attr.IsValid(), f"{target_attr_path} is not a valid attribute")

        source_paths = {str(path) for path in target_attr.GetConnections()}
        self.assertIn(source_attr_path, source_paths)

    def _define_sensor_prim(self, prim_path: str, type_name: str, api_schema: str) -> None:
        """Author a minimal RTX sensor prim so the graph helpers pass prim validation."""
        prim = stage_utils.define_prim(prim_path, type_name)
        prim.ApplyAPI(api_schema)

    def _define_render_product(
        self, render_product_path: str, sensor_path: str, resolution: tuple[int, int] = (320, 240)
    ) -> UsdRender.Product:
        """Author a render product prim targeting a sensor prim."""
        stage = omni.usd.get_context().get_stage()
        render_product = UsdRender.Product.Define(stage, render_product_path)
        render_product.GetResolutionAttr().Set(Gf.Vec2i(resolution[0], resolution[1]))
        render_product.GetCameraRel().SetTargets([Sdf.Path(sensor_path)])
        return render_product

    def _assert_graph_reuses_render_product(self, graph_path: str, render_product_path: str) -> None:
        """Assert a graph attaches to an existing render product instead of creating a new one."""
        graph = og.get_graph_by_path(graph_path)
        self.assertIsNotNone(graph, f"Graph {graph_path} was not created")

        nodes = graph.get_nodes()
        node_types = {node.get_type_name() for node in nodes}
        self.assertIn("isaacsim.core.nodes.IsaacAttachHydraTexture", node_types)
        self.assertNotIn("isaacsim.core.nodes.IsaacCreateRenderProduct", node_types)

        attach_nodes = [node for node in nodes if node.get_type_name() == "isaacsim.core.nodes.IsaacAttachHydraTexture"]
        self.assertEqual(len(attach_nodes), 1, "Expected exactly one AttachHydraTexture node")
        render_product_targets = og.Controller.get(attach_nodes[0].get_attribute("inputs:renderProductPrim"))
        self.assertTrue(render_product_targets)
        self.assertEqual(str(render_product_targets[0]), render_product_path)

    async def test_create_ros2_clock_graph_creates_clock_nodes(self) -> None:
        """Verify clock helper graph structure."""
        graph_path = create_ros2_clock_graph(Ros2ClockGraphConfig(graph_path="/Graph/ROS_Clock"))
        await omni.kit.app.get_app().next_update_async()

        self.assertEqual(graph_path, "/Graph/ROS_Clock")
        nodes_by_type = self._get_nodes_by_type(graph_path)
        self.assertIn("omni.graph.action.OnPlaybackTick", nodes_by_type)
        self.assertIn("isaacsim.core.nodes.IsaacReadSimulationTime", nodes_by_type)
        self.assertIn("isaacsim.ros2.bridge.ROS2PublishClock", nodes_by_type)
        self.assertIn("isaacsim.ros2.bridge.ROS2Context", nodes_by_type)

    async def test_create_ros2_bool_publisher_graph_configures_message_type(self) -> None:
        """Verify generic Bool publisher helper graph configuration."""
        graph_path = create_ros2_generic_publisher_graph(
            Ros2GenericPublisherGraphConfig(graph_path="/Graph/ROS_GenericPub", publisher_kind="bool")
        )
        await omni.kit.app.get_app().next_update_async()

        nodes_by_type = self._get_nodes_by_type(graph_path)
        publisher = nodes_by_type["isaacsim.ros2.bridge.ROS2Publisher"][0]
        self.assertEqual(og.Controller.get(publisher.get_attribute("inputs:messageName")), "Bool")
        self.assertEqual(og.Controller.get(publisher.get_attribute("inputs:messagePackage")), "std_msgs")
        self.assertIn("omni.graph.nodes.ConstantBool", nodes_by_type)

    async def test_create_ros2_camera_graph_configures_rgb_helper(self) -> None:
        """Verify camera helper graph creates render and RGB helper nodes."""
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Camera.Define(stage, "/World/TestCamera")

        graph_path = create_ros2_camera_graph(
            Ros2CameraGraphConfig(
                graph_path="/Graph/ROS_Camera",
                camera_prim="/World/TestCamera",
                frame_id="camera_frame",
                camera_info_topic="camera_info",
                rgb_topic="/rgb/compressed",
                rgb_type="rgb_hevc",
                publish_depth=False,
            )
        )
        await omni.kit.app.get_app().next_update_async()

        nodes_by_type = self._get_nodes_by_type(graph_path)
        camera_helper = nodes_by_type["isaacsim.ros2.bridge.ROS2CameraHelper"][0]
        self.assertIn("isaacsim.core.nodes.IsaacCreateRenderProduct", nodes_by_type)
        self.assertIn("isaacsim.ros2.bridge.ROS2CameraInfoHelper", nodes_by_type)
        self.assertIn(("HEVC", "rgb_hevc", "/rgb/compressed"), RGB_COMPRESSION_OPTIONS)
        self.assertEqual(og.Controller.get(camera_helper.get_attribute("inputs:type")), "rgb_hevc")
        self.assertEqual(og.Controller.get(camera_helper.get_attribute("inputs:topicName")), "/rgb/compressed")

    async def test_create_ros2_camera_graph_reuses_existing_render_product(self) -> None:
        """Verify the camera helper attaches to an existing render product instead of creating one."""
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Camera.Define(stage, "/World/TestCamera")
        render_product_path = "/Render/ReusableCameraRP"
        render_product = self._define_render_product(render_product_path, "/World/TestCamera")

        graph_path = create_ros2_camera_graph(
            Ros2CameraGraphConfig(
                graph_path="/Graph/ROS_CameraReuse",
                camera_prim="/World/TestCamera",
                render_product_prim=render_product_path,
                publish_depth=False,
            )
        )
        await omni.kit.app.get_app().next_update_async()

        self.assertEqual(graph_path, "/Graph/ROS_CameraReuse")
        self._assert_graph_reuses_render_product(graph_path, render_product_path)
        self.assertEqual([str(path) for path in render_product.GetCameraRel().GetTargets()], ["/World/TestCamera"])

    async def test_create_ros2_camera_graph_rejects_mismatched_render_product(self) -> None:
        """Verify the camera helper rejects a render product that targets a different sensor."""
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Camera.Define(stage, "/World/TestCamera")
        UsdGeom.Camera.Define(stage, "/World/OtherCamera")
        render_product_path = "/Render/MismatchedCameraRP"
        self._define_render_product(render_product_path, "/World/OtherCamera")

        with self.assertRaisesRegex(ValueError, "does not target"):
            create_ros2_camera_graph(
                Ros2CameraGraphConfig(
                    graph_path="/Graph/ROS_CameraMismatch",
                    camera_prim="/World/TestCamera",
                    render_product_prim=render_product_path,
                )
            )

    async def test_create_ros2_joint_states_graph_creates_publisher_and_subscriber(self) -> None:
        """Verify joint states helper graph creates publisher, subscriber, and controller nodes."""
        graph_path = create_ros2_joint_states_graph(
            Ros2JointStatesGraphConfig(
                graph_path="/Graph/ROS_JointStates",
                articulation_root="/World/Robot",
                node_namespace="/robot_a",
                publish_joint_states=True,
                subscribe_joint_states=True,
                move_robot_on_subscribe=True,
            )
        )
        await omni.kit.app.get_app().next_update_async()

        self.assertEqual(graph_path, "/Graph/ROS_JointStates")
        nodes_by_type = self._get_nodes_by_type(graph_path)
        self.assertIn("omni.graph.action.OnPlaybackTick", nodes_by_type)
        self.assertIn("isaacsim.ros2.bridge.ROS2Context", nodes_by_type)
        self.assertIn("isaacsim.core.nodes.IsaacReadSimulationTime", nodes_by_type)
        self.assertIn("isaacsim.sensors.physics.IsaacReadJointState", nodes_by_type)
        self.assertIn("isaacsim.ros2.bridge.ROS2PublishJointState", nodes_by_type)
        self.assertIn("isaacsim.ros2.bridge.ROS2SubscribeJointState", nodes_by_type)
        self.assertIn("isaacsim.core.nodes.IsaacArticulationController", nodes_by_type)

        read_joint_state = nodes_by_type["isaacsim.sensors.physics.IsaacReadJointState"][0]
        read_joint_state_targets = og.Controller.get(read_joint_state.get_attribute("inputs:prim"))
        self.assertEqual([str(path) for path in read_joint_state_targets], ["/World/Robot"])

        publisher = nodes_by_type["isaacsim.ros2.bridge.ROS2PublishJointState"][0]
        publisher_targets = og.Controller.get(publisher.get_attribute("inputs:targetPrim"))
        self.assertFalse(publisher_targets)

        self._assert_usd_connection(
            f"{graph_path}/OnPlaybackTick.outputs:tick",
            f"{graph_path}/ReadJointState.inputs:execIn",
        )
        for output_name, input_name in (
            ("execOut", "execIn"),
            ("jointNames", "jointNames"),
            ("jointPositions", "jointPositions"),
            ("jointVelocities", "jointVelocities"),
            ("jointEfforts", "jointEfforts"),
            ("jointDofTypes", "jointDofTypes"),
            ("stageMetersPerUnit", "stageMetersPerUnit"),
            ("sensorTime", "sensorTime"),
        ):
            self._assert_usd_connection(
                f"{graph_path}/ReadJointState.outputs:{output_name}",
                f"{graph_path}/PublisherJointState.inputs:{input_name}",
            )

        # Round-trip the subscriber namespace to guard against the publisher/subscriber attribute mismatch.
        subscriber = nodes_by_type["isaacsim.ros2.bridge.ROS2SubscribeJointState"][0]
        self.assertEqual(og.Controller.get(subscriber.get_attribute("inputs:nodeNamespace")), "/robot_a")

    async def test_create_ros2_tf_graph_creates_transform_publisher(self) -> None:
        """Verify TF helper graph creates transform compute and publisher nodes."""
        graph_path = create_ros2_tf_graph(
            Ros2TfGraphConfig(
                graph_path="/Graph/ROS_TF",
                target_prim="/World/Robot",
                parent_prim="/World",
            )
        )
        await omni.kit.app.get_app().next_update_async()

        self.assertEqual(graph_path, "/Graph/ROS_TF")
        nodes_by_type = self._get_nodes_by_type(graph_path)
        self.assertIn("omni.graph.action.OnPlaybackTick", nodes_by_type)
        self.assertIn("isaacsim.ros2.bridge.ROS2Context", nodes_by_type)
        self.assertIn("isaacsim.core.nodes.IsaacComputeTransformTree", nodes_by_type)
        self.assertIn("isaacsim.ros2.bridge.ROS2PublishTransformTree", nodes_by_type)

    async def test_create_ros2_odometry_graph_creates_odometry_nodes(self) -> None:
        """Verify odometry helper graph creates odometry compute, publisher, and TF nodes."""
        graph_path = create_ros2_odometry_graph(
            Ros2OdometryGraphConfig(
                graph_path="/Graph/ROS_Odometry",
                articulation_root="/World/Robot",
                chassis_prim="/World/Robot/chassis",
            )
        )
        await omni.kit.app.get_app().next_update_async()

        self.assertEqual(graph_path, "/Graph/ROS_Odometry")
        nodes_by_type = self._get_nodes_by_type(graph_path)
        self.assertIn("isaacsim.core.nodes.IsaacComputeOdometry", nodes_by_type)
        self.assertIn("isaacsim.ros2.bridge.ROS2PublishOdometry", nodes_by_type)
        self.assertIn("isaacsim.ros2.bridge.ROS2PublishRawTransformTree", nodes_by_type)
        self.assertEqual(len(nodes_by_type["isaacsim.ros2.bridge.ROS2PublishRawTransformTree"]), 2)
        self.assertIn("isaacsim.ros2.bridge.ROS2PublishTransformTree", nodes_by_type)

    async def test_create_ros2_rtx_lidar_graph_creates_lidar_helper(self) -> None:
        """Verify RTX lidar helper graph creates render product and lidar helper nodes."""
        self._define_sensor_prim("/World/Lidar", "OmniLidar", "OmniSensorGenericLidarCoreAPI")

        graph_path = create_ros2_rtx_lidar_graph(
            Ros2RtxLidarGraphConfig(
                graph_path="/Graph/ROS_LidarRTX",
                lidar_prim="/World/Lidar",
                frame_id="lidar_frame",
            )
        )
        await omni.kit.app.get_app().next_update_async()

        self.assertEqual(graph_path, "/Graph/ROS_LidarRTX")
        nodes_by_type = self._get_nodes_by_type(graph_path)
        self.assertIn("omni.graph.action.OnPlaybackTick", nodes_by_type)
        self.assertIn("isaacsim.core.nodes.IsaacCreateRenderProduct", nodes_by_type)
        self.assertIn("isaacsim.ros2.bridge.ROS2RtxLidarHelper", nodes_by_type)

    async def test_create_ros2_rtx_lidar_graph_reuses_existing_render_product(self) -> None:
        """Verify the RTX lidar helper attaches to an existing render product instead of creating one."""
        self._define_sensor_prim("/World/Lidar", "OmniLidar", "OmniSensorGenericLidarCoreAPI")
        render_product_path = "/Render/ReusableLidarRP"
        render_product = self._define_render_product(render_product_path, "/World/Lidar", (64, 64))

        graph_path = create_ros2_rtx_lidar_graph(
            Ros2RtxLidarGraphConfig(
                graph_path="/Graph/ROS_LidarReuse",
                lidar_prim="/World/Lidar",
                render_product_prim=render_product_path,
                publish_laser_scan=False,
                publish_point_cloud=True,
            )
        )
        await omni.kit.app.get_app().next_update_async()

        self.assertEqual(graph_path, "/Graph/ROS_LidarReuse")
        self._assert_graph_reuses_render_product(graph_path, render_product_path)
        nodes_by_type = self._get_nodes_by_type(graph_path)
        self.assertIn("isaacsim.ros2.bridge.ROS2RtxLidarHelper", nodes_by_type)
        self.assertEqual(render_product.GetResolutionAttr().Get(), Gf.Vec2i(64, 64))
        self.assertEqual([str(path) for path in render_product.GetCameraRel().GetTargets()], ["/World/Lidar"])

    async def test_create_ros2_rtx_radar_graph_creates_radar_helper(self) -> None:
        """Verify RTX radar helper graph creates render product and radar helper nodes."""
        self._define_sensor_prim("/World/Radar", "OmniRadar", "OmniSensorGenericRadarWpmDmatAPI")

        graph_path = create_ros2_rtx_radar_graph(
            Ros2RtxRadarGraphConfig(
                graph_path="/Graph/ROS_RadarRTX",
                radar_prim="/World/Radar",
                frame_id="radar_frame",
            )
        )
        await omni.kit.app.get_app().next_update_async()

        self.assertEqual(graph_path, "/Graph/ROS_RadarRTX")
        nodes_by_type = self._get_nodes_by_type(graph_path)
        self.assertIn("omni.graph.action.OnPlaybackTick", nodes_by_type)
        self.assertIn("isaacsim.core.nodes.IsaacCreateRenderProduct", nodes_by_type)
        self.assertIn("isaacsim.ros2.bridge.ROS2RtxRadarHelper", nodes_by_type)

    async def test_create_ros2_rtx_radar_graph_reuses_existing_render_product(self) -> None:
        """Verify the RTX radar helper attaches to an existing render product instead of creating one."""
        self._define_sensor_prim("/World/Radar", "OmniRadar", "OmniSensorGenericRadarWpmDmatAPI")
        render_product_path = "/Render/ReusableRadarRP"
        render_product = self._define_render_product(render_product_path, "/World/Radar", (64, 64))

        graph_path = create_ros2_rtx_radar_graph(
            Ros2RtxRadarGraphConfig(
                graph_path="/Graph/ROS_RadarReuse",
                radar_prim="/World/Radar",
                render_product_prim=render_product_path,
            )
        )
        await omni.kit.app.get_app().next_update_async()

        self.assertEqual(graph_path, "/Graph/ROS_RadarReuse")
        self._assert_graph_reuses_render_product(graph_path, render_product_path)
        nodes_by_type = self._get_nodes_by_type(graph_path)
        self.assertIn("isaacsim.ros2.bridge.ROS2RtxRadarHelper", nodes_by_type)
        self.assertEqual(render_product.GetResolutionAttr().Get(), Gf.Vec2i(64, 64))
        self.assertEqual([str(path) for path in render_product.GetCameraRel().GetTargets()], ["/World/Radar"])
