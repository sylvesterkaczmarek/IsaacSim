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

"""User interface for generating ROS2 camera, RTX lidar, and RTX radar sensor OmniGraph action graphs."""

import isaacsim.core.experimental.utils.prim as prim_utils
import omni.kit.viewport.utility
import omni.ui as ui
from isaacsim.gui.components.widgets import ParamWidget, SelectPrimWidget
from isaacsim.ros2.nodes import (
    LIDAR_METADATA_WITHOUT_POINT_CLOUD_WARNING as ROS2_LIDAR_METADATA_WITHOUT_POINT_CLOUD_WARNING,
)
from isaacsim.ros2.nodes import LIDAR_POINT_CLOUD_METADATA_OPTIONS as ROS2_LIDAR_POINT_CLOUD_METADATA_OPTIONS
from isaacsim.ros2.nodes import RADAR_POINT_CLOUD_METADATA_OPTIONS as ROS2_RADAR_POINT_CLOUD_METADATA_OPTIONS
from isaacsim.ros2.nodes import (
    RADAR_RADIAL_VELOCITY_AUX_OUTPUT_WARNING as ROS2_RADAR_RADIAL_VELOCITY_AUX_OUTPUT_WARNING,
)
from isaacsim.ros2.nodes import RGB_COMPRESSION_OPTIONS as ROS2_RGB_COMPRESSION_OPTIONS
from isaacsim.ros2.nodes import (
    Ros2CameraGraphConfig,
    Ros2RtxLidarGraphConfig,
    Ros2RtxRadarGraphConfig,
    create_ros2_camera_graph,
    create_ros2_rtx_lidar_graph,
    create_ros2_rtx_radar_graph,
    radar_supports_basic_aux_output,
)
from omni.kit.notification_manager import NotificationStatus, post_notification
from omni.kit.window.extensions import SimpleCheckBox
from pxr import UsdGeom, UsdRender

from .og_common import (
    Ros2GraphWindow,
    add_ok_cancel_buttons,
    add_script_docs_footer,
    validate_existing_graph_path,
)


def _check_render_product_prim(render_product_prim: str, sensor_prim: str) -> bool:
    if not render_product_prim:
        return True

    prim = prim_utils.get_prim_at_path(render_product_prim)
    if not prim.IsValid() or not prim.IsA(UsdRender.Product):
        post_notification(
            render_product_prim + " is not a valid render product prim, check the render product prim",
            status=NotificationStatus.WARNING,
        )
        return False

    camera_targets = UsdRender.Product(prim).GetCameraRel().GetTargets()
    if not camera_targets or str(camera_targets[0]) != sensor_prim:
        post_notification(
            render_product_prim + " does not target the selected sensor prim, check the render product prim",
            status=NotificationStatus.WARNING,
        )
        return False

    return True


class Ros2CameraGraph(Ros2GraphWindow):
    """A UI window for generating ROS2 camera graphs in Isaac Sim.

    This class provides a graphical interface to create or extend OmniGraph action graphs that publish camera data to ROS2 topics. It supports multiple camera output types including RGB, depth, point clouds, semantic segmentation, instance segmentation, and 2D/3D bounding boxes.

    The generated graph can either be created as a new standalone graph or added to an existing graph. It automatically configures the necessary nodes including tick sources, render products, ROS2 contexts, and topic publishers based on user selections.

    Users can configure:

    - Graph path and camera prim selection
    - ROS2 frame ID and node namespace
    - Publication of RGB images with optional H.264 or HEVC compression
    - Publication of depth images
    - Publication of depth point clouds
    - Publication of instance segmentation data
    - Publication of semantic segmentation data
    - Publication of 2D tight bounding boxes
    - Publication of 2D loose bounding boxes
    - Publication of 3D bounding boxes
    - Custom topic names for each output type

    The window validates the selected camera prim to ensure it is a valid UsdGeom.Camera before generating the graph. If adding to an existing graph, it verifies the graph contains required nodes such as OnPlaybackTick and ROS2Context.
    """

    RGB_COMPRESSION_OPTIONS = ROS2_RGB_COMPRESSION_OPTIONS
    """RGB compression options as (display name, camera helper type, default topic)."""

    def __init__(self) -> None:
        super().__init__("ROS2 Camera Graph", width=500, height=630)
        # Initialize parameters
        self._og_path = "/Graph/ROS_Camera"
        self._camera_prim = "/OmniverseKit_Persp"  # default camera prim is the perspective camera
        self._render_product_prim = ""
        self._add_to_existing_graph = False
        self._frame_id = "sim_camera"
        self._node_namespace = ""
        self._camera_info_topic = "camera_info"
        self._rgb_pub = True
        self._rgb_topic = "/rgb"
        self._rgb_compression_index = 0
        self._depth_pub = True
        self._depth_topic = "/depth"
        self._depth_pcl_pub = False
        self._depth_pcl_topic = "/depth_pcl"
        self._instance_pub = False
        self._instance_topic = "/instance_segmentation"
        self._semantic_pub = False
        self._semantic_topic = "/semantic_segmentation"
        self._bbox2d_tight_pub = False
        self._bbox2d_tight_topic = "/bbox_2d_tight"
        self._bbox2d_loose_pub = False
        self._bbox2d_loose_topic = "/bbox_2d_loose"
        self._bbox3d_pub = False
        self._bbox3d_topic = "/bbox_3d"
        # Build the UI automatically from here
        self._build_ui()

    def make_graph(self) -> None:
        """Create a ROS2 camera graph in OmniGraph based on the configured parameters.

        If adding to an existing graph, validates and reuses existing nodes (tick, context, render product).
        Otherwise, creates a new graph with base nodes. Adds publisher nodes for each enabled camera output type
        (RGB, depth, depth point cloud, instance segmentation, semantic segmentation, 2D/3D bounding boxes) and
        connects them to the render product and ROS2 context.
        """
        try:
            self._og_path = create_ros2_camera_graph(
                Ros2CameraGraphConfig(
                    graph_path=self._og_path,
                    camera_prim=self._camera_prim,
                    frame_id=self._frame_id,
                    node_namespace=self._node_namespace,
                    camera_info_topic=self._camera_info_topic,
                    add_to_existing_graph=self._add_to_existing_graph,
                    render_product_prim=self._render_product_prim,
                    publish_rgb=self._rgb_pub,
                    rgb_topic=self._rgb_topic,
                    rgb_type=self._get_rgb_type(),
                    publish_depth=self._depth_pub,
                    depth_topic=self._depth_topic,
                    publish_depth_point_cloud=self._depth_pcl_pub,
                    depth_point_cloud_topic=self._depth_pcl_topic,
                    publish_instance_segmentation=self._instance_pub,
                    instance_segmentation_topic=self._instance_topic,
                    publish_semantic_segmentation=self._semantic_pub,
                    semantic_segmentation_topic=self._semantic_topic,
                    publish_bbox_2d_tight=self._bbox2d_tight_pub,
                    bbox_2d_tight_topic=self._bbox2d_tight_topic,
                    publish_bbox_2d_loose=self._bbox2d_loose_pub,
                    bbox_2d_loose_topic=self._bbox2d_loose_topic,
                    publish_bbox_3d=self._bbox3d_pub,
                    bbox_3d_topic=self._bbox3d_topic,
                )
            )
        except (ValueError, RuntimeError) as exc:
            post_notification(str(exc), status=NotificationStatus.WARNING)

    def _build_ui(self) -> None:
        """Construct the user interface for configuring the ROS2 camera graph.

        Creates UI elements for graph path, camera prim selection, frame ID, node namespace, and topic
        configuration. Includes checkboxes to enable/disable publishing for each camera output type (RGB, depth,
        depth point cloud, instance, semantic, bounding boxes) with corresponding topic name fields. Adds OK/Cancel
        buttons and documentation links.
        """
        og_path_def = ParamWidget.FieldDef(
            name="og_path", label="Graph Path", type=ui.StringField, default=self._og_path
        )
        frame_id_def = ParamWidget.FieldDef(
            name="frame_id", label="Frame ID", type=ui.StringField, default=self._frame_id
        )
        node_namespace_def = ParamWidget.FieldDef(
            name="node_namespace", label="Node Namespace", type=ui.StringField, default=self._node_namespace
        )
        rgb_topic_def = ParamWidget.FieldDef(
            name="rgb topic", label="RGB Topic", type=ui.StringField, default=self._rgb_topic
        )
        depth_topic_def = ParamWidget.FieldDef(
            name="depth topic", label="Depth Topic", type=ui.StringField, default=self._depth_topic
        )
        depth_pcl_topic_def = ParamWidget.FieldDef(
            name="depth_pcl topic", label="Depth PCL Topic", type=ui.StringField, default=self._depth_pcl_topic
        )
        instance_topic_def = ParamWidget.FieldDef(
            name="instance topic", label="Instance Topic", type=ui.StringField, default=self._instance_topic
        )
        semantic_topic_def = ParamWidget.FieldDef(
            name="semantic topic", label="Semantic Topic", type=ui.StringField, default=self._semantic_topic
        )
        bbox2d_tight_topic_def = ParamWidget.FieldDef(
            name="bbox2d_tight topic", label="Bbox2d Tight Topic", type=ui.StringField, default=self._bbox2d_tight_topic
        )
        bbox2d_loose_topic_def = ParamWidget.FieldDef(
            name="bbox2d_loose topic", label="Bbox2d Loose Topic", type=ui.StringField, default=self._bbox2d_loose_topic
        )
        bbox3d_topic_def = ParamWidget.FieldDef(
            name="bbox3d topic", label="Bbox3d Topic", type=ui.StringField, default=self._bbox3d_topic
        )

        with self.frame:
            with ui.VStack(spacing=4):
                with ui.HStack():
                    ui.Label("Add to an existing graph?", width=ui.Percent(30))
                    cb = ui.SimpleBoolModel(default_value=self._add_to_existing_graph)
                    SimpleCheckBox(self._add_to_existing_graph, self._on_use_existing_graph, model=cb)
                self.og_path_input = ParamWidget(field_def=og_path_def)
                self.camera_prim_input = SelectPrimWidget(label="Camera Prim", default=self._camera_prim)
                self.render_product_prim_input = SelectPrimWidget(
                    label="Render Product Prim (Optional)", default=self._render_product_prim
                )
                self.frame_id_input = ParamWidget(field_def=frame_id_def)
                self.node_namespace_input = ParamWidget(field_def=node_namespace_def)
                ui.Spacer(height=5)
                with ui.HStack():
                    ui.Label("RGB", width=ui.Percent(15))
                    cb = ui.SimpleBoolModel(default_value=self._rgb_pub)
                    SimpleCheckBox(self._rgb_pub, self._on_rgb_pub, model=cb)
                    ui.Label("Compression", width=ui.Percent(17), word_wrap=True)
                    self.rgb_compression_combo = ui.ComboBox(
                        self._rgb_compression_index,
                        *(option[0] for option in self.RGB_COMPRESSION_OPTIONS),
                        width=ui.Percent(18),
                        identifier="ros2_camera_rgb_compression",
                    )
                    ui.Spacer(width=ui.Percent(2))
                    self.rgb_topic_input = ParamWidget(field_def=rgb_topic_def)
                    self.rgb_compression_combo.model.add_item_changed_fn(self._on_rgb_compression_changed)
                with ui.HStack():
                    ui.Label("Depth", width=ui.Percent(15))
                    cb = ui.SimpleBoolModel(default_value=self._depth_pub)
                    SimpleCheckBox(self._depth_pub, self._on_depth_pub, model=cb)
                    ui.Spacer(width=ui.Percent(5))
                    self.depth_topic_input = ParamWidget(field_def=depth_topic_def)
                with ui.HStack():
                    ui.Label("Depth Point Clouds", width=ui.Percent(15), word_wrap=True)
                    cb = ui.SimpleBoolModel(default_value=self._depth_pcl_pub)
                    SimpleCheckBox(self._depth_pcl_pub, self._on_depth_pcl_pub, model=cb)
                    ui.Spacer(width=ui.Percent(5))
                    self.depth_pcl_topic_input = ParamWidget(field_def=depth_pcl_topic_def)
                with ui.HStack():
                    ui.Label("Instance", width=ui.Percent(15))
                    cb = ui.SimpleBoolModel(default_value=self._instance_pub)
                    SimpleCheckBox(self._instance_pub, self._on_instance_pub, model=cb)
                    ui.Spacer(width=ui.Percent(5))
                    self.instance_topic_input = ParamWidget(field_def=instance_topic_def)
                with ui.HStack():
                    ui.Label("Semantic", width=ui.Percent(15), word_wrap=True)
                    cb = ui.SimpleBoolModel(default_value=self._semantic_pub)
                    SimpleCheckBox(self._semantic_pub, self._on_semantic_pub, model=cb)
                    ui.Spacer(width=ui.Percent(5))
                    self.semantic_topic_input = ParamWidget(field_def=semantic_topic_def)
                with ui.HStack():
                    ui.Label("BoundingBox2D Tight", width=ui.Percent(15), word_wrap=True)
                    cb = ui.SimpleBoolModel(default_value=self._bbox2d_tight_pub)
                    SimpleCheckBox(self._bbox2d_tight_pub, self._on_bbox2d_tight_pub, model=cb)
                    ui.Spacer(width=ui.Percent(5))
                    self.bbox2d_tight_topic_input = ParamWidget(field_def=bbox2d_tight_topic_def)
                with ui.HStack():
                    ui.Label("BoundingBox2D Loose", width=ui.Percent(15), word_wrap=True)
                    cb = ui.SimpleBoolModel(default_value=self._bbox2d_loose_pub)
                    SimpleCheckBox(self._bbox2d_loose_pub, self._on_bbox2d_loose_pub, model=cb)
                    ui.Spacer(width=ui.Percent(5))
                    self.bbox2d_loose_topic_input = ParamWidget(field_def=bbox2d_loose_topic_def)
                with ui.HStack():
                    ui.Label("BoundingBox3D", width=ui.Percent(15), word_wrap=True)
                    cb = ui.SimpleBoolModel(default_value=self._bbox3d_pub)
                    SimpleCheckBox(self._bbox3d_pub, self._on_bbox3d_pub, model=cb)
                    ui.Spacer(width=ui.Percent(5))
                    self.bbox3d_topic_input = ParamWidget(field_def=bbox3d_topic_def)

                add_ok_cancel_buttons(self._on_ok, self._on_cancel)
                add_script_docs_footer(
                    __file__,
                    "https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_camera.html#graph-shortcut",
                )

        return

    def _on_ok(self) -> None:
        """Handle the OK button click event.

        Retrieves all parameter values from UI widgets, validates them using parameter check, and if validation
        passes, generates the ROS2 camera graph and closes the window. Otherwise, displays a warning notification.
        """
        self._og_path = self.og_path_input.get_value()
        self._camera_prim = self.camera_prim_input.get_value()
        self._render_product_prim = self.render_product_prim_input.get_value()
        self._frame_id = self.frame_id_input.get_value()
        self._node_namespace = self.node_namespace_input.get_value()
        self._rgb_topic = self.rgb_topic_input.get_value()
        self._rgb_compression_index = self.rgb_compression_combo.model.get_item_value_model().get_value_as_int()
        self._depth_topic = self.depth_topic_input.get_value()
        self._depth_pcl_topic = self.depth_pcl_topic_input.get_value()
        self._instance_topic = self.instance_topic_input.get_value()
        self._semantic_topic = self.semantic_topic_input.get_value()
        self._bbox2d_tight_topic = self.bbox2d_tight_topic_input.get_value()
        self._bbox2d_loose_topic = self.bbox2d_loose_topic_input.get_value()
        self._bbox3d_topic = self.bbox3d_topic_input.get_value()

        self._finish_on_ok()

    def _check_params(self) -> bool:
        """Validate the configured parameters before graph creation.

        When adding to an existing graph, verifies the graph path points to a valid OmniGraph prim. Validates that
        the camera prim path points to a valid UsdGeom.Camera prim. Displays warning notifications for invalid
        parameters.

        Returns:
            True if all parameters are valid, False otherwise.

        """
        if self._add_to_existing_graph and not validate_existing_graph_path(self._og_path):
            return False

        # check if the camera prim is valid
        camera_prim = prim_utils.get_prim_at_path(self._camera_prim)
        if camera_prim.IsValid() and camera_prim.IsA(UsdGeom.Camera):
            return _check_render_product_prim(self._render_product_prim, self._camera_prim)

        msg = self._camera_prim + " is not a valid camera prim, check the camera prim"
        post_notification(msg, status=NotificationStatus.WARNING)
        return False

    def _on_rgb_pub(self, check_state: bool) -> None:
        """Handle the checkbox state change for RGB publishing.

        Args:
            check_state: Whether the checkbox is checked.

        """
        self._rgb_pub = check_state

    def _get_rgb_type(self) -> str:
        """Get the camera helper type for the selected RGB compression option.

        Returns:
            ROS 2 camera helper type for RGB output.
        """
        return self.RGB_COMPRESSION_OPTIONS[self._rgb_compression_index][1]

    def _get_rgb_default_topic(self) -> str:
        """Get the default topic for the selected RGB compression option.

        Returns:
            Default RGB topic name.
        """
        return self.RGB_COMPRESSION_OPTIONS[self._rgb_compression_index][2]

    def _on_rgb_compression_changed(self, model: ui.AbstractItemModel, _item: object) -> None:
        """Handle the RGB compression dropdown selection change.

        Args:
            model: ComboBox model containing the selected compression index.
            _item: ComboBox item emitted by the UI callback.

        """
        old_topic = self._rgb_topic
        self._rgb_compression_index = model.get_item_value_model().get_value_as_int()
        if self.rgb_topic_input.get_value() in {option[2] for option in self.RGB_COMPRESSION_OPTIONS}:
            self._rgb_topic = self._get_rgb_default_topic()
            if self._rgb_topic != old_topic:
                self.rgb_topic_input.set_value(self._rgb_topic)

    def _on_depth_pub(self, check_state: bool) -> None:
        """Handle the checkbox state change for depth publishing.

        Args:
            check_state: Whether the checkbox is checked.

        """
        self._depth_pub = check_state

    def _on_depth_pcl_pub(self, check_state: bool) -> None:
        """Handle the checkbox state change for depth point cloud publishing.

        Args:
            check_state: Whether the checkbox is checked.

        """
        self._depth_pcl_pub = check_state

    def _on_instance_pub(self, check_state: bool) -> None:
        """Handle the checkbox state change for instance segmentation publishing.

        Args:
            check_state: Whether the checkbox is checked.

        """
        self._instance_pub = check_state

    def _on_semantic_pub(self, check_state: bool) -> None:
        """Handle the semantic segmentation publishing checkbox state change.

        Args:
            check_state: The new state of the semantic segmentation checkbox.

        """
        self._semantic_pub = check_state

    def _on_bbox2d_tight_pub(self, check_state: bool) -> None:
        """Handle the 2D tight bounding box publishing checkbox state change.

        Args:
            check_state: The new state of the 2D tight bounding box checkbox.

        """
        self._bbox2d_tight_pub = check_state

    def _on_bbox2d_loose_pub(self, check_state: bool) -> None:
        """Handle the 2D loose bounding box publishing checkbox state change.

        Args:
            check_state: The new state of the 2D loose bounding box checkbox.

        """
        self._bbox2d_loose_pub = check_state

    def _on_bbox3d_pub(self, check_state: bool) -> None:
        """Handle the 3D bounding box publishing checkbox state change.

        Args:
            check_state: The new state of the 3D bounding box checkbox.

        """
        self._bbox3d_pub = check_state


class Ros2RtxLidarGraph(Ros2GraphWindow):
    """A UI helper window for generating ROS2 action graphs for RTX lidar sensors.

    This window provides an interface to configure and generate OmniGraph action graphs that publish RTX lidar data to ROS2 topics. It supports both creating new graphs and adding nodes to existing graphs. The generated graph can publish laser scan messages and point cloud messages with configurable metadata fields. Users can select which point cloud metadata to include such as intensity, timestamp, emitter ID, channel ID, material ID, tick ID, hit normal, velocity, object ID, echo ID, and tick state.
    """

    # Point cloud metadata options: (display_name, attribute_name)
    # attribute_name corresponds to the input on ROS2RtxLidarPointCloudConfig node
    METADATA_OPTIONS = ROS2_LIDAR_POINT_CLOUD_METADATA_OPTIONS
    """Point cloud metadata options available for selection.

    Each tuple contains (display_name, attribute_name) where display_name is shown in the UI and attribute_name
    corresponds to the input on the ROS2RtxLidarPointCloudConfig node.
    """

    def __init__(self) -> None:
        super().__init__("ROS2 RTX Lidar Graph", width=400, height=680)
        self._og_path = "/Graph/ROS_LidarRTX"
        self._frame_id = "sim_lidar"
        self._node_namespace = ""
        self._add_to_existing_graph = False
        self._lidar_prim = ""
        self._render_product_prim = ""
        self._laser_scan_pub = True
        self._laser_scan_topic = "/laser_scan"
        self._point_cloud_pub = False
        self._point_cloud_topic = "/point_cloud"

        # Point cloud metadata options - dictionary keyed by attribute name
        self._metadata_selected = {attr_name: False for _, attr_name in self.METADATA_OPTIONS}

        # build UI
        self._build_ui()

    def make_graph(self) -> None:
        """Create or modifies an action graph for ROS2 RTX Lidar publishing.

        Generates a new graph or extends an existing one to publish RTX Lidar data. The graph includes nodes for laser scan and/or point cloud publishing based on the configured settings. When point cloud metadata options are selected, a configuration node is created to control which metadata fields are included in the published point cloud messages.
        """
        metadata = {attr for attr, selected in self._metadata_selected.items() if selected}
        if metadata and not self._point_cloud_pub:
            post_notification(ROS2_LIDAR_METADATA_WITHOUT_POINT_CLOUD_WARNING, status=NotificationStatus.WARNING)
        try:
            self._og_path = create_ros2_rtx_lidar_graph(
                Ros2RtxLidarGraphConfig(
                    graph_path=self._og_path,
                    lidar_prim=self._lidar_prim,
                    frame_id=self._frame_id,
                    node_namespace=self._node_namespace,
                    add_to_existing_graph=self._add_to_existing_graph,
                    render_product_prim=self._render_product_prim,
                    publish_laser_scan=self._laser_scan_pub,
                    laser_scan_topic=self._laser_scan_topic,
                    publish_point_cloud=self._point_cloud_pub,
                    point_cloud_topic=self._point_cloud_topic,
                    metadata=metadata,
                )
            )
        except (ValueError, RuntimeError) as exc:
            post_notification(str(exc), status=NotificationStatus.WARNING)

    def _build_ui(self) -> None:
        """Build the user interface for the RTX Lidar graph configuration window.

        Creates input fields for graph path, lidar prim selection, frame ID, node namespace, and topic names. Includes checkboxes for enabling laser scan and point cloud publishing, along with a two-column layout for selecting point cloud metadata options.
        """
        og_path_def = ParamWidget.FieldDef(
            name="og_path", label="Graph Path", type=ui.StringField, default=self._og_path
        )
        frame_id_def = ParamWidget.FieldDef(
            name="frame_id", label="Frame ID", type=ui.StringField, default=self._frame_id
        )
        node_namespace_def = ParamWidget.FieldDef(
            name="node_namespace", label="Node Namespace", type=ui.StringField, default=self._node_namespace
        )
        laser_scan_topic_def = ParamWidget.FieldDef(
            name="laser_scan_topic", label="LaserScan Topic", type=ui.StringField, default=self._laser_scan_topic
        )
        point_cloud_topic_def = ParamWidget.FieldDef(
            name="point_cloud_topic", label="Point Cloud Topic", type=ui.StringField, default=self._point_cloud_topic
        )

        with self.frame:
            with ui.VStack(spacing=4):
                with ui.HStack():
                    ui.Label("Add to an existing graph?", width=ui.Percent(30))
                    cb = ui.SimpleBoolModel(default_value=self._add_to_existing_graph)
                    SimpleCheckBox(self._add_to_existing_graph, self._on_use_existing_graph, model=cb)
                self.og_path_input = ParamWidget(field_def=og_path_def)
                self.lidar_prim_input = SelectPrimWidget(label="Lidar Prim", default=self._lidar_prim)
                self.render_product_prim_input = SelectPrimWidget(
                    label="Render Product Prim (Optional)", default=self._render_product_prim
                )
                self.frame_id_input = ParamWidget(field_def=frame_id_def)
                self.node_namespace_input = ParamWidget(field_def=node_namespace_def)
                ui.Spacer(height=5)
                with ui.HStack():
                    ui.Label("Laser Scan", width=ui.Percent(15))
                    cb = ui.SimpleBoolModel(default_value=self._laser_scan_pub)
                    SimpleCheckBox(self._laser_scan_pub, self._on_laser_scan_pub, model=cb)
                    ui.Spacer(width=ui.Percent(5))
                    self.laser_scan_topic_input = ParamWidget(field_def=laser_scan_topic_def)
                with ui.HStack():
                    ui.Label("Point Cloud", width=ui.Percent(15))
                    cb = ui.SimpleBoolModel(default_value=self._point_cloud_pub)
                    SimpleCheckBox(self._point_cloud_pub, self._on_point_cloud_pub, model=cb)
                    ui.Spacer(width=ui.Percent(5))
                    self.point_cloud_topic_input = ParamWidget(field_def=point_cloud_topic_def)

                # Point Cloud Metadata Section
                ui.Spacer(height=5)
                ui.Label("Point Cloud Metadata", word_wrap=True)
                with ui.HStack():
                    # Split metadata options into two columns
                    mid_point = (len(self.METADATA_OPTIONS) + 1) // 2
                    left_options = self.METADATA_OPTIONS[:mid_point]
                    right_options = self.METADATA_OPTIONS[mid_point:]

                    with ui.VStack(spacing=2):
                        for display_name, attr_name in left_options:
                            with ui.HStack():
                                ui.Label(display_name, width=ui.Percent(30))
                                cb = ui.SimpleBoolModel(default_value=self._metadata_selected[attr_name])
                                SimpleCheckBox(
                                    self._metadata_selected[attr_name],
                                    lambda checked, attr=attr_name: self._on_metadata_changed(attr, checked),
                                    model=cb,
                                )
                    with ui.VStack(spacing=2):
                        for display_name, attr_name in right_options:
                            with ui.HStack():
                                ui.Label(display_name, width=ui.Percent(30))
                                cb = ui.SimpleBoolModel(default_value=self._metadata_selected[attr_name])
                                SimpleCheckBox(
                                    self._metadata_selected[attr_name],
                                    lambda checked, attr=attr_name: self._on_metadata_changed(attr, checked),
                                    model=cb,
                                )

                add_ok_cancel_buttons(self._on_ok, self._on_cancel)
                add_script_docs_footer(
                    __file__,
                    "https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_rtx_lidar.html#graph-shortcut",
                )

        return

    def _on_ok(self) -> None:
        """Handle the OK button click event.

        Collects values from all UI input fields, validates the parameters, and generates the graph if validation passes. Closes the window upon successful graph generation.
        """
        self._og_path = self.og_path_input.get_value()
        self._lidar_prim = self.lidar_prim_input.get_value()
        self._render_product_prim = self.render_product_prim_input.get_value()
        self._frame_id = self.frame_id_input.get_value()
        self._node_namespace = self.node_namespace_input.get_value()
        self._laser_scan_topic = self.laser_scan_topic_input.get_value()
        self._point_cloud_topic = self.point_cloud_topic_input.get_value()

        self._finish_on_ok()

    def _check_params(self) -> bool:
        """Validate the graph and lidar prim parameters.

        Verifies that the specified graph path exists if adding to an existing graph, and confirms that the lidar prim is a valid RTX lidar (either a Camera with IsaacRtxLidarSensorAPI or an OmniLidar with OmniSensorGenericLidarCoreAPI). Displays warning notifications for any validation failures.

        Returns:
            True if all parameters are valid, False otherwise.

        """
        if self._add_to_existing_graph and not validate_existing_graph_path(self._og_path):
            return False

        # check if the lidar prim is valid
        lidar_prim = prim_utils.get_prim_at_path(self._lidar_prim)
        if lidar_prim.IsValid():
            if lidar_prim.GetTypeName() == "OmniLidar" and lidar_prim.HasAPI("OmniSensorGenericLidarCoreAPI"):
                return _check_render_product_prim(self._render_product_prim, self._lidar_prim)

        msg = self._lidar_prim + " is not a valid RTX lidar prim, check the lidar prim"
        post_notification(msg, status=NotificationStatus.WARNING)
        return False

    def _on_laser_scan_pub(self, check_state: bool) -> None:
        """Handle the checkbox state change for laser scan publishing.

        Args:
            check_state: Whether to enable laser scan publishing.

        """
        self._laser_scan_pub = check_state

    def _on_point_cloud_pub(self, check_state: bool) -> None:
        """Handle the checkbox state change for point cloud publishing.

        Args:
            check_state: Whether to enable point cloud publishing.

        """
        self._point_cloud_pub = check_state

    def _on_metadata_changed(self, attr_name: str, check_state: bool) -> None:
        """Handle metadata checkbox state change.

        Args:
            attr_name: Name of the metadata attribute being toggled.
            check_state: Whether the metadata option is enabled.

        """
        self._metadata_selected[attr_name] = check_state


class Ros2RtxRadarGraph(Ros2GraphWindow):
    """A UI helper window for generating ROS2 action graphs for RTX radar sensors.

    This window provides an interface to configure and generate OmniGraph action graphs that publish RTX radar detections to ROS2 as ``sensor_msgs/PointCloud2`` messages. It supports both creating new graphs and adding nodes to existing graphs. Users can optionally include per-point radial velocity, intensity, and timestamp metadata in the published point cloud.

    RTX Radar requires Motion BVH to be enabled in the renderer for Doppler velocity estimation. Radial velocity metadata additionally requires the OmniRadar prim to be authored with auxiliary output level ``"BASIC"``.
    """

    # Point cloud metadata options: (display_name, attribute_name)
    # attribute_name corresponds to the boolean input on ROS2RtxRadarHelper:
    # outputRadialVelocityMS, outputIntensity, outputTimestamp.
    METADATA_OPTIONS = ROS2_RADAR_POINT_CLOUD_METADATA_OPTIONS
    """Point cloud metadata options exposed by the ROS2RtxRadarHelper node.

    Each tuple is ``(display_name, attribute_suffix)``. The helper input name is
    ``inputs:output<attribute_suffix>``.
    """

    def __init__(self) -> None:
        super().__init__("ROS2 RTX Radar Graph", width=400, height=430)
        self._og_path = "/Graph/ROS_RadarRTX"
        self._frame_id = "radar"
        self._node_namespace = ""
        self._add_to_existing_graph = False
        self._radar_prim = ""
        self._render_product_prim = ""
        self._point_cloud_topic = "/radar_point_cloud"

        # Metadata options - dictionary keyed by attribute suffix.
        self._metadata_selected = {attr_name: False for _, attr_name in self.METADATA_OPTIONS}

        # build UI
        self._build_ui()

    def make_graph(self) -> None:
        """Create or modify an action graph for ROS2 RTX Radar publishing.

        Generates a new graph or extends an existing one to publish RTX Radar detections to ROS 2 as PointCloud2 messages. Selected metadata options (radial velocity, intensity, timestamp) are enabled directly on the ROS2RtxRadarHelper node.
        """
        try:
            self._og_path = create_ros2_rtx_radar_graph(
                Ros2RtxRadarGraphConfig(
                    graph_path=self._og_path,
                    radar_prim=self._radar_prim,
                    frame_id=self._frame_id,
                    node_namespace=self._node_namespace,
                    add_to_existing_graph=self._add_to_existing_graph,
                    render_product_prim=self._render_product_prim,
                    point_cloud_topic=self._point_cloud_topic,
                    metadata={attr for attr, selected in self._metadata_selected.items() if selected},
                )
            )
        except (ValueError, RuntimeError) as exc:
            post_notification(str(exc), status=NotificationStatus.WARNING)
            return

        # Radial velocity metadata requires the radar prim to be authored with aux output level "BASIC".
        if self._metadata_selected.get("RadialVelocityMS") and not radar_supports_basic_aux_output(self._radar_prim):
            post_notification(ROS2_RADAR_RADIAL_VELOCITY_AUX_OUTPUT_WARNING, status=NotificationStatus.WARNING)

    def _build_ui(self) -> None:
        """Build the user interface for the RTX Radar graph configuration window.

        Creates input fields for graph path, radar prim selection, frame ID, node namespace, and point cloud topic. Includes checkboxes for the three per-point metadata fields exposed by the ROS2RtxRadarHelper node.
        """
        og_path_def = ParamWidget.FieldDef(
            name="og_path", label="Graph Path", type=ui.StringField, default=self._og_path
        )
        frame_id_def = ParamWidget.FieldDef(
            name="frame_id", label="Frame ID", type=ui.StringField, default=self._frame_id
        )
        node_namespace_def = ParamWidget.FieldDef(
            name="node_namespace", label="Node Namespace", type=ui.StringField, default=self._node_namespace
        )
        point_cloud_topic_def = ParamWidget.FieldDef(
            name="point_cloud_topic", label="Point Cloud Topic", type=ui.StringField, default=self._point_cloud_topic
        )

        with self.frame:
            with ui.VStack(spacing=4):
                with ui.HStack():
                    ui.Label("Add to an existing graph?", width=ui.Percent(30))
                    cb = ui.SimpleBoolModel(default_value=self._add_to_existing_graph)
                    SimpleCheckBox(self._add_to_existing_graph, self._on_use_existing_graph, model=cb)
                self.og_path_input = ParamWidget(field_def=og_path_def)
                self.radar_prim_input = SelectPrimWidget(label="Radar Prim", default=self._radar_prim)
                self.render_product_prim_input = SelectPrimWidget(
                    label="Render Product Prim (Optional)", default=self._render_product_prim
                )
                self.frame_id_input = ParamWidget(field_def=frame_id_def)
                self.node_namespace_input = ParamWidget(field_def=node_namespace_def)
                ui.Spacer(height=5)
                self.point_cloud_topic_input = ParamWidget(field_def=point_cloud_topic_def)

                ui.Spacer(height=5)
                ui.Label("Point Cloud Metadata", word_wrap=True)
                with ui.VStack(spacing=2):
                    for display_name, attr_name in self.METADATA_OPTIONS:
                        with ui.HStack():
                            ui.Label(display_name, width=ui.Percent(30))
                            cb = ui.SimpleBoolModel(default_value=self._metadata_selected[attr_name])
                            SimpleCheckBox(
                                self._metadata_selected[attr_name],
                                lambda checked, attr=attr_name: self._on_metadata_changed(attr, checked),
                                model=cb,
                            )

                add_ok_cancel_buttons(self._on_ok, self._on_cancel)
                add_script_docs_footer(
                    __file__,
                    "https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_rtx_radar.html#graph-shortcut",
                )

        return

    def _on_ok(self) -> None:
        """Handle the OK button click event.

        Collects values from all UI input fields, validates the parameters, and generates the graph if validation passes. Closes the window upon successful graph generation.
        """
        self._og_path = self.og_path_input.get_value()
        self._radar_prim = self.radar_prim_input.get_value()
        self._render_product_prim = self.render_product_prim_input.get_value()
        self._frame_id = self.frame_id_input.get_value()
        self._node_namespace = self.node_namespace_input.get_value()
        self._point_cloud_topic = self.point_cloud_topic_input.get_value()

        self._finish_on_ok()

    def _check_params(self) -> bool:
        """Validate the graph and radar prim parameters.

        Verifies that the specified graph path exists if adding to an existing graph, and confirms that the radar prim is a valid RTX radar (an ``OmniRadar`` prim with the ``OmniSensorGenericRadarWpmDmatAPI`` schema applied).

        Returns:
            True if all parameters are valid, False otherwise.
        """
        if self._add_to_existing_graph and not validate_existing_graph_path(self._og_path):
            return False

        radar_prim = prim_utils.get_prim_at_path(self._radar_prim)
        if (
            radar_prim.IsValid()
            and radar_prim.GetTypeName() == "OmniRadar"
            and radar_prim.HasAPI("OmniSensorGenericRadarWpmDmatAPI")
        ):
            return _check_render_product_prim(self._render_product_prim, self._radar_prim)

        post_notification(
            self._radar_prim + " is not a valid RTX radar prim, check the radar prim",
            status=NotificationStatus.WARNING,
        )
        return False

    def _on_metadata_changed(self, attr_name: str, check_state: bool) -> None:
        """Handle metadata checkbox state change.

        Args:
            attr_name: Attribute suffix of the metadata field being toggled.
            check_state: Whether the metadata option is enabled.
        """
        self._metadata_selected[attr_name] = check_state
