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

"""Public ROS 2 graph creation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.graph.core as og
import OmniGraphSchema
from pxr import Sdf, UsdGeom, UsdRender

GenericPublisherKind = Literal["rtf_float32", "bool", "int64", "string"]
RgbCompression = Literal["rgb", "rgb_h264", "rgb_hevc"]
LidarMetadataOption = Literal[
    "Intensity",
    "Timestamp",
    "EmitterId",
    "ChannelId",
    "MaterialId",
    "TickId",
    "HitNormal",
    "Velocity",
    "ObjectId",
    "EchoId",
    "TickState",
]
RadarMetadataOption = Literal["RadialVelocityMS", "Intensity", "Timestamp"]

RGB_COMPRESSION_OPTIONS: list[tuple[str, RgbCompression, str]] = [
    ("Raw", "rgb", "/rgb"),
    ("H.264", "rgb_h264", "/rgb/compressed"),
    ("HEVC", "rgb_hevc", "/rgb/compressed"),
]
"""RGB camera shortcut options as display name, helper type, and default topic."""

LIDAR_POINT_CLOUD_METADATA_OPTIONS: list[tuple[str, LidarMetadataOption]] = [
    ("Intensity", "Intensity"),
    ("Timestamp", "Timestamp"),
    ("Emitter ID", "EmitterId"),
    ("Channel ID", "ChannelId"),
    ("Material ID", "MaterialId"),
    ("Tick ID", "TickId"),
    ("Hit Normal", "HitNormal"),
    ("Velocity", "Velocity"),
    ("Object ID", "ObjectId"),
    ("Echo ID", "EchoId"),
    ("Tick State", "TickState"),
]
"""RTX lidar point-cloud metadata options as display name and node attribute suffix."""

RADAR_POINT_CLOUD_METADATA_OPTIONS: list[tuple[str, RadarMetadataOption]] = [
    ("Radial Velocity (m/s)", "RadialVelocityMS"),
    ("Intensity", "Intensity"),
    ("Timestamp", "Timestamp"),
]
"""RTX radar point-cloud metadata options as display name and node attribute suffix."""

LIDAR_METADATA_WITHOUT_POINT_CLOUD_WARNING = (
    "Point cloud metadata options are selected but Point Cloud publishing is disabled. Metadata options will be "
    "ignored."
)
"""Warning used when lidar metadata is selected without point cloud publishing."""

RADAR_RADIAL_VELOCITY_AUX_OUTPUT_WARNING = (
    "Radial Velocity selected but the radar prim does not advertise the BASIC auxiliary output channel. Recreate "
    "the radar with aux_output_level='BASIC' (or author "
    "_replicator:rendervar:GenericModelOutput:channels = ['BASIC']) for radial velocity to populate."
)
"""Warning used when radar radial velocity metadata cannot populate."""

_GENERIC_PUBLISHER_KINDS = set(get_args(GenericPublisherKind))
_RGB_TYPES = set(get_args(RgbCompression))
_LIDAR_METADATA_OPTIONS = set(get_args(LidarMetadataOption))
_RADAR_METADATA_OPTIONS = set(get_args(RadarMetadataOption))


@dataclass(slots=True)
class Ros2ClockGraphConfig:
    """Configuration for a ROS 2 clock graph."""

    graph_path: str = "/Graph/ROS_Clock"


@dataclass(slots=True)
class Ros2GenericPublisherGraphConfig:
    """Configuration for a ROS 2 generic publisher graph."""

    graph_path: str = "/Graph/ROS_GenericPub"
    publisher_kind: GenericPublisherKind = "rtf_float32"
    topic_name: str | None = None
    bool_value: bool = True
    int64_value: int = 42
    string_value: str = "Hello from Isaac Sim!"


@dataclass(slots=True)
class Ros2JointStatesGraphConfig:
    """Configuration for a ROS 2 joint states graph."""

    graph_path: str = "/Graph/ROS_JointStates"
    articulation_root: str = ""
    node_namespace: str = ""
    publish_topic: str = "/joint_states"
    subscribe_topic: str = "/joint_command"
    add_to_existing_graph: bool = False
    publish_joint_states: bool = False
    subscribe_joint_states: bool = False
    move_robot_on_subscribe: bool = True


@dataclass(slots=True)
class Ros2TfGraphConfig:
    """Configuration for a ROS 2 TF publisher graph."""

    graph_path: str = "/Graph/ROS_TF"
    target_prim: str = ""
    parent_prim: str = ""
    topic: str = "/tf"
    node_namespace: str = ""
    add_to_existing_graph: bool = False
    append_to_existing_tf_node: bool = False
    existing_tf_node_path: str = ""


@dataclass(slots=True)
class Ros2OdometryGraphConfig:
    """Configuration for a ROS 2 odometry graph."""

    graph_path: str = "/Graph/ROS_Odometry"
    articulation_root: str = ""
    chassis_prim: str = ""
    node_namespace: str = ""
    odometry_topic: str = "/odom"
    tf_topic: str = "/tf"
    add_to_existing_graph: bool = False
    publish_robot_tf: bool = True


@dataclass(slots=True)
class Ros2CameraGraphConfig:
    """Configuration for a ROS 2 camera graph."""

    graph_path: str = "/Graph/ROS_Camera"
    camera_prim: str = "/OmniverseKit_Persp"
    frame_id: str = "sim_camera"
    node_namespace: str = ""
    camera_info_topic: str = "camera_info"
    add_to_existing_graph: bool = False
    render_product_prim: str = ""
    publish_rgb: bool = True
    rgb_topic: str = "/rgb"
    rgb_type: RgbCompression = "rgb"
    publish_depth: bool = True
    depth_topic: str = "/depth"
    publish_depth_point_cloud: bool = False
    depth_point_cloud_topic: str = "/depth_pcl"
    publish_instance_segmentation: bool = False
    instance_segmentation_topic: str = "/instance_segmentation"
    publish_semantic_segmentation: bool = False
    semantic_segmentation_topic: str = "/semantic_segmentation"
    publish_bbox_2d_tight: bool = False
    bbox_2d_tight_topic: str = "/bbox_2d_tight"
    publish_bbox_2d_loose: bool = False
    bbox_2d_loose_topic: str = "/bbox_2d_loose"
    publish_bbox_3d: bool = False
    bbox_3d_topic: str = "/bbox_3d"


@dataclass(slots=True)
class Ros2RtxLidarGraphConfig:
    """Configuration for a ROS 2 RTX lidar graph."""

    graph_path: str = "/Graph/ROS_LidarRTX"
    lidar_prim: str = ""
    frame_id: str = "sim_lidar"
    node_namespace: str = ""
    add_to_existing_graph: bool = False
    render_product_prim: str = ""
    publish_laser_scan: bool = True
    laser_scan_topic: str = "/laser_scan"
    publish_point_cloud: bool = False
    point_cloud_topic: str = "/point_cloud"
    metadata: set[LidarMetadataOption] = field(default_factory=set)


@dataclass(slots=True)
class Ros2RtxRadarGraphConfig:
    """Configuration for a ROS 2 RTX radar graph."""

    graph_path: str = "/Graph/ROS_RadarRTX"
    radar_prim: str = ""
    frame_id: str = "radar"
    node_namespace: str = ""
    add_to_existing_graph: bool = False
    render_product_prim: str = ""
    point_cloud_topic: str = "/radar_point_cloud"
    metadata: set[RadarMetadataOption] = field(default_factory=set)


def _node_path_and_name(graph_path: str, base_name: str) -> tuple[str, str]:
    node_path = stage_utils.generate_next_free_path(f"{graph_path}/{base_name}", prepend_default_prim=False)
    return node_path, Path(node_path).name


def _next_free_node_name(node_path: str) -> str:
    return Path(stage_utils.generate_next_free_path(node_path, prepend_default_prim=False)).name


def _input_values(node_path: str, values: list[tuple[str, object]]) -> list[tuple[str, object]]:
    return [(f"{node_path}.inputs:{name}", value) for name, value in values]


def _edit_graph(target, *, create_nodes=(), set_values=(), connects=()):
    keys = og.Controller.Keys
    edits = {}
    if create_nodes:
        edits[keys.CREATE_NODES] = list(create_nodes)
    if set_values:
        edits[keys.SET_VALUES] = list(set_values)
    if connects:
        edits[keys.CONNECT] = list(connects)
    return og.Controller.edit(target, edits)


def _create_execution_graph(graph_path: str, *, create_nodes=(), set_values=(), connects=()):
    graph, _, _, _ = _edit_graph(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        create_nodes=create_nodes,
        set_values=set_values,
        connects=connects,
    )
    return graph


def _ensure_graph_does_not_exist(graph_path: str) -> None:
    prim = prim_utils.get_prim_at_path(graph_path)
    if prim.IsValid() and prim.IsA(OmniGraphSchema.OmniGraph):
        raise ValueError(f"{graph_path} already exists")


def _get_existing_graph(graph_path: str):
    prim = prim_utils.get_prim_at_path(graph_path)
    if not (prim.IsValid() and prim.IsA(OmniGraphSchema.OmniGraph)):
        raise ValueError(f"{graph_path} is not an existing graph")
    graph = og.get_graph_by_path(graph_path)
    if graph is None:
        raise RuntimeError(f"Could not open OmniGraph at {graph_path}")
    return graph


def _target_value_matches(node, attr_name: str, target_path: str) -> bool:
    attr = node.get_attribute(attr_name)
    if attr is None:
        return False
    value = attr.get()
    return bool(value) and str(value[0]) == target_path


def _first_target(node, attr_name: str) -> str:
    attr = node.get_attribute(attr_name)
    if attr is None:
        return ""
    value = attr.get()
    return str(value[0]) if value else ""


def _discover_common_nodes(graph) -> dict[str, str | None]:
    nodes: dict[str, str | None] = {
        "tick": None,
        "context": None,
        "sim_time": None,
        "run_once": None,
    }
    for node in graph.get_nodes():
        node_path = node.get_prim_path()
        node_type = node.get_type_name()
        if node_type in ("omni.graph.action.OnPlaybackTick", "omni.graph.action.OnTick"):
            nodes["tick"] = node_path
        elif node_type == "isaacsim.ros2.bridge.ROS2Context":
            nodes["context"] = node_path
        elif node_type == "isaacsim.core.nodes.IsaacReadSimulationTime":
            nodes["sim_time"] = node_path
        elif node_type == "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame":
            nodes["run_once"] = node_path
    return nodes


def _require_common_nodes(graph_path: str, nodes: dict[str, str | None], required: tuple[str, ...]) -> None:
    missing = [name for name in required if not nodes.get(name)]
    if missing:
        raise RuntimeError(f"ActionGraph {graph_path} missing required node(s): {', '.join(missing)}")


def _get_or_create_common_graph(
    graph_path: str,
    add_to_existing_graph: bool,
    *,
    include_sim_time: bool = False,
    include_run_once: bool = False,
    required: tuple[str, ...] = ("tick", "context"),
):
    if add_to_existing_graph:
        graph = _get_existing_graph(graph_path)
    else:
        graph_path = stage_utils.generate_next_free_path(graph_path, prepend_default_prim=False)
        create_nodes = [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        ]
        set_values = []
        connects = []
        if include_sim_time:
            create_nodes.append(("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"))
            set_values.append(("ReadSimTime.inputs:resetOnStop", True))
        if include_run_once:
            create_nodes.append(("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"))
            connects.append(("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"))
        graph = _create_execution_graph(graph_path, create_nodes=create_nodes, set_values=set_values, connects=connects)

    common = _discover_common_nodes(graph)
    _require_common_nodes(graph_path, common, required)
    return graph_path, graph, common


def _validate_options(values: set[str], valid_values: set[str], field_name: str) -> None:
    invalid_values = sorted(values - valid_values)
    if invalid_values:
        raise ValueError(f"Unsupported {field_name}: {', '.join(invalid_values)}")


def _validate_render_product_prim(render_product_prim: str, expected_camera_prim: str = "") -> None:
    prim = prim_utils.get_prim_at_path(render_product_prim)
    if not prim.IsValid() or not prim.IsA(UsdRender.Product):
        raise ValueError(f"{render_product_prim} is not a valid render product prim")
    if expected_camera_prim:
        render_product = UsdRender.Product(prim)
        camera_targets = render_product.GetCameraRel().GetTargets()
        if not camera_targets or str(camera_targets[0]) != expected_camera_prim:
            raise ValueError(
                f"{render_product_prim} does not target the configured camera or sensor prim {expected_camera_prim}"
            )


def _validate_camera_prim(camera_prim: str) -> None:
    prim = prim_utils.get_prim_at_path(camera_prim)
    if not prim.IsValid() or not UsdGeom.Camera(prim):
        raise ValueError(f"{camera_prim} is not a valid camera prim")


def _validate_lidar_prim(lidar_prim: str) -> None:
    prim = prim_utils.get_prim_at_path(lidar_prim)
    if not (prim.IsValid() and prim.GetTypeName() == "OmniLidar" and prim.HasAPI("OmniSensorGenericLidarCoreAPI")):
        raise ValueError(f"{lidar_prim} is not a valid RTX lidar prim")


def _validate_radar_prim(radar_prim: str) -> None:
    prim = prim_utils.get_prim_at_path(radar_prim)
    if not (prim.IsValid() and prim.GetTypeName() == "OmniRadar" and prim.HasAPI("OmniSensorGenericRadarWpmDmatAPI")):
        raise ValueError(f"{radar_prim} is not a valid RTX radar prim")


def radar_supports_basic_aux_output(radar_prim: str) -> bool:
    """Return whether an RTX radar prim advertises the BASIC auxiliary output channel.

    Args:
        radar_prim: Prim path of the RTX radar to inspect.

    Returns:
        True if the radar prim advertises the BASIC auxiliary output channel, False otherwise.
    """
    prim = prim_utils.get_prim_at_path(radar_prim)
    if not prim.IsValid():
        return False
    channels_attr = prim.GetAttribute("_replicator:rendervar:GenericModelOutput:channels")
    return channels_attr.IsValid() and "BASIC" in (channels_attr.Get() or [])


def _connect_attrs(source_attr: str, target_attr: str) -> None:
    og.Controller.connect(og.Controller.attribute(source_attr), og.Controller.attribute(target_attr))


def _render_source_connects(render_source_node: str, target_node: str) -> list[tuple[str, str]]:
    return [
        (render_source_node + ".outputs:execOut", target_node + ".inputs:execIn"),
        (render_source_node + ".outputs:renderProductPath", target_node + ".inputs:renderProductPath"),
    ]


def _create_render_source(
    graph,
    graph_path: str,
    sensor_prim: str,
    exec_source_attr: str,
    render_product_prim: str = "",
) -> str:
    if render_product_prim:
        _validate_render_product_prim(render_product_prim, sensor_prim)
        for node in graph.get_nodes():
            if (
                node.get_type_name() == "isaacsim.core.nodes.IsaacAttachHydraTexture"
                and _first_target(node, "inputs:renderProductPrim") == render_product_prim
            ):
                return node.get_prim_path()

        attach_node, attach_node_name = _node_path_and_name(graph_path, "AttachHydraTexture")
        _edit_graph(
            graph,
            create_nodes=[(attach_node_name, "isaacsim.core.nodes.IsaacAttachHydraTexture")],
            set_values=[(attach_node + ".inputs:renderProductPrim", render_product_prim)],
            connects=[(exec_source_attr, attach_node + ".inputs:execIn")],
        )
        return attach_node

    for node in graph.get_nodes():
        if node.get_type_name() == "isaacsim.core.nodes.IsaacCreateRenderProduct" and _target_value_matches(
            node, "inputs:cameraPrim", sensor_prim
        ):
            return node.get_prim_path()

    render_node_path, render_node_name = _node_path_and_name(graph_path, "RenderProduct")
    _edit_graph(
        graph,
        create_nodes=[(render_node_name, "isaacsim.core.nodes.IsaacCreateRenderProduct")],
        set_values=[(render_node_path + ".inputs:cameraPrim", sensor_prim)],
        connects=[(exec_source_attr, render_node_path + ".inputs:execIn")],
    )
    return render_node_path


def _get_or_create_sensor_graph(
    graph_path: str,
    add_to_existing_graph: bool,
    sensor_prim: str,
    render_product_prim: str,
    *,
    use_run_once: bool,
):
    render_source_uses_run_once = use_run_once or (not add_to_existing_graph and not render_product_prim)
    required = ("tick", "context", "run_once") if render_source_uses_run_once else ("tick", "context")
    graph_path, graph, common = _get_or_create_common_graph(
        graph_path,
        add_to_existing_graph,
        include_run_once=render_source_uses_run_once,
        required=required,
    )
    exec_source_attr = (
        common["run_once"] + ".outputs:step" if render_source_uses_run_once else common["tick"] + ".outputs:tick"
    )
    render_source_node = _create_render_source(graph, graph_path, sensor_prim, exec_source_attr, render_product_prim)
    return graph_path, graph, common, render_source_node


def _connect_context(context_node: str | None, target_node: str) -> None:
    if context_node:
        _connect_attrs(context_node + ".outputs:context", target_node + ".inputs:context")


def _connect_context_and_time(context_node: str, sim_time_node: str, target_node: str) -> None:
    _connect_attrs(context_node + ".outputs:context", target_node + ".inputs:context")
    _connect_attrs(sim_time_node + ".outputs:simulationTime", target_node + ".inputs:timeStamp")


def _create_render_product_helper(
    graph,
    graph_path: str,
    render_source_node: str,
    context_node: str | None,
    node_base_name: str,
    helper_node_type: str,
    input_values: list[tuple[str, object]],
    extra_connects=(),
) -> str:
    node_path, node_name = _node_path_and_name(graph_path, node_base_name)
    _edit_graph(
        graph,
        create_nodes=[(node_name, helper_node_type)],
        set_values=_input_values(node_path, input_values),
        connects=_render_source_connects(render_source_node, node_path) + list(extra_connects),
    )
    _connect_context(context_node, node_path)
    return node_path


def create_ros2_clock_graph(config: Ros2ClockGraphConfig) -> str:
    """Create a ROS 2 clock graph and return the graph path.

    Args:
        config: Configuration for the ROS 2 clock graph.

    Returns:
        Path to the created graph.
    """
    app_utils.stop(commit=False)
    _ensure_graph_does_not_exist(config.graph_path)
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": config.graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [("ReadSimTime.inputs:resetOnStop", True)],
        },
    )
    return config.graph_path


def create_ros2_generic_publisher_graph(config: Ros2GenericPublisherGraphConfig) -> str:
    """Create a ROS 2 generic publisher graph and return the graph path.

    Args:
        config: Configuration for the ROS 2 generic publisher graph.

    Returns:
        Path to the created graph.
    """
    app_utils.stop(commit=False)
    if config.publisher_kind not in _GENERIC_PUBLISHER_KINDS:
        raise ValueError(f"Unsupported generic publisher kind: {config.publisher_kind}")
    _ensure_graph_does_not_exist(config.graph_path)
    keys = og.Controller.Keys
    create_nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("GenericPublisher", "isaacsim.ros2.bridge.ROS2Publisher"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
    ]
    set_values = [
        ("GenericPublisher.inputs:messagePackage", "std_msgs"),
        ("GenericPublisher.inputs:messageSubfolder", "msg"),
    ]
    connects = [
        ("OnPlaybackTick.outputs:tick", "GenericPublisher.inputs:execIn"),
        ("Context.outputs:context", "GenericPublisher.inputs:context"),
    ]
    data_source = ""
    if config.publisher_kind == "rtf_float32":
        create_nodes.append(("RTF", "isaacsim.core.nodes.IsaacRealTimeFactor"))
        set_values.append(("GenericPublisher.inputs:messageName", "Float32"))
        data_source = config.graph_path + "/RTF.outputs:rtf"
    elif config.publisher_kind == "bool":
        create_nodes.append(("ConstBool", "omni.graph.nodes.ConstantBool"))
        set_values.extend(
            [("GenericPublisher.inputs:messageName", "Bool"), ("ConstBool.inputs:value", config.bool_value)]
        )
        data_source = config.graph_path + "/ConstBool.inputs:value"
    elif config.publisher_kind == "int64":
        create_nodes.append(("ConstInt", "omni.graph.nodes.ConstantInt64"))
        set_values.extend(
            [("GenericPublisher.inputs:messageName", "Int64"), ("ConstInt.inputs:value", config.int64_value)]
        )
        data_source = config.graph_path + "/ConstInt.inputs:value"
    elif config.publisher_kind == "string":
        create_nodes.append(("ConstToken", "omni.graph.nodes.ConstantToken"))
        set_values.extend(
            [("GenericPublisher.inputs:messageName", "String"), ("ConstToken.inputs:value", config.string_value)]
        )
        data_source = config.graph_path + "/ConstToken.inputs:value"
    if config.topic_name is not None:
        set_values.append(("GenericPublisher.inputs:topicName", config.topic_name))

    og.Controller.edit(
        {"graph_path": config.graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: create_nodes,
            keys.SET_VALUES: set_values,
            keys.CONNECT: connects,
        },
    )
    og.Controller.connect(
        og.Controller.attribute(data_source),
        og.Controller.attribute(config.graph_path + "/GenericPublisher.inputs:data"),
    )
    return config.graph_path


def create_ros2_joint_states_graph(config: Ros2JointStatesGraphConfig) -> str:
    """Create or extend a ROS 2 joint states graph and return the graph path.

    Args:
        config: Configuration for the ROS 2 joint states graph.

    Returns:
        Path to the created or extended graph.
    """
    app_utils.stop(commit=False)
    keys = og.Controller.Keys
    graph_path, graph, common = _get_or_create_common_graph(
        config.graph_path,
        config.add_to_existing_graph,
        include_sim_time=True,
        required=("tick", "context", "sim_time"),
    )
    tick_node = common["tick"]
    context_node = common["context"]
    sim_time_node = common["sim_time"]

    js_pub_name = "PublisherJointState"
    js_read_name = "ReadJointState"
    js_sub_name = "SubscriberJointState"
    art_node_name = "ArticulationController"
    for node in graph.get_nodes():
        node_path = node.get_prim_path()
        node_type = node.get_type_name()
        if node_type == "isaacsim.ros2.bridge.ROS2PublishJointState":
            js_pub_name = _next_free_node_name(node_path)
        elif node_type == "isaacsim.sensors.physics.IsaacReadJointState":
            js_read_name = _next_free_node_name(node_path)
        elif node_type == "isaacsim.ros2.bridge.ROS2SubscribeJointState":
            js_sub_name = _next_free_node_name(node_path)
        elif node_type == "isaacsim.core.nodes.IsaacArticulationController":
            art_node_name = _next_free_node_name(node_path)

    if config.publish_joint_states:
        read_path = graph_path + "/" + js_read_name
        pub_path = graph_path + "/" + js_pub_name
        og.Controller.edit(
            graph,
            {
                keys.CREATE_NODES: [
                    (js_read_name, "isaacsim.sensors.physics.IsaacReadJointState"),
                    (js_pub_name, "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ],
                keys.SET_VALUES: [
                    (
                        js_read_name + ".inputs:prim",
                        [Sdf.Path(config.articulation_root)] if config.articulation_root else [],
                    ),
                    (js_pub_name + ".inputs:topicName", config.publish_topic),
                    (js_pub_name + ".inputs:nodeNamespace", config.node_namespace),
                ],
                keys.CONNECT: [
                    (tick_node + ".outputs:tick", read_path + ".inputs:execIn"),
                    (read_path + ".outputs:execOut", pub_path + ".inputs:execIn"),
                    (read_path + ".outputs:jointNames", pub_path + ".inputs:jointNames"),
                    (read_path + ".outputs:jointPositions", pub_path + ".inputs:jointPositions"),
                    (read_path + ".outputs:jointVelocities", pub_path + ".inputs:jointVelocities"),
                    (read_path + ".outputs:jointEfforts", pub_path + ".inputs:jointEfforts"),
                    (read_path + ".outputs:jointDofTypes", pub_path + ".inputs:jointDofTypes"),
                    (read_path + ".outputs:stageMetersPerUnit", pub_path + ".inputs:stageMetersPerUnit"),
                    (read_path + ".outputs:sensorTime", pub_path + ".inputs:sensorTime"),
                ],
            },
        )
        _connect_context_and_time(context_node, sim_time_node, pub_path)

    if config.subscribe_joint_states:
        sub_path = graph_path + "/" + js_sub_name
        og.Controller.edit(
            graph,
            {
                keys.CREATE_NODES: [(js_sub_name, "isaacsim.ros2.bridge.ROS2SubscribeJointState")],
                keys.SET_VALUES: [
                    (js_sub_name + ".inputs:topicName", config.subscribe_topic),
                    (js_sub_name + ".inputs:nodeNamespace", config.node_namespace),
                ],
            },
        )
        _connect_attrs(tick_node + ".outputs:tick", sub_path + ".inputs:execIn")
        _connect_context(context_node, sub_path)
        if config.move_robot_on_subscribe:
            art_path = graph_path + "/" + art_node_name
            og.Controller.edit(
                graph,
                {
                    keys.CREATE_NODES: [(art_node_name, "isaacsim.core.nodes.IsaacArticulationController")],
                    keys.SET_VALUES: [(art_node_name + ".inputs:targetPrim", config.articulation_root)],
                    keys.CONNECT: [
                        (tick_node + ".outputs:tick", art_path + ".inputs:execIn"),
                        (sub_path + ".outputs:positionCommand", art_path + ".inputs:positionCommand"),
                        (sub_path + ".outputs:velocityCommand", art_path + ".inputs:velocityCommand"),
                        (sub_path + ".outputs:effortCommand", art_path + ".inputs:effortCommand"),
                        (sub_path + ".outputs:jointNames", art_path + ".inputs:jointNames"),
                    ],
                },
            )

    return graph_path


def create_ros2_tf_graph(config: Ros2TfGraphConfig) -> str:
    """Create or extend a ROS 2 TF graph and return the graph path.

    Args:
        config: Configuration for the ROS 2 TF publisher graph.

    Returns:
        Path to the created or extended graph.
    """
    app_utils.stop(commit=False)
    if not config.target_prim:
        raise ValueError("Target prim is required")
    if config.append_to_existing_tf_node and not config.add_to_existing_graph:
        raise ValueError("Appending to an existing TF node requires add_to_existing_graph")

    keys = og.Controller.Keys
    graph_path, graph, common = _get_or_create_common_graph(
        config.graph_path,
        config.add_to_existing_graph,
        include_sim_time=True,
        required=("tick", "context", "sim_time"),
    )
    tick_node = common["tick"]
    context_node = common["context"]
    sim_time_node = common["sim_time"]

    tf_pub_name = "PublisherTF"
    tf_pub_node = config.existing_tf_node_path
    has_existing_tf = False
    selected_existing_tf_node = False
    for node in graph.get_nodes():
        node_path = node.get_prim_path()
        if node.get_type_name() == "isaacsim.ros2.bridge.ROS2PublishTransformTree":
            has_existing_tf = True
            if node_path == config.existing_tf_node_path:
                selected_existing_tf_node = True
            if not config.append_to_existing_tf_node:
                tf_pub_name = _next_free_node_name(node_path)

    if config.append_to_existing_tf_node:
        if not (has_existing_tf and selected_existing_tf_node):
            raise ValueError(f"{config.existing_tf_node_path} is not an existing TF publisher node")
        existing_targets = og.Controller.attribute(tf_pub_node + ".inputs:targetPrims").get()
        existing_targets.append(Sdf.Path(config.target_prim))
        og.Controller.edit(graph, {keys.SET_VALUES: [(tf_pub_node + ".inputs:targetPrims", existing_targets)]})
        return graph_path

    compute_tf_name = "Compute" + tf_pub_name
    og.Controller.edit(
        graph,
        {
            keys.CREATE_NODES: [
                (compute_tf_name, "isaacsim.core.nodes.IsaacComputeTransformTree"),
                (tf_pub_name, "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            keys.SET_VALUES: [
                (compute_tf_name + ".inputs:parentPrim", config.parent_prim),
                (compute_tf_name + ".inputs:targetPrims", config.target_prim),
                (tf_pub_name + ".inputs:topicName", config.topic),
                (tf_pub_name + ".inputs:nodeNamespace", config.node_namespace),
            ],
            keys.CONNECT: [
                (tick_node + ".outputs:tick", compute_tf_name + ".inputs:execIn"),
                (compute_tf_name + ".outputs:execOut", tf_pub_name + ".inputs:execIn"),
                (compute_tf_name + ".outputs:parentFrames", tf_pub_name + ".inputs:parentFrames"),
                (compute_tf_name + ".outputs:childFrames", tf_pub_name + ".inputs:childFrames"),
                (compute_tf_name + ".outputs:translations", tf_pub_name + ".inputs:translations"),
                (compute_tf_name + ".outputs:orientations", tf_pub_name + ".inputs:orientations"),
                (sim_time_node + ".outputs:simulationTime", tf_pub_name + ".inputs:timeStamp"),
                (context_node + ".outputs:context", tf_pub_name + ".inputs:context"),
            ],
        },
    )
    return graph_path


def create_ros2_odometry_graph(config: Ros2OdometryGraphConfig) -> str:
    """Create or extend a ROS 2 odometry graph and return the graph path.

    Args:
        config: Configuration for the ROS 2 odometry graph.

    Returns:
        Path to the created or extended graph.
    """
    app_utils.stop(commit=False)
    if not config.articulation_root:
        raise ValueError("Robot Articulation Root prim is required")

    keys = og.Controller.Keys
    graph_path, graph, common = _get_or_create_common_graph(
        config.graph_path,
        config.add_to_existing_graph,
        include_sim_time=True,
        required=("tick", "context", "sim_time"),
    )
    tick_node = common["tick"]
    context_node = common["context"]
    sim_time_node = common["sim_time"]
    chassis_link_name = config.chassis_prim.split("/")[-1]

    odom_compute_name = "ComputeOdometry"
    odom_pub_name = "PublisherOdometry"
    tf_odom2robot_name = "TFOdom2Robot"
    tf_world2odom_name = "TFWorld2Odom"
    tf_robot_name = "TFRobot"
    for node in graph.get_nodes():
        node_path = node.get_prim_path()
        node_type = node.get_type_name()
        if node_type == "isaacsim.ros2.bridge.ROS2PublishOdometry":
            odom_pub_name = _next_free_node_name(node_path)
        elif node_type == "isaacsim.core.nodes.IsaacComputeOdometry":
            odom_compute_name = _next_free_node_name(node_path)
        elif node_type == "isaacsim.ros2.bridge.ROS2PublishRawTransformTree":
            tf_world2odom_name = _next_free_node_name(graph_path + "/" + tf_world2odom_name)
            tf_odom2robot_name = _next_free_node_name(graph_path + "/" + tf_odom2robot_name)
        elif node_type == "isaacsim.ros2.bridge.ROS2PublishTransformTree":
            tf_robot_name = _next_free_node_name(node_path)

    og.Controller.edit(
        graph,
        {
            keys.CREATE_NODES: [
                (tf_world2odom_name, "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                (tf_odom2robot_name, "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                (odom_compute_name, "isaacsim.core.nodes.IsaacComputeOdometry"),
                (odom_pub_name, "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ],
            keys.SET_VALUES: [
                (odom_compute_name + ".inputs:chassisPrim", config.articulation_root),
                (odom_pub_name + ".inputs:topicName", config.odometry_topic),
                (odom_pub_name + ".inputs:chassisFrameId", chassis_link_name),
                (odom_pub_name + ".inputs:nodeNamespace", config.node_namespace),
                (tf_odom2robot_name + ".inputs:childFrameId", chassis_link_name),
                (tf_world2odom_name + ".inputs:childFrameId", "odom"),
                (tf_world2odom_name + ".inputs:parentFrameId", "world"),
                (tf_odom2robot_name + ".inputs:nodeNamespace", config.node_namespace),
                (tf_world2odom_name + ".inputs:nodeNamespace", config.node_namespace),
            ],
            keys.CONNECT: [
                (tick_node + ".outputs:tick", tf_world2odom_name + ".inputs:execIn"),
                (tick_node + ".outputs:tick", tf_odom2robot_name + ".inputs:execIn"),
                (tick_node + ".outputs:tick", odom_compute_name + ".inputs:execIn"),
                (odom_compute_name + ".outputs:execOut", odom_pub_name + ".inputs:execIn"),
                (odom_compute_name + ".outputs:angularVelocity", odom_pub_name + ".inputs:angularVelocity"),
                (odom_compute_name + ".outputs:linearVelocity", odom_pub_name + ".inputs:linearVelocity"),
                (odom_compute_name + ".outputs:orientation", odom_pub_name + ".inputs:orientation"),
                (odom_compute_name + ".outputs:position", odom_pub_name + ".inputs:position"),
                (odom_compute_name + ".outputs:orientation", tf_odom2robot_name + ".inputs:rotation"),
                (odom_compute_name + ".outputs:position", tf_odom2robot_name + ".inputs:translation"),
            ],
        },
    )
    for node_name in (tf_world2odom_name, tf_odom2robot_name, odom_pub_name):
        _connect_context_and_time(context_node, sim_time_node, graph_path + "/" + node_name)

    if config.publish_robot_tf:
        compute_tf_robot_name = "Compute" + tf_robot_name
        og.Controller.edit(
            graph,
            {
                keys.CREATE_NODES: [
                    (compute_tf_robot_name, "isaacsim.core.nodes.IsaacComputeTransformTree"),
                    (tf_robot_name, "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ],
                keys.SET_VALUES: [
                    (compute_tf_robot_name + ".inputs:parentPrim", config.chassis_prim),
                    (compute_tf_robot_name + ".inputs:targetPrims", config.articulation_root),
                    (tf_robot_name + ".inputs:topicName", config.tf_topic),
                    (tf_robot_name + ".inputs:nodeNamespace", config.node_namespace),
                ],
                keys.CONNECT: [
                    (tick_node + ".outputs:tick", compute_tf_robot_name + ".inputs:execIn"),
                    (compute_tf_robot_name + ".outputs:execOut", tf_robot_name + ".inputs:execIn"),
                    (compute_tf_robot_name + ".outputs:parentFrames", tf_robot_name + ".inputs:parentFrames"),
                    (compute_tf_robot_name + ".outputs:childFrames", tf_robot_name + ".inputs:childFrames"),
                    (compute_tf_robot_name + ".outputs:translations", tf_robot_name + ".inputs:translations"),
                    (compute_tf_robot_name + ".outputs:orientations", tf_robot_name + ".inputs:orientations"),
                    (sim_time_node + ".outputs:simulationTime", tf_robot_name + ".inputs:timeStamp"),
                    (context_node + ".outputs:context", tf_robot_name + ".inputs:context"),
                ],
            },
        )
    return graph_path


def _create_camera_helper(
    graph,
    graph_path: str,
    render_source_node: str,
    context_node: str | None,
    node_base_name: str,
    topic_name: str,
    helper_type: str,
    frame_id: str,
    node_namespace: str,
    enable_semantic_labels: bool = False,
) -> None:
    input_values = [
        ("topicName", topic_name),
        ("type", helper_type),
        ("resetSimulationTimeOnStop", True),
        ("frameId", frame_id),
        ("nodeNamespace", node_namespace),
    ]
    if enable_semantic_labels:
        input_values.append(("enableSemanticLabels", True))
    _create_render_product_helper(
        graph,
        graph_path,
        render_source_node,
        context_node,
        node_base_name,
        "isaacsim.ros2.bridge.ROS2CameraHelper",
        input_values,
    )


def create_ros2_camera_graph(config: Ros2CameraGraphConfig) -> str:
    """Create or extend a ROS 2 camera graph and return the graph path.

    Args:
        config: Configuration for the ROS 2 camera graph.

    Returns:
        Path to the created or extended graph.
    """
    app_utils.stop(commit=False)
    if config.rgb_type not in _RGB_TYPES:
        raise ValueError(f"Unsupported RGB compression type: {config.rgb_type}")
    _validate_camera_prim(config.camera_prim)
    graph_path, graph, common, render_source_node = _get_or_create_sensor_graph(
        config.graph_path,
        config.add_to_existing_graph,
        config.camera_prim,
        config.render_product_prim,
        use_run_once=False,
    )
    context_node = common["context"]

    if not config.add_to_existing_graph:
        _create_render_product_helper(
            graph,
            graph_path,
            render_source_node,
            context_node,
            "CameraInfoPublish",
            "isaacsim.ros2.bridge.ROS2CameraInfoHelper",
            [
                ("topicName", config.camera_info_topic),
                ("frameId", config.frame_id),
                ("nodeNamespace", config.node_namespace),
                ("resetSimulationTimeOnStop", True),
            ],
        )

    camera_helper_configs = [
        (config.publish_rgb, "RGBPublish", config.rgb_topic, config.rgb_type, False),
        (config.publish_depth, "DepthPublish", config.depth_topic, "depth", False),
        (config.publish_depth_point_cloud, "DepthPclPublish", config.depth_point_cloud_topic, "depth_pcl", False),
        (
            config.publish_instance_segmentation,
            "InstancePublish",
            config.instance_segmentation_topic,
            "instance_segmentation",
            True,
        ),
        (
            config.publish_semantic_segmentation,
            "SemanticPublish",
            config.semantic_segmentation_topic,
            "semantic_segmentation",
            True,
        ),
        (config.publish_bbox_2d_tight, "Bbox2dTightPublish", config.bbox_2d_tight_topic, "bbox_2d_tight", True),
        (config.publish_bbox_2d_loose, "Bbox2dLoosePublish", config.bbox_2d_loose_topic, "bbox_2d_loose", True),
        (config.publish_bbox_3d, "Bbox3dPublish", config.bbox_3d_topic, "bbox_3d", True),
    ]
    for enabled, node_name, topic, helper_type, enable_semantic_labels in camera_helper_configs:
        if not enabled:
            continue
        _create_camera_helper(
            graph,
            graph_path,
            render_source_node,
            context_node,
            node_name,
            topic,
            helper_type,
            config.frame_id,
            config.node_namespace,
            enable_semantic_labels,
        )
    return graph_path


def create_ros2_rtx_lidar_graph(config: Ros2RtxLidarGraphConfig) -> str:
    """Create or extend a ROS 2 RTX lidar graph and return the graph path.

    Args:
        config: Configuration for the ROS 2 RTX lidar graph.

    Returns:
        Path to the created or extended graph.
    """
    app_utils.stop(commit=False)
    _validate_options(set(config.metadata), _LIDAR_METADATA_OPTIONS, "lidar metadata option(s)")
    _validate_lidar_prim(config.lidar_prim)
    graph_path, graph, common, render_source_node = _get_or_create_sensor_graph(
        config.graph_path,
        config.add_to_existing_graph,
        config.lidar_prim,
        config.render_product_prim,
        use_run_once=True,
    )
    context_node = common["context"]
    if config.publish_laser_scan:
        _create_render_product_helper(
            graph,
            graph_path,
            render_source_node,
            context_node,
            "LaserScanPublish",
            "isaacsim.ros2.bridge.ROS2RtxLidarHelper",
            [
                ("topicName", config.laser_scan_topic),
                ("type", "laser_scan"),
                ("frameId", config.frame_id),
                ("nodeNamespace", config.node_namespace),
            ],
        )

    if config.publish_point_cloud:
        point_cloud_node, point_cloud_name = _node_path_and_name(graph_path, "PointCloudPublish")
        selected_metadata = sorted(config.metadata)
        create_nodes = [(point_cloud_name, "isaacsim.ros2.bridge.ROS2RtxLidarHelper")]
        set_values = _input_values(
            point_cloud_node,
            [
                ("topicName", config.point_cloud_topic),
                ("type", "point_cloud"),
                ("frameId", config.frame_id),
                ("nodeNamespace", config.node_namespace),
            ],
        )
        connects = _render_source_connects(render_source_node, point_cloud_node)
        if selected_metadata:
            pcl_config_node, pcl_config_name = _node_path_and_name(graph_path, "PointCloudConfig")
            create_nodes.insert(0, (pcl_config_name, "isaacsim.ros2.bridge.ROS2RtxLidarPointCloudConfig"))
            set_values = [(f"{pcl_config_node}.inputs:output{attr}", True) for attr in selected_metadata] + set_values
            connects.append(
                (pcl_config_node + ".outputs:selectedMetadata", point_cloud_node + ".inputs:selectedMetadata")
            )
            if "ObjectId" in selected_metadata:
                set_values.append((point_cloud_node + ".inputs:enableObjectIdMap", True))
        _edit_graph(graph, create_nodes=create_nodes, set_values=set_values, connects=connects)
        _connect_context(context_node, point_cloud_node)
    elif config.metadata:
        carb.log_warn(LIDAR_METADATA_WITHOUT_POINT_CLOUD_WARNING)

    return graph_path


def create_ros2_rtx_radar_graph(config: Ros2RtxRadarGraphConfig) -> str:
    """Create or extend a ROS 2 RTX radar graph and return the graph path.

    Args:
        config: Configuration for the ROS 2 RTX radar graph.

    Returns:
        Path to the created or extended graph.
    """
    app_utils.stop(commit=False)
    _validate_options(set(config.metadata), _RADAR_METADATA_OPTIONS, "radar metadata option(s)")
    _validate_radar_prim(config.radar_prim)
    graph_path, graph, common, render_source_node = _get_or_create_sensor_graph(
        config.graph_path,
        config.add_to_existing_graph,
        config.radar_prim,
        config.render_product_prim,
        use_run_once=True,
    )
    context_node = common["context"]
    helper_inputs = [
        ("topicName", config.point_cloud_topic),
        ("frameId", config.frame_id),
        ("nodeNamespace", config.node_namespace),
    ]
    for attr_suffix in sorted(config.metadata):
        helper_inputs.append((f"output{attr_suffix}", True))
    _create_render_product_helper(
        graph,
        graph_path,
        render_source_node,
        context_node,
        "RadarHelper",
        "isaacsim.ros2.bridge.ROS2RtxRadarHelper",
        helper_inputs,
    )

    if "RadialVelocityMS" in config.metadata and not radar_supports_basic_aux_output(config.radar_prim):
        carb.log_warn(RADAR_RADIAL_VELOCITY_AUX_OUTPUT_WARNING)
    return graph_path


def set_isaac_namespace(prim_path: str, namespace: str) -> None:
    """Author the Isaac namespace attribute on a prim.

    Args:
        prim_path: Prim path on which to author the namespace attribute.
        namespace: Namespace value to author.

    Raises:
        ValueError: If the namespace is empty, starts with a digit, or the prim does not exist.
    """
    if not namespace:
        raise ValueError("namespace must not be empty")
    if namespace[0].isdigit():
        raise ValueError("namespace must not start with a digit")
    from usd.schema.isaac import robot_schema

    prim = prim_utils.get_prim_at_path(prim_path)
    if not prim.IsValid():
        raise ValueError(f"{prim_path} does not exist")
    prim.CreateAttribute(robot_schema.Attributes.NAMESPACE.name, robot_schema.Attributes.NAMESPACE.type, True).Set(
        namespace
    )


def set_isaac_name_override(prim_path: str, name: str) -> None:
    """Author the Isaac name override attribute on a prim.

    Args:
        prim_path: Prim path on which to author the name override attribute.
        name: Name override value to author.

    Raises:
        ValueError: If the name is empty or the prim does not exist.
    """
    if not name:
        raise ValueError("name must not be empty")
    from usd.schema.isaac import robot_schema

    prim = prim_utils.get_prim_at_path(prim_path)
    if not prim.IsValid():
        raise ValueError(f"{prim_path} does not exist")
    prim.CreateAttribute(
        robot_schema.Attributes.NAME_OVERRIDE.name, robot_schema.Attributes.NAME_OVERRIDE.type, True
    ).Set(name)
