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

"""ROS 2 OmniGraph nodes for Isaac Sim."""

from isaacsim.core.nodes import register_node_writer_with_telemetry as register_node_writer_with_telemetry
from isaacsim.core.rendering_manager import ViewportManager as ViewportManager
from isaacsim.ros2.core import collect_namespace as collect_namespace
from isaacsim.ros2.core import compute_relative_pose as compute_relative_pose
from isaacsim.ros2.core import read_camera_info as read_camera_info

from .bindings._ros2_nodes import acquire_interface as acquire_interface
from .bindings._ros2_nodes import release_interface as release_interface
from .impl.extension import ROS2NodesExtension as ROS2NodesExtension
from .impl.graph_api import LIDAR_METADATA_WITHOUT_POINT_CLOUD_WARNING as LIDAR_METADATA_WITHOUT_POINT_CLOUD_WARNING
from .impl.graph_api import LIDAR_POINT_CLOUD_METADATA_OPTIONS as LIDAR_POINT_CLOUD_METADATA_OPTIONS
from .impl.graph_api import RADAR_POINT_CLOUD_METADATA_OPTIONS as RADAR_POINT_CLOUD_METADATA_OPTIONS
from .impl.graph_api import RADAR_RADIAL_VELOCITY_AUX_OUTPUT_WARNING as RADAR_RADIAL_VELOCITY_AUX_OUTPUT_WARNING
from .impl.graph_api import RGB_COMPRESSION_OPTIONS as RGB_COMPRESSION_OPTIONS
from .impl.graph_api import GenericPublisherKind as GenericPublisherKind
from .impl.graph_api import LidarMetadataOption as LidarMetadataOption
from .impl.graph_api import RadarMetadataOption as RadarMetadataOption
from .impl.graph_api import RgbCompression as RgbCompression
from .impl.graph_api import Ros2CameraGraphConfig as Ros2CameraGraphConfig
from .impl.graph_api import Ros2ClockGraphConfig as Ros2ClockGraphConfig
from .impl.graph_api import Ros2GenericPublisherGraphConfig as Ros2GenericPublisherGraphConfig
from .impl.graph_api import Ros2JointStatesGraphConfig as Ros2JointStatesGraphConfig
from .impl.graph_api import Ros2OdometryGraphConfig as Ros2OdometryGraphConfig
from .impl.graph_api import Ros2RtxLidarGraphConfig as Ros2RtxLidarGraphConfig
from .impl.graph_api import Ros2RtxRadarGraphConfig as Ros2RtxRadarGraphConfig
from .impl.graph_api import Ros2TfGraphConfig as Ros2TfGraphConfig
from .impl.graph_api import create_ros2_camera_graph as create_ros2_camera_graph
from .impl.graph_api import create_ros2_clock_graph as create_ros2_clock_graph
from .impl.graph_api import create_ros2_generic_publisher_graph as create_ros2_generic_publisher_graph
from .impl.graph_api import create_ros2_joint_states_graph as create_ros2_joint_states_graph
from .impl.graph_api import create_ros2_odometry_graph as create_ros2_odometry_graph
from .impl.graph_api import create_ros2_rtx_lidar_graph as create_ros2_rtx_lidar_graph
from .impl.graph_api import create_ros2_rtx_radar_graph as create_ros2_rtx_radar_graph
from .impl.graph_api import create_ros2_tf_graph as create_ros2_tf_graph
from .impl.graph_api import radar_supports_basic_aux_output as radar_supports_basic_aux_output
from .impl.graph_api import set_isaac_name_override as set_isaac_name_override
from .impl.graph_api import set_isaac_namespace as set_isaac_namespace
from .impl.point_cloud_utils import PointFieldDescription as PointFieldDescription
from .impl.point_cloud_utils import fill_point_cloud2_message as fill_point_cloud2_message
from .impl.point_cloud_utils import interleave_point_cloud as interleave_point_cloud
from .impl.ros2_common import CompressedImageManager as CompressedImageManager
from .impl.ros2_common import SrtxCaptureState as SrtxCaptureState
from .impl.ros2_common import SrtxSensorSetConfig as SrtxSensorSetConfig
from .impl.ros2_common import cleanup_srtx_state as cleanup_srtx_state
from .impl.ros2_common import ensure_render_var_on_product as ensure_render_var_on_product
from .impl.ros2_common import get_srtx_sensor_set_config as get_srtx_sensor_set_config
from .impl.ros2_common import get_srtx_sensor_set_name as get_srtx_sensor_set_name
from .impl.ros2_common import is_srtx_supported_platform as is_srtx_supported_platform
from .impl.ros2_common import prepare_srtx_sensor_set as prepare_srtx_sensor_set
from .impl.ros2_common import validate_srtx_platform as validate_srtx_platform

__all__ = [
    "ViewportManager",
    "SrtxSensorSetConfig",
    "SrtxCaptureState",
    "CompressedImageManager",
    "PointFieldDescription",
    "LIDAR_METADATA_WITHOUT_POINT_CLOUD_WARNING",
    "LIDAR_POINT_CLOUD_METADATA_OPTIONS",
    "RADAR_POINT_CLOUD_METADATA_OPTIONS",
    "RADAR_RADIAL_VELOCITY_AUX_OUTPUT_WARNING",
    "RGB_COMPRESSION_OPTIONS",
    "GenericPublisherKind",
    "RgbCompression",
    "LidarMetadataOption",
    "RadarMetadataOption",
    "Ros2CameraGraphConfig",
    "Ros2ClockGraphConfig",
    "Ros2GenericPublisherGraphConfig",
    "Ros2JointStatesGraphConfig",
    "Ros2OdometryGraphConfig",
    "Ros2RtxLidarGraphConfig",
    "Ros2RtxRadarGraphConfig",
    "Ros2TfGraphConfig",
    "read_camera_info",
    "compute_relative_pose",
    "collect_namespace",
    "register_node_writer_with_telemetry",
    "acquire_interface",
    "release_interface",
    "fill_point_cloud2_message",
    "interleave_point_cloud",
    "is_srtx_supported_platform",
    "validate_srtx_platform",
    "get_srtx_sensor_set_config",
    "get_srtx_sensor_set_name",
    "prepare_srtx_sensor_set",
    "ensure_render_var_on_product",
    "cleanup_srtx_state",
    "create_ros2_camera_graph",
    "create_ros2_clock_graph",
    "create_ros2_generic_publisher_graph",
    "create_ros2_joint_states_graph",
    "create_ros2_odometry_graph",
    "create_ros2_rtx_lidar_graph",
    "create_ros2_rtx_radar_graph",
    "create_ros2_tf_graph",
    "radar_supports_basic_aux_output",
    "set_isaac_name_override",
    "set_isaac_namespace",
]
