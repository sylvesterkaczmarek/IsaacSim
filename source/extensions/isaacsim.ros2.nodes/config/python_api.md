# Public API for module isaacsim.ros2.nodes:

## Classes

- class ViewportManager
  - class def wait_for_viewport(cls) -> tuple[bool, int]
  - class async def wait_for_viewport_async(cls) -> tuple[bool, int]
  - class def set_camera(cls, camera: str | Usd.Prim | UsdGeom.Camera)
  - class def get_camera(cls, render_product_or_viewport: str | Usd.Prim | UsdRender.Product | 'ViewportAPI' | None = None) -> UsdGeom.Camera
  - class def get_viewport_api(cls, render_product_or_viewport: str | Usd.Prim | UsdRender.Product | 'ViewportAPI' | None = None) -> 'ViewportAPI' | None
  - class def get_render_product(cls, render_product_or_viewport: str | Usd.Prim | UsdRender.Product | 'ViewportAPI' | None = None) -> UsdRender.Product | None
  - class def get_resolution(cls, render_product_or_viewport: str | Usd.Prim | UsdRender.Product | 'ViewportAPI' | None = None) -> tuple[int, int]
  - class def set_resolution(cls, resolution: tuple[int, int] | str)
  - class def create_viewport_window(cls) -> ViewportWindow
  - class def get_viewport_windows(cls) -> list
  - class def destroy_viewport_windows(cls) -> list[str]
  - class def set_camera_view(cls, camera: str | Usd.Prim | UsdGeom.Camera)

- class SrtxSensorSetConfig
  - name: str
  - render_product_paths: list[str] | None

- class SrtxCaptureState
  - def __init__(self)
  - def start_or_extend(self, srtx_instance: object, sensor_set_name: str, output_path: str)
  - def stop_or_shrink(self, srtx_instance: object, sensor_set_name: str, output_paths_to_remove: list[str])

- class CompressedImageManager
  - class def reset(cls)
  - class def attach(cls, render_product_path: str, compression_type: str = 'h264')
  - class def detach(cls, render_product_path: str, compression_type: str = 'h264')
  - class def get_writer(cls, render_product_path: str, use_system_time: bool = False, compression_type: str = 'h264') -> rep.Writer

- class Ros2CameraGraphConfig
  - graph_path: str
  - camera_prim: str
  - frame_id: str
  - node_namespace: str
  - camera_info_topic: str
  - add_to_existing_graph: bool
  - render_product_prim: str
  - publish_rgb: bool
  - rgb_topic: str
  - rgb_type: RgbCompression
  - publish_depth: bool
  - depth_topic: str
  - publish_depth_point_cloud: bool
  - depth_point_cloud_topic: str
  - publish_instance_segmentation: bool
  - instance_segmentation_topic: str
  - publish_semantic_segmentation: bool
  - semantic_segmentation_topic: str
  - publish_bbox_2d_tight: bool
  - bbox_2d_tight_topic: str
  - publish_bbox_2d_loose: bool
  - bbox_2d_loose_topic: str
  - publish_bbox_3d: bool
  - bbox_3d_topic: str

- class Ros2ClockGraphConfig
  - graph_path: str

- class Ros2GenericPublisherGraphConfig
  - graph_path: str
  - publisher_kind: GenericPublisherKind
  - topic_name: str | None
  - bool_value: bool
  - int64_value: int
  - string_value: str

- class Ros2JointStatesGraphConfig
  - graph_path: str
  - articulation_root: str
  - node_namespace: str
  - publish_topic: str
  - subscribe_topic: str
  - add_to_existing_graph: bool
  - publish_joint_states: bool
  - subscribe_joint_states: bool
  - move_robot_on_subscribe: bool

- class Ros2OdometryGraphConfig
  - graph_path: str
  - articulation_root: str
  - chassis_prim: str
  - node_namespace: str
  - odometry_topic: str
  - tf_topic: str
  - add_to_existing_graph: bool
  - publish_robot_tf: bool

- class Ros2RtxLidarGraphConfig
  - graph_path: str
  - lidar_prim: str
  - frame_id: str
  - node_namespace: str
  - add_to_existing_graph: bool
  - render_product_prim: str
  - publish_laser_scan: bool
  - laser_scan_topic: str
  - publish_point_cloud: bool
  - point_cloud_topic: str
  - metadata: set[LidarMetadataOption]

- class Ros2RtxRadarGraphConfig
  - graph_path: str
  - radar_prim: str
  - frame_id: str
  - node_namespace: str
  - add_to_existing_graph: bool
  - render_product_prim: str
  - point_cloud_topic: str
  - metadata: set[RadarMetadataOption]

- class Ros2TfGraphConfig
  - graph_path: str
  - target_prim: str
  - parent_prim: str
  - topic: str
  - node_namespace: str
  - add_to_existing_graph: bool
  - append_to_existing_tf_node: bool
  - existing_tf_node_path: str

## Functions

- def create_ros2_camera_graph(config: Ros2CameraGraphConfig) -> str
- def create_ros2_clock_graph(config: Ros2ClockGraphConfig) -> str
- def create_ros2_generic_publisher_graph(config: Ros2GenericPublisherGraphConfig) -> str
- def create_ros2_joint_states_graph(config: Ros2JointStatesGraphConfig) -> str
- def create_ros2_odometry_graph(config: Ros2OdometryGraphConfig) -> str
- def create_ros2_rtx_lidar_graph(config: Ros2RtxLidarGraphConfig) -> str
- def create_ros2_rtx_radar_graph(config: Ros2RtxRadarGraphConfig) -> str
- def create_ros2_tf_graph(config: Ros2TfGraphConfig) -> str
- def radar_supports_basic_aux_output(radar_prim: str) -> bool
- def read_camera_info(render_product_path: str) -> tuple
- def compute_relative_pose(left_camera_prim: Usd.Prim, right_camera_prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]
- def collect_namespace(namespace_input: str, render_product_path: str) -> str
- def register_node_writer_with_telemetry(*args: Any, **kwargs: Any)
- def acquire_interface(plugin_name: str = None, library_path: str = None) -> IRos2Nodes
- def release_interface(arg0: IRos2Nodes)
- def is_srtx_supported_platform() -> bool
- def validate_srtx_platform() -> bool
- def get_srtx_sensor_set_config(render_product_path: str | None = None) -> SrtxSensorSetConfig
- def get_srtx_sensor_set_name(render_product_path: str | None = None) -> str
- def prepare_srtx_sensor_set(srtx_instance: object, render_product_path: str) -> str | None
- def ensure_render_var_on_product(stage: object, render_product_path: str, aov_name: str, compression_type: str | None = None, is_image: bool = False) -> tuple[bool, str | None]
- def cleanup_srtx_state(state: object)
- def set_isaac_name_override(prim_path: str, name: str)
- def set_isaac_namespace(prim_path: str, namespace: str)

## Data

- RGB_COMPRESSION_OPTIONS
- LIDAR_POINT_CLOUD_METADATA_OPTIONS
- RADAR_POINT_CLOUD_METADATA_OPTIONS
- LIDAR_METADATA_WITHOUT_POINT_CLOUD_WARNING
- RADAR_RADIAL_VELOCITY_AUX_OUTPUT_WARNING
