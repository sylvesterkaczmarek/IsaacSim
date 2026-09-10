# Public API for module isaacsim.sensors.rtx:

## Classes

- class IsaacSensorCreateRtxSensor(omni.kit.commands.Command)
  - def __init__(self, path: str | None = None, parent: str | None = None, config: str | None = None, usd_path: str | None = None, translation: Gf.Vec3d | None = Gf.Vec3d(0, 0, 0), orientation: Gf.Quatd | None = Gf.Quatd(1, 0, 0, 0), visibility: bool = False, variant: str | dict[str, str] | None = None, force_camera_prim: bool = False, **kwargs: Any)
  - def do(self) -> Usd.Prim
  - def undo(self)

- class IsaacSensorCreateRtxLidar(IsaacSensorCreateRtxSensor)
  - def __init__(self, **kwargs: Any)
  - def do(self) -> Usd.Prim

- class IsaacSensorCreateRtxRadar(IsaacSensorCreateRtxSensor)
  - def do(self) -> Usd.Prim | None

- class IsaacSensorCreateRtxIDS(IsaacSensorCreateRtxSensor)
  - def __init__(self, **kwargs: Any)

- class IsaacSensorCreateRtxUltrasonic(IsaacSensorCreateRtxSensor)

- class Extension(omni.ext.IExt)
  - def on_startup(self, ext_id: str)
  - def on_shutdown(self)

- class LidarRtx(BaseSensor)
  - static def make_add_remove_deprecated_attr(deprecated_attr: str) -> list[Callable]
  - def __init__(self, prim_path: str, name: str = 'lidar_rtx', position: np.ndarray | None = None, translation: np.ndarray | None = None, orientation: np.ndarray | None = None, config_file_name: str | None = None, **kwargs: Any)
  - def get_render_product_path(self) -> str | None
  - def get_current_frame(self) -> dict
  - def get_annotators(self) -> dict
  - def attach_annotator(self, annotator_name: Literal[IsaacComputeRTXLidarFlatScan, IsaacExtractRTXSensorPointCloudNoAccumulator, IsaacCreateRTXLidarScanBuffer, StableIdMap, GenericModelOutput], **kwargs: object)
  - def detach_annotator(self, annotator_name: str)
  - def detach_all_annotators(self)
  - def get_writers(self) -> dict
  - def attach_writer(self, writer_name: str, **kwargs: object)
  - def detach_writer(self, writer_name: str)
  - def detach_all_writers(self)
  - def initialize(self, physics_sim_view: Any = None)
  - def post_reset(self)
  - def resume(self)
  - def pause(self)
  - def is_paused(self) -> bool
  - def get_horizontal_resolution(self) -> float | None
  - def get_horizontal_fov(self) -> float | None
  - def get_num_rows(self) -> int | None
  - def get_num_cols(self) -> int | None
  - def get_rotation_frequency(self) -> float | None
  - def get_depth_range(self) -> tuple[float, float] | None
  - def get_azimuth_range(self) -> tuple[float, float] | None
  - def enable_visualization(self)
  - def disable_visualization(self)
  - def add_point_cloud_data_to_frame(self)
  - def add_linear_depth_data_to_frame(self)
  - def add_intensities_data_to_frame(self)
  - def add_azimuth_range_to_frame(self)
  - def add_horizontal_resolution_to_frame(self)
  - def add_range_data_to_frame(self)
  - def add_azimuth_data_to_frame(self)
  - def add_elevation_data_to_frame(self)
  - def remove_point_cloud_data_to_frame(self)
  - def remove_linear_depth_data_to_frame(self)
  - def remove_intensities_data_to_frame(self)
  - def remove_azimuth_range_to_frame(self)
  - def remove_horizontal_resolution_to_frame(self)
  - def remove_range_data_to_frame(self)
  - def remove_azimuth_data_to_frame(self)
  - def remove_elevation_data_to_frame(self)
  - static def decode_stable_id_mapping(stable_id_mapping_raw: bytes) -> dict
  - static def get_object_ids(obj_ids: np.ndarray) -> list[int]

## Functions

- def delete_prim(prim_path: str)
- def add_reference_to_stage(usd_path: str, prim_path: str, prim_type: str = 'Xform') -> Usd.Prim
- def get_next_free_path(path: str, parent: str = None) -> str
- def reset_and_set_xform_ops(prim: Usd.Prim, translation: Gf.Vec3d, orientation: Gf.Quatd, scale: Gf.Vec3d = Gf.Vec3d([1.0, 1.0, 1.0]))
- def get_assets_root_path() -> str
- def register_annotator_from_node_with_telemetry(*args: Any, **kwargs: Any)
- def register_node_writer_with_telemetry(*args: Any, **kwargs: Any)
- def get_prim_at_path(prim_path: str, fabric: bool = False) -> Usd.Prim | usdrt.Usd._Usd.Prim | None
- def get_gmo_data(dataPtr: int | np.ndarray) -> gmo_utils.GenericModelOutput
- def apply_nonvisual_material(prim: Usd.Prim, base: str | int, coating: str | int = 'none', attribute: str | int = 'none') -> bool
- def get_material_id(prim: Usd.Prim) -> int
- def decode_material_id(material_id: int) -> tuple[str, str, str]

## Variables

- SUPPORTED_LIDAR_CONFIGS: dict[str, set[str] | list[dict[str, str]]]
- SUPPORTED_LIDAR_VARIANT_SET_NAME: str
- EXTENSION_NAME: str
- NONE_BASE: Dict
- METALS_BASE: Dict
- POLYMERS_BASE: Dict
- GLASS_BASE: Dict
- OTHER_BASE: Dict
- BASE_MATERIALS: Dict
- COATINGS: Dict
- ATTRIBUTES: Dict
- ATTR_PREFIX: Unknown
- ATTR_BASE: Unknown
- ATTR_COATING: Unknown
- ATTR_ATTRIBUTE: Unknown

## Other

- annotations: unknown
- Callable: unknown
- Path: unknown
- Any: unknown
- carb: unknown module
- omni.isaac.IsaacSensorSchema: public module
- omni.kit.commands: unknown module
- omni.kit.utils: unknown module
- omni.replicator.core: unknown module
- omni.usd: unknown module
- Gf: unknown
- Sdf: unknown
- Usd: unknown module
- UsdGeom: unknown
- ctypes: builtin module
- gc: builtin module
- carb.settings: unknown module
- numpy: unknown module
- omni.ext: unknown module
- omni.graph.core: unknown module
- AnnotatorRegistry: unknown
- sensors: unknown
- UsdShade: unknown


# Public API for module isaacsim.sensors.rtx.generic_model_output:

No public API

# Public API for module isaacsim.sensors.rtx.generic_model_output:

No public API

# Public API for module isaacsim.sensors.rtx.sensor_checker:

No public API
