# Overview

```{deprecated} 6.0.0
This extension is deprecated in favor of `isaacsim.sensors.experimental.rtx`.
```

`**isaacsim.sensors.rtx**` provides Python APIs for RTX-based sensor simulation, including creation commands for RTX Lidar, RTX Radar, and RTX IDS sensors. It is mainly used to create sensor prims from configs or USD assets, then collect sensor output through annotators and writers. RTX Radar creation has one important requirement: Motion BVH must be enabled, otherwise {class}`IsaacSensorCreateRtxRadar <isaacsim.sensors.rtx.IsaacSensorCreateRtxRadar>` logs a warning and does not create a prim.

## Concepts

### Sensor creation commands

The extension exposes command classes for creating RTX sensor prims:

- {class}`IsaacSensorCreateRtxLidar <isaacsim.sensors.rtx.IsaacSensorCreateRtxLidar>`
- {class}`IsaacSensorCreateRtxRadar <isaacsim.sensors.rtx.IsaacSensorCreateRtxRadar>`
- {class}`IsaacSensorCreateRtxIDS <isaacsim.sensors.rtx.IsaacSensorCreateRtxIDS>`

These commands inherit from `**omni.kit.commands.Command**`, so they can be executed through the Kit command system and support undo by deleting the created prim.

The shared creation parameters include:

- `path`: target path for the sensor prim
- `parent`: parent prim path
- `config`: named sensor configuration
- `usd_path`: USD asset path for the sensor
- `translation`: sensor placement
- `orientation`: sensor orientation
- `visibility`: sensor visibility
- `variant`: sensor variant selection
- `force_camera_prim`: forces direct camera prim creation

If both `config` and `usd_path` are provided, `config` takes precedence.

### Lidar frames

{class}`LidarRtx <isaacsim.sensors.rtx.LidarRtx>` is the main runtime API for working with an RTX Lidar sensor. It wraps an existing Lidar prim and provides access to the current frame through `get_current_frame()`.

A frame contains timing information, frame number information, and data from any attached annotators. The exact contents depend on which annotators are attached.

### Annotators and writers

Annotators add sensor outputs to the current frame. Writers are used for output handling or visualization workflows.

Supported `LidarRtx.attach_annotator()` values include:

- `IsaacComputeRTXLidarFlatScan`
- `IsaacExtractRTXSensorPointCloudNoAccumulator`
- `IsaacCreateRTXLidarScanBuffer`
- `StableIdMap`
- `GenericModelOutput`

Writers are attached by name with `attach_writer()`, such as `RtxLidarDebugDrawPointCloud`.

## Key Components

### {class}`LidarRtx <isaacsim.sensors.rtx.LidarRtx>`

{class}`LidarRtx <isaacsim.sensors.rtx.LidarRtx>` provides the Python interface for creating and managing an RTX-based Lidar sensor object in a simulation script.

It accepts a `prim_path`, optional transform values, and an optional `config_file_name`. The prim at `prim_path` must be an `OmniLidar` or have the required sensor API, otherwise construction raises an exception.

Common operations include:

- `initialize()` to prepare sensor data acquisition
- `get_current_frame()` to read the latest frame data
- `attach_annotator()` and `detach_annotator()` to control frame outputs
- `attach_writer()` and `detach_writer()` to connect writers
- `pause()`, `resume()`, and `is_paused()` to control data acquisition
- `enable_visualization()` and `disable_visualization()` for Lidar point cloud visualization
- `get_render_product_path()` to inspect the render product used by the sensor

### {class}`IsaacSensorCreateRtxLidar <isaacsim.sensors.rtx.IsaacSensorCreateRtxLidar>`

{class}`IsaacSensorCreateRtxLidar <isaacsim.sensors.rtx.IsaacSensorCreateRtxLidar>` creates an RTX Lidar prim. After creation, it applies Lidar-specific output settings, including keeping invalid points, accumulating outputs, and mapping `auxOutputType` to the Replicator `RenderVar` channels attribute.

Use this command when you want to create the sensor prim first, then wrap it with {class}`LidarRtx <isaacsim.sensors.rtx.LidarRtx>` for frame access.

```{note}
This extension is deprecated. Prefer the experimental RTX sensor APIs, where
`Lidar(..., aux_output_level="EXTRA")` and `Radar(..., aux_output_level="BASIC")`
author the metadata needed by GenericModelOutput.

For this legacy command API, set aux output attributes such as
`omni:sensor:Core:auxOutputType` at sensor creation time and before creating the
render product or attaching GenericModelOutput/ScanBuffer annotators. Changing
`auxOutputType` after the sensor prim has already been authored is not a
supported way to reconfigure an existing GenericModelOutput RenderVar; recreate
the render product/annotator path or use the experimental API instead.
```

### {class}`IsaacSensorCreateRtxRadar <isaacsim.sensors.rtx.IsaacSensorCreateRtxRadar>`

{class}`IsaacSensorCreateRtxRadar <isaacsim.sensors.rtx.IsaacSensorCreateRtxRadar>` creates an RTX Radar prim. It checks Motion BVH settings before creating the sensor. If Motion BVH is not enabled, the command returns `None`.

For a valid Radar prim, it maps Radar `auxOutputType` to the `RenderVar` channels attribute.

### {class}`IsaacSensorCreateRtxIDS <isaacsim.sensors.rtx.IsaacSensorCreateRtxIDS>`

{class}`IsaacSensorCreateRtxIDS <isaacsim.sensors.rtx.IsaacSensorCreateRtxIDS>` creates an RTX Idealized Depth Sensor. If no config is provided, it uses `idsoccupancy` as the default configuration.

## Functionality

### Create RTX sensors

The creation commands can be executed through `**omni.kit.commands**`. This is useful when sensor creation should participate in the command and undo system.

```python
import omni.kit.commands
from pxr import Gf

# Use a supported Lidar config name for your installation.
config_name = "..."

prim = omni.kit.commands.execute(
    "IsaacSensorCreateRtxLidar",
    path="/World/Lidar",
    config=config_name,
    translation=Gf.Vec3d(0.0, 0.0, 1.0),
    orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
    visibility=False,
)

print(prim.GetPath())
```

### Read Lidar data

After a Lidar prim exists, use {class}`LidarRtx <isaacsim.sensors.rtx.LidarRtx>` to attach annotators and read the current frame.

```python
from isaacsim.sensors.rtx import LidarRtx

lidar = LidarRtx(prim_path="/World/Lidar")
lidar.attach_annotator("IsaacComputeRTXLidarFlatScan")
lidar.initialize()

frame = lidar.get_current_frame()

print(frame.keys())
print(frame.get("rendering_time"))
```

### Visualize Lidar output

{class}`LidarRtx <isaacsim.sensors.rtx.LidarRtx>` can attach writers for visualization or output workflows. For point cloud debug drawing, attach a writer such as `RtxLidarDebugDrawPointCloud`.

```python
from isaacsim.sensors.rtx import LidarRtx

lidar = LidarRtx(prim_path="/World/Lidar")
lidar.attach_writer("RtxLidarDebugDrawPointCloud")
lidar.enable_visualization()
```

### Decode object IDs and labels

{class}`LidarRtx <isaacsim.sensors.rtx.LidarRtx>` includes helper methods for working with object identity outputs from `StableIdMap` and `GenericModelOutput`.

```python
from isaacsim.sensors.rtx import LidarRtx

lidar = LidarRtx(prim_path="/World/Lidar")
lidar.attach_annotator("StableIdMap")
lidar.attach_annotator("IsaacCreateRTXLidarScanBuffer")
lidar.initialize()

frame = lidar.get_current_frame()

stable_id_data = frame.get("StableIdMap")
scan_buffer = frame.get("IsaacCreateRTXLidarScanBuffer")

if stable_id_data is not None and scan_buffer is not None:
    stable_id_to_label = LidarRtx.decode_stable_id_mapping(stable_id_data)
    object_ids = LidarRtx.get_object_ids(scan_buffer["objectId"])

    labels = [stable_id_to_label.get(object_id) for object_id in object_ids]
    print(labels)
```

## Configuration

The extension defines sensor-related settings that affect RTX sensor output behavior:

- `app.sensors.nv.lidar.outputBufferOnGPU`: controls whether the renderer keeps the Lidar return buffer on GPU for post-processing.
- `app.sensors.nv.radar.outputBufferOnGPU`: controls whether the renderer keeps the Radar return buffer on GPU for post-processing.
- `rtx.materialDb.nonVisualMaterialCSV.enabled`: enables non-visual materials using USD attributes.
- `rtx.materialDb.nonVisualMaterialSemantics.prefix`: sets the USD attribute prefix used for non-visual material semantics.
- `rtx.rtxsensor.useHydraTimeAlways`: uses Hydra time from `**omni.timeline**` in RTX sensor models when multi-tick rendering is disabled.

## Relationships

{class}`LidarRtx <isaacsim.sensors.rtx.LidarRtx>` inherits from `**isaacsim.core.api.sensors.base_sensor.BaseSensor**`, so it follows the same general sensor object pattern used by other Isaac Sim sensor APIs.

The sensor creation classes inherit from `**omni.kit.commands.Command**`, which gives them command execution and undo behavior.

The modules `**isaacsim.sensors.rtx.generic_model_output**` and `**isaacsim.sensors.rtx.sensor_checker**` forward their public symbols from `**isaacsim.sensors.experimental.rtx.generic_model_output**` and `**isaacsim.sensors.experimental.rtx.sensor_checker**`.

The extension is backed by a Carbonite C++ plugin (`isaacsim.sensors.rtx.plugin`) with an `_isaacsim_sensors_rtx` Python binding module. The plugin registers the OmniGraph nodes that implement the RTX sensor annotators, such as `IsaacComputeRTXLidarFlatScan` and `IsaacCreateRTXLidarScanBuffer`, which {class}`LidarRtx <isaacsim.sensors.rtx.LidarRtx>` attaches to read sensor output.
