# Overview

```{deprecated} 6.0.0
This extension is deprecated in favor of `isaacsim.sensors.experimental.physics`.
```

`**isaacsim.sensors.physx**` provides APIs for creating and working with PhysX-based range sensors in simulation. It focuses on raycast and overlap-based sensing, including rotating lidar, generic range sensors, light beam sensors, and proximity detection zones. The extension is useful when you need simulated distance, point cloud, semantic, or overlap information from USD prims during physics simulation.

## Concepts

### Range sensor prims

Range sensors are represented as USD prims with sensor-specific attributes such as range, field of view, resolution, sampling rate, and visualization options. The command APIs create these prims and support undo by removing the created prim when undone.

Common range sensor settings include:

- `min_range` and `max_range` for valid detection distance.
- `draw_points` and `draw_lines` for sensor visualization.
- `translation` and `orientation` for placing the sensor in the stage.

### Rotating lidar frames

{class}`RotatingLidarPhysX <isaacsim.sensors.physx.RotatingLidarPhysX>` collects lidar data into a frame dictionary. The frame always tracks timing information such as simulation time and physics step, and you can opt in to specific data types such as depth, intensity, point cloud, azimuth, zenith, and semantics.

Only enabled data types are added to the current frame, which lets you control what the sensor collects.

### Proximity zones

{class}`ProximitySensor <isaacsim.sensors.physx.ProximitySensor>` uses the parent prim transform and scale to define a box-shaped detection zone. It performs physics overlap checks and tracks which collision objects entered, remain inside, or exited the zone.

Each overlap entry can include metadata such as:

- `duration`: time since the overlap began.
- `distance`: distance from the sensor origin to the overlapped geometry origin.

## Functionality

### Creating range sensors

The extension provides command classes for creating several sensor prim types:

- {class}`RangeSensorCreateLidar <isaacsim.sensors.physx.RangeSensorCreateLidar>` creates a rotating lidar sensor with field of view, resolution, rotation rate, range, visualization, yaw offset, and optional semantics.
- {class}`RangeSensorCreateGeneric <isaacsim.sensors.physx.RangeSensorCreateGeneric>` creates a generic range sensor with a configurable sampling rate.
- {class}`IsaacSensorCreateLightBeamSensor <isaacsim.sensors.physx.IsaacSensorCreateLightBeamSensor>` creates a light beam sensor with one or more rays, curtain length, direction axes, and range settings.
- {class}`RangeSensorCreatePrim <isaacsim.sensors.physx.RangeSensorCreatePrim>` is the base command used to create range sensor prims with shared attributes.

These commands are intended to be executed through `**omni.kit.commands.execute**`, which returns the created schema object when successful.

```python
import omni.kit.commands
from pxr import Gf

result, lidar = omni.kit.commands.execute(
    "RangeSensorCreateLidar",
    path="/Lidar",
    parent=None,
    translation=Gf.Vec3d(0, 0, 0),
    orientation=Gf.Quatd(1, 0, 0, 0),
    min_range=0.4,
    max_range=100.0,
    draw_points=False,
    draw_lines=False,
    horizontal_fov=360.0,
    vertical_fov=30.0,
    horizontal_resolution=0.4,
    vertical_resolution=4.0,
    rotation_rate=20.0,
    high_lod=False,
    yaw_offset=0.0,
    enable_semantics=False,
)
```

### Collecting lidar data

{class}`RotatingLidarPhysX <isaacsim.sensors.physx.RotatingLidarPhysX>` is a `BaseSensor` implementation for rotating lidar data acquisition. It can create or attach to a lidar prim path, then collect data during physics steps.

You can configure the sensor with either `rotation_frequency` or `rotation_dt`, but not both.

```python
from isaacsim.sensors.physx import RotatingLidarPhysX

lidar = RotatingLidarPhysX(
    prim_path="/World/Lidar",
    name="rotating_lidar_physX",
    rotation_frequency=20.0,
    fov=(360.0, 30.0),
    resolution=(0.4, 4.0),
    valid_range=(0.4, 100.0),
)

lidar.initialize()

lidar.add_depth_data_to_frame()
lidar.add_intensity_data_to_frame()
lidar.add_point_cloud_data_to_frame()

frame = lidar.get_current_frame()
```

The sensor can be paused and resumed without destroying it:

```python
lidar.pause()

if lidar.is_paused():
    lidar.resume()
```

### Tracking proximity overlaps

{class}`ProximitySensor <isaacsim.sensors.physx.ProximitySensor>` detects overlapping physics objects using box overlap queries. The parent prim provides the sensor pose, and the parent prim scale defines the detection box dimensions.

Callbacks can be supplied for enter, inside, and exit events. Each callback receives the {class}`ProximitySensor <isaacsim.sensors.physx.ProximitySensor>` instance.

```python
from isaacsim.sensors.physx import ProximitySensor

def on_enter(sensor):
    print("Entered:", sensor.get_entered_zones())

def on_inside(sensor):
    print("Overlapping:", sensor.get_active_zones())

def on_exit(sensor):
    print("Exited:", sensor.get_exited_zones())

sensor = ProximitySensor(
    parent=my_sensor_prim,
    callback_fns=[on_enter, on_inside, on_exit],
    exclusions=["/World/IgnoreThisPrim"],
)

sensor.update()

overlapping, data = sensor.status()
print(overlapping)
print(data)
```

The proximity sensor also provides direct accessors for active, entered, and exited zones:

```python
if sensor.is_overlapping():
    print(sensor.get_data())

sensor.reset()
```

## Key Components

### {class}`RotatingLidarPhysX <isaacsim.sensors.physx.RotatingLidarPhysX>`

{class}`RotatingLidarPhysX <isaacsim.sensors.physx.RotatingLidarPhysX>` provides rotating lidar simulation with configurable field of view, resolution, rotation rate, and valid range. It supports optional frame data channels including depth, linear depth, intensity, zenith, azimuth, point cloud, and semantics.

It also includes visualization controls:

```python
lidar.enable_visualization(
    high_lod=False,
    draw_points=True,
    draw_lines=True,
)

lidar.disable_visualization()
```

### {class}`RangeSensorCreateLidar <isaacsim.sensors.physx.RangeSensorCreateLidar>`

{class}`RangeSensorCreateLidar <isaacsim.sensors.physx.RangeSensorCreateLidar>` creates a lidar sensor prim through the command system. It is the main creation command when you need a lidar prim with explicit horizontal and vertical field of view, resolution, rotation rate, and optional semantic output.

### {class}`RangeSensorCreateGeneric <isaacsim.sensors.physx.RangeSensorCreateGeneric>`

{class}`RangeSensorCreateGeneric <isaacsim.sensors.physx.RangeSensorCreateGeneric>` creates a generic range sensor prim. Use it when you need a configurable sampling rate but do not need the rotating lidar-specific settings.

```python
import omni.kit.commands
from pxr import Gf

result, sensor = omni.kit.commands.execute(
    "RangeSensorCreateGeneric",
    path="/GenericSensor",
    parent=None,
    translation=Gf.Vec3d(0, 0, 0),
    orientation=Gf.Quatd(1, 0, 0, 0),
    min_range=0.4,
    max_range=100.0,
    draw_points=False,
    draw_lines=False,
    sampling_rate=60,
)
```

### {class}`IsaacSensorCreateLightBeamSensor <isaacsim.sensors.physx.IsaacSensorCreateLightBeamSensor>`

{class}`IsaacSensorCreateLightBeamSensor <isaacsim.sensors.physx.IsaacSensorCreateLightBeamSensor>` creates a light beam sensor prim. It supports single-ray and multi-ray style setups through `num_rays`, `curtain_length`, `forward_axis`, and `curtain_axis`.

```python
import omni.kit.commands
from pxr import Gf

result, light_beam = omni.kit.commands.execute(
    "IsaacSensorCreateLightBeamSensor",
    path="/LightBeam_Sensor",
    parent=None,
    translation=Gf.Vec3d(0, 0, 0),
    orientation=Gf.Quatd(1, 0, 0, 0),
    num_rays=1,
    curtain_length=0.0,
    forward_axis=Gf.Vec3d(1, 0, 0),
    curtain_axis=Gf.Vec3d(0, 1, 0),
    min_range=0.4,
    max_range=100.0,
    draw_points=False,
    draw_lines=False,
)
```

### {class}`ProximitySensor <isaacsim.sensors.physx.ProximitySensor>`

{class}`ProximitySensor <isaacsim.sensors.physx.ProximitySensor>` is used for overlap-based detection rather than raycast range measurement. It is useful for trigger volumes, detection zones, and workflows where the important event is whether another physics object is inside a region.

The sensor tracks state transitions across updates, so callers can distinguish newly entered objects, currently active overlaps, and objects that just exited.

## Relationships

{class}`RotatingLidarPhysX <isaacsim.sensors.physx.RotatingLidarPhysX>` inherits from `**isaacsim.core.api.sensors.base_sensor.BaseSensor**`, so it follows the same sensor pattern for initialization, reset behavior, and frame access.

The range sensor creation APIs are `**omni.kit.commands.Command**` classes. They are executed through `**omni.kit.commands.execute**` and provide `do` and `undo` behavior for creating and removing sensor prims.

The proximity workflow uses USD prims for sensor placement and PhysX overlap queries for detection. The parent `Usd.Prim` defines the proximity sensor transform, while its scale defines the query box size.

The extension is backed by a Carbonite C++ plugin and a `_range_sensor` Python binding module. The plugin exposes the underlying sensor query interfaces (`LidarSensorInterface`, `GenericSensorInterface`, and `LightBeamSensorInterface`) that the Python sensor classes acquire to read raycast results during simulation.
