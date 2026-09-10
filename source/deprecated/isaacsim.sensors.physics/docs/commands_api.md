# Commands
Public command API for module **isaacsim.sensors.physics**:

- [IsaacSensorCreateContactSensor](#isaacsensorcreatecontactsensor)
- [IsaacSensorCreateImuSensor](#isaacsensorcreateimusensor)
- [IsaacSensorCreatePrim](#isaacsensorcreateprim)


## IsaacSensorCreateContactSensor
Create an Isaac Contact Sensor prim in the USD stage.

This command creates a contact sensor that detects physical contact between objects in a simulation.
The sensor monitors contact forces and reports when they fall within specified threshold ranges.
It automatically applies the PhysX Contact Report API to the parent prim to enable contact detection.

### Arguments
- path: USD path where the contact sensor prim will be created.
- parent: USD path of the parent prim. Must be specified for successful sensor creation.
- min_threshold: Minimum contact force threshold for detection.
- max_threshold: Maximum contact force threshold for detection.
- color: RGBA color values for sensor visualization.
- radius: Detection radius for the contact sensor. Negative values use default behavior.
- sensor_period: Data collection frequency in seconds. Negative values use default behavior.
- translation: 3D translation offset from the parent prim's origin.

### Usage

```python
import omni.kit.commands
import omni.usd
from pxr import Gf, UsdGeom

# Get the current USD stage
stage = omni.usd.get_context().get_stage()

# Create a parent prim for the contact sensor if it does not already exist
parent_path = "/World/ContactSensorParent"
if not stage.GetPrimAtPath(parent_path).IsValid():
    UsdGeom.Xform.Define(stage, parent_path)

# Create an Isaac contact sensor under the parent prim
success, contact_sensor = omni.kit.commands.execute(
    "IsaacSensorCreateContactSensor",
    path="/Contact_Sensor",
    parent=parent_path,
    min_threshold=0.0,
    max_threshold=100000.0,
    color=Gf.Vec4f(1.0, 0.0, 0.0, 1.0),
    radius=0.1,
    sensor_period=1.0 / 60.0,
    translation=Gf.Vec3d(0.0, 0.0, 0.0),
)
```

## IsaacSensorCreateImuSensor
Command for creating an IMU (Inertial Measurement Unit) sensor prim in the stage.

This command creates an Isaac IMU sensor that can measure linear acceleration, angular velocity, and
orientation. The sensor supports configurable filter sizes for smoothing the measured values and can be
positioned and oriented relative to its parent prim.

### Arguments
- path: USD path where the IMU sensor prim will be created.
- parent: USD path of the parent prim to attach the sensor to.
- sensor_period: Sensor update period in simulation time. Negative values use the default period.
- translation: Local translation offset from the parent prim.
- orientation: Local rotation offset from the parent prim as a quaternion (w, x, y, z).
- linear_acceleration_filter_size: Number of samples to use for linear acceleration filtering.
- angular_velocity_filter_size: Number of samples to use for angular velocity filtering.
- orientation_filter_size: Number of samples to use for orientation filtering.

### Usage

```python
import omni.kit.commands
import omni.usd
from pxr import Gf

# Get the current USD stage.
stage = omni.usd.get_context().get_stage()

# Create a parent prim for the IMU sensor if it does not already exist.
parent_path = "/World/Robot"
if not stage.GetPrimAtPath(parent_path):
    stage.DefinePrim(parent_path, "Xform")

# Create an IMU sensor under the parent prim.
success, imu_sensor = omni.kit.commands.execute(
    "IsaacSensorCreateImuSensor",
    path="/Imu_Sensor",
    parent=parent_path,
    sensor_period=1.0 / 60.0,
    translation=Gf.Vec3d(0.0, 0.0, 0.1),
    orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
    linear_acceleration_filter_size=1,
    angular_velocity_filter_size=1,
    orientation_filter_size=1,
)

if success and imu_sensor:
    print(f"Created IMU sensor at: {imu_sensor.GetPath()}")
```

## IsaacSensorCreatePrim
Command for creating Isaac sensor prims in the USD stage.

This command creates a new sensor prim using the specified Isaac sensor schema type and applies the provided
transformation properties. The sensor is created at the specified path with the given translation and orientation.
It serves as a base command for creating various types of Isaac sensors.

### Arguments
- path: USD path where the sensor prim will be created.
- parent: Parent prim path under which the sensor will be created.
- translation: Translation vector for positioning the sensor in 3D space.
- orientation: Quaternion rotation for orienting the sensor.
- schema_type: Isaac sensor schema type to apply to the created prim.

### Usage

```python
import omni.kit.commands
import omni.isaac.IsaacSensorSchema as IsaacSensorSchema
from pxr import Gf

# Create an Isaac base sensor prim under /World.
# The final path will be something like /World/Base_Sensor.
success, sensor_prim = omni.kit.commands.execute(
    "IsaacSensorCreatePrim",
    path="/Base_Sensor",
    parent="/World",
    translation=Gf.Vec3d(0.0, 0.0, 1.0),
    orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
    schema_type=IsaacSensorSchema.IsaacBaseSensor,
)

if success:
    print(f"Created sensor prim at: {sensor_prim.GetPath()}")
```

