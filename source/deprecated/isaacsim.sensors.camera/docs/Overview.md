# Overview

```{deprecated} 6.0.0
This extension is deprecated. Use {mod}`isaacsim.sensors.experimental.rtx` instead, which provides
{class}`RtxCamera <isaacsim.sensors.experimental.rtx.RtxCamera>`,
{class}`CameraSensor <isaacsim.sensors.experimental.rtx.CameraSensor>`,
{class}`TiledCameraSensor <isaacsim.sensors.experimental.rtx.TiledCameraSensor>`, and
{class}`SingleViewDepthCameraSensor <isaacsim.sensors.experimental.rtx.SingleViewDepthCameraSensor>`.
```

`**isaacsim.sensors.camera**` provides Python APIs for working with camera prims as simulation sensors. It helps create or wrap camera prims, configure camera properties such as pose, resolution, focal length, clipping range, lens distortion, and collect rendered sensor outputs such as RGB, depth, segmentation, bounding boxes, and point clouds.

The extension exposes two main APIs: {class}`Camera <isaacsim.sensors.camera.Camera>` for a single camera sensor and {class}`CameraView <isaacsim.sensors.camera.CameraView>` for batched access to multiple cameras. Use {class}`Camera <isaacsim.sensors.camera.Camera>` when you need direct control over one camera, and use {class}`CameraView <isaacsim.sensors.camera.CameraView>` when you need tiled or batched data from many cameras.

## Concepts

### {class}`Camera <isaacsim.sensors.camera.Camera>` prims and render products

A {class}`Camera <isaacsim.sensors.camera.Camera>` represents a camera prim at a specific `prim_path`. If a camera prim already exists at that path, {class}`Camera <isaacsim.sensors.camera.Camera>` wraps it. Otherwise, it creates a new camera prim.

Each camera is associated with a render product. You can let {class}`Camera <isaacsim.sensors.camera.Camera>` create one automatically, or provide an existing `render_product_path`. The same render product path should not be shared by two {class}`Camera <isaacsim.sensors.camera.Camera>` objects with different camera prims or resolutions.

### Annotators

{class}`Camera <isaacsim.sensors.camera.Camera>` image and sensor outputs are provided through annotators. Common annotator outputs include:

- `rgb` and `rgba`
- `distance_to_image_plane`
- `distance_to_camera`
- `normals`
- `motion_vectors`
- `occlusion`
- `bounding_box_2d_tight`
- `bounding_box_2d_loose`
- `bounding_box_3d`
- `semantic_segmentation`
- `instance_segmentation`
- `instance_id_segmentation`
- `pointcloud`

Annotators must be attached before their data appears in `get_current_frame()` or specialized getters such as `get_rgb()` and `get_depth()`.

### {class}`Camera <isaacsim.sensors.camera.Camera>` axes

Pose APIs support multiple camera axis conventions:

- `world`: `+Z` up, `+X` forward
- `ros`: `+Y` up, `+Z` forward
- `usd`: `+Y` up, `-Z` forward

This is useful when moving between stage transforms, robotics conventions, and camera calibration workflows.

## Key Components

### {class}`Camera <isaacsim.sensors.camera.Camera>`

{class}`Camera <isaacsim.sensors.camera.Camera>` provides high level control over a single camera sensor.

It supports:

- Creating or wrapping a camera prim
- Setting world and local poses
- Setting sensor update frequency or `dt`
- Configuring resolution and aspect ratio
- Attaching and detaching annotators
- Reading current frame data
- Reading RGB, RGBA, depth, and point cloud outputs
- Configuring focal length, apertures, focus distance, clipping range, shutter properties, projection mode, and stereo role
- Working with camera calibration helpers such as intrinsics and view matrices
- Projecting world points to image coordinates and inverse-projecting image coordinates with depth

A typical single-camera setup looks like this:

```python
from isaacsim.sensors.camera import Camera

camera = Camera(
    prim_path="/World/Camera",
    name="front_camera",
    resolution=(640, 480),
    frequency=30,
)

camera.initialize()
camera.add_rgb_to_frame()
camera.add_distance_to_image_plane_to_frame()

frame = camera.get_current_frame()
rgb = camera.get_rgb()
depth = camera.get_depth()
```

`initialize()` should be called before attaching annotators, because the camera needs a render product before annotator data can be collected.

### {class}`CameraView <isaacsim.sensors.camera.CameraView>`

{class}`CameraView <isaacsim.sensors.camera.CameraView>` provides batched access to multiple camera prims matched by a prim path expression. It is useful for multi-environment or multi-camera workflows where the same operation needs to apply to many cameras.

It supports:

- Matching camera prims with a path expression, such as `/World/Env[1-5]/Camera`
- Setting a shared camera resolution
- Configuring output annotators at construction time
- Reading data as a batch of images
- Reading data as a single tiled image
- Getting and setting poses for selected camera indices
- Getting and setting camera properties such as focal length, focus distance, apertures, projection mode, stereo role, and shutter properties

Example:

```python
from isaacsim.sensors.camera import CameraView

camera_view = CameraView(
    prim_paths_expr="/World/Env[1-5]/Camera",
    name="env_cameras",
    camera_resolution=(256, 256),
    output_annotators=["rgb", "distance_to_image_plane"],
)

rgb_batch = camera_view.get_rgb()
depth_batch = camera_view.get_depth()

rgb_tiled = camera_view.get_rgb_tiled(device="cpu")
depth_tiled = camera_view.get_depth_tiled(device="cpu")
```

`get_rgb()` and `get_depth()` return batched image tensors. The tiled getters return a single image containing all camera outputs arranged into tiles.

## Functionality

### Frame collection

{class}`Camera <isaacsim.sensors.camera.Camera>` stores the latest available sensor outputs in its current frame. You can retrieve the whole frame with `get_current_frame()` or use specific getters such as `get_rgb()`, `get_rgba()`, `get_depth()`, and `get_pointcloud()`.

A few rendered frames may be required after initialization before valid data becomes available.

```python
camera.initialize()
camera.add_rgb_to_frame()

# Step/render a few frames before expecting valid data.
rgb = camera.get_rgb()
if rgb is not None:
    print(rgb.shape)
```

### Sensor timing

{class}`Camera <isaacsim.sensors.camera.Camera>` can update at a requested `frequency` or `dt`, but both cannot be specified at the same time. The requested rate must align with the configured rendering frequency.

If `/app/runLoops/main/rateLimitFrequency` is not set, the requested frequency or `dt` cannot be honored. In that case, the camera processes every rendered frame and logs a warning.

### {class}`Camera <isaacsim.sensors.camera.Camera>` calibration

{class}`Camera <isaacsim.sensors.camera.Camera>` includes calibration-oriented helpers for pinhole projection workflows:

- `get_intrinsics_matrix()`
- `get_view_matrix_ros()`
- `get_image_coords_from_world_points()`
- `get_camera_points_from_image_coords()`
- `get_world_points_from_image_coords()`
- `get_horizontal_fov()`
- `get_vertical_fov()`

These APIs are useful when connecting rendered sensor data to perception pipelines that expect camera matrices and pixel-space projections.

```python
points_world = ...  # shape: (N, 3)

image_points = camera.get_image_coords_from_world_points(points_world)
intrinsics = camera.get_intrinsics_matrix()
```

Projection and inverse-projection helpers require a pinhole projection setup.

### Lens and image model configuration

{class}`Camera <isaacsim.sensors.camera.Camera>` provides camera property APIs for configuring the sensor model. This includes focal length, apertures, focus distance, clipping range, shutter timing, projection mode, stereo role, and lens distortion model properties.

Supported lens distortion configuration APIs include:

- `set_ftheta_properties()`
- `set_kannala_brandt_k3_properties()`
- `set_rad_tan_thin_prism_properties()`
- `set_lut_properties()`
- `set_opencv_pinhole_properties()`
- `set_opencv_fisheye_properties()`

These methods apply the corresponding distortion model to the camera prim and set the required model parameters.

## Usage Examples

### Single camera with RGB and depth

```python
from isaacsim.sensors.camera import Camera

camera = Camera(
    prim_path="/World/Robot/front_camera",
    resolution=(1280, 720),
    dt=1.0 / 30.0,
)

camera.initialize(attach_rgb_annotator=False)
camera.add_rgb_to_frame()
camera.add_distance_to_image_plane_to_frame()

rgb = camera.get_rgb()
depth = camera.get_depth()
frame = camera.get_current_frame()
```

### Move a camera using ROS camera axes

```python
import numpy as np
from isaacsim.sensors.camera import Camera

camera = Camera("/World/Camera")
camera.initialize()

camera.set_world_pose(
    position=np.array([1.0, 0.0, 1.5]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    camera_axes="ros",
)

position, orientation = camera.get_world_pose(camera_axes="ros")
```

### Batched camera data

```python
from isaacsim.sensors.camera import CameraView

cameras = CameraView(
    prim_paths_expr="/World/Env.*/Camera",
    camera_resolution=(320, 240),
    output_annotators=["rgb", "distance_to_image_plane"],
)

rgb = cameras.get_rgb()
depth = cameras.get_depth()

print(rgb.shape)    # (num_cameras, height, width, 3)
print(depth.shape)  # (num_cameras, height, width, 1)
```

## Considerations

- Call `Camera.initialize()` before attaching annotators or reading annotator data.
- Do not specify both `frequency` and `dt` for a {class}`Camera <isaacsim.sensors.camera.Camera>`.
- Requested camera update rates must align with the configured rendering frequency.
- {class}`CameraView <isaacsim.sensors.camera.CameraView>` requires its requested annotator types to be configured when the object is created.
- Projection and inverse-projection helpers are intended for pinhole projection.
- {class}`Camera <isaacsim.sensors.camera.Camera>` aperture APIs maintain square pixels when `maintain_square_pixels=True`.
