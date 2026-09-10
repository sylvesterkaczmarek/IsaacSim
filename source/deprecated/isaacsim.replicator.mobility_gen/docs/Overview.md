# Overview

```{deprecated} 6.0.0
Extension deprecated since Isaac Sim 6.0.0 in favor of the experimental extension: isaacsim.replicator.experimental.mobility_gen
```

`**isaacsim.replicator.mobility_gen**` is a toolset for generating mobility data for robots in simulated environments. It focuses on robot movement, occupancy-map-based navigation, camera sensor capture, and structured recording of simulation state. The extension provides the building blocks for creating mobility generation scenarios, collecting robot and sensor data, and replaying or reading generated recordings.

## Concepts

### Scenario

A mobility generation scenario combines a robot with an `OccupancyMap`. The scenario owns the robot, keeps both the original and buffered occupancy maps, and defines how the robot is reset and stepped through the environment.

Scenario implementations are organized through the `SCENARIOS` registry, so different scenario types can be selected by name when loading a saved scenario configuration.

### Robot

`MobilityGenRobot` represents a robot that can move through a scenario and expose its state through buffers. Robot implementations define how the robot is built and how actions are written into the simulation.

The base robot tracks common state such as position, orientation, joint positions, joint velocities, linear velocity, and angular velocity. It can also create a front camera and a chase camera using configured offsets and rotations.

### Occupancy Map

`OccupancyMap` represents a 2D navigation map with three cell states:

- `UNKNOWN`
- `FREESPACE`
- `OCCUPIED`

The map supports ROS-style image and YAML export, coordinate conversion between pixel and world space, and buffering occupied regions for collision-aware navigation. Buffered maps are useful when planning motion for robots that need clearance around obstacles.

### Module and Buffer

`Module` and `Buffer` provide the state management model used across the extension. A `Module` can contain child modules and tagged buffers, while a `Buffer` stores a value with optional tags such as `rgb`, `segmentation`, `depth`, or `normals`.

This tagging system lets scenarios separate lightweight common state from larger image outputs.

## Functionality

### Mobility Data Generation

The extension supports robot mobility generation workflows built around a scenario, robot, and occupancy map. A scenario can be stepped forward over time while the robot state and enabled sensor outputs are collected into state dictionaries.

Typical captured state includes:

- Robot pose and velocity
- Joint positions and joint velocities
- Camera pose
- RGB images
- Semantic segmentation
- Instance ID segmentation
- Depth images
- Surface normals

### Camera Capture

`MobilityGenCamera` wraps a USD camera prim and can enable different rendering outputs independently. It uses render products and annotators to collect camera data from the current simulation view.

Supported outputs include:

- RGB color images
- Semantic segmentation masks and metadata
- Instance ID segmentation masks and metadata
- Depth maps
- Surface normals
- Camera position and orientation

Rendering outputs are stored in tagged buffers, so they can be saved and loaded separately from common simulation state.

### Input Control

The extension includes keyboard and gamepad input modules for manual control or human-in-the-loop data collection.

`Keyboard` monitors the W, A, S, and D keys and exposes their pressed states through a button buffer. `Gamepad` connects to the first available gamepad and tracks four analog stick axes: left vertical, left horizontal, right vertical, and right horizontal.

These inputs can be used by scenario logic or robot action logic to produce movement commands.

### Path and Pose Sampling

The extension includes utilities for selecting valid robot poses and generating paths through freespace.

`UniformPoseSampler` samples a random pose uniformly from freespace in an occupancy map. `GridPoseSampler` partitions the map into grid regions, samples a region, then samples a freespace pose inside it.

Path utilities can generate shortest paths from a start cell, sample reachable end points, unroll paths, and compress redundant points from approximately straight path segments.

The core path search is implemented in C++ and exposed through the `_path_planner` pybind11 module, which provides `generate_paths` and `unroll_path` functions used by the Python path utilities.

## Key Components

### MobilityGenScenario

`MobilityGenScenario` is the base class for scenario behavior. It stores the robot, the source occupancy map, and a buffered occupancy map derived from the robot collision radius.

Subclasses define the actual reset and step behavior. The base class also provides a visualization image using the ROS image representation of the occupancy map.

### MobilityGenRobot

`MobilityGenRobot` is the base robot abstraction used by scenarios. It manages robot state buffers, camera construction, replay state writing, and 2D pose conversion.

Robot subclasses provide the build logic and the action-writing logic. Registered robot types are stored in the `ROBOTS` registry.

### MobilityGenCamera

`MobilityGenCamera` manages camera sensor outputs for mobility recordings. It can enable RGB, segmentation, instance segmentation, depth, and normals capture, then update its buffers from the active annotators.

This component is the main source of image-like training data in a mobility generation scenario.

### OccupancyMap

`OccupancyMap` stores navigability information and provides the spatial operations needed for robot placement and path planning. It can save and load ROS-compatible occupancy map files, convert between world and pixel coordinates, and create buffered maps by dilating occupied regions.

### MobilityGenWriter and MobilityGenReader

`MobilityGenWriter` saves generated mobility data into a structured recording directory. `MobilityGenReader` reads the same structure back and provides indexed access to recorded timesteps.

Together, they support recording, inspection, and replay workflows.

## Data

Mobility generation recordings are organized by data type. Common state is saved separately from larger image outputs so recordings can be read selectively.

The writer supports:

- Common state dictionaries as NumPy files
- RGB images as JPEG files
- Segmentation images as PNG files
- Depth images as 16-bit inverse depth PNG files
- Normals as NumPy files
- Scenario configuration as JSON
- Occupancy maps in ROS format
- USD or USDZ stage copies

The reader can load each data type independently or return a complete state dictionary for a timestep. It also supports sequence-style access, so a recording can be indexed by frame.

## Usage Examples

### Load a Scenario

A saved mobility generation scenario can be loaded from a directory containing its configuration, stage, and occupancy map data.

```python
from isaacsim.replicator.mobility_gen.impl.build import load_scenario

scenario = load_scenario("/path/to/mobility_recording")
```

### Read a Recording

`MobilityGenReader` discovers available timesteps and sensor folders from a recording directory.

```python
from isaacsim.replicator.mobility_gen.impl.reader import MobilityGenReader

reader = MobilityGenReader("/path/to/mobility_recording")

config = reader.read_config()
occupancy_map = reader.read_occupancy_map()

state = reader[0]
rgb_state = reader.read_state_dict_rgb(0)
depth_state = reader.read_state_dict_depth(0)
```

### Save Scenario State

A scenario can produce tagged state dictionaries that are written by `MobilityGenWriter`.

```python
from isaacsim.replicator.mobility_gen.impl.writer import MobilityGenWriter

writer = MobilityGenWriter("/path/to/output")

scenario.update_state()

writer.write_state_dict_common(scenario.state_dict_common(), step=0)
writer.write_state_dict_rgb(scenario.state_dict_rgb(), step=0)
writer.write_state_dict_segmentation(scenario.state_dict_segmentation(), step=0)
writer.write_state_dict_depth(scenario.state_dict_depth(), step=0)
writer.write_state_dict_normals(scenario.state_dict_normals(), step=0)
```

### Work With an Occupancy Map

Occupancy maps can be buffered for safer robot placement or path planning.

```python
from isaacsim.replicator.mobility_gen.impl.occupancy_map import OccupancyMap
from isaacsim.replicator.mobility_gen.impl.types import Point2d

occupancy_map = OccupancyMap.from_ros_yaml("/path/to/map.yaml")

buffered_map = occupancy_map.buffered_meters(0.25)

point = Point2d(1.0, 2.0)
is_valid = buffered_map.check_world_point_in_freespace(point)
```

## Relationships

`MobilityGenCamera` uses `**omni.replicator.core**` render products and annotators to capture RGB, segmentation, depth, and normals data.

`MobilityGenRobot` works with `**isaacsim.core.api.robots.robot.Robot**` and `**isaacsim.core.prims.Articulation**` to read robot state and apply replay data. The utility functions also create and access an `**isaacsim.core.api.World**` for simulation setup.

USD stage and prim utilities use `pxr.Usd`, `pxr.UsdGeom`, and `pxr.Gf` for camera creation, transform operations, and pose extraction.
