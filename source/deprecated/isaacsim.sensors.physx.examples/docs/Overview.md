# Overview

```{deprecated} 6.0.0
This extension is deprecated in favor of `isaacsim.sensors.physics.examples`.
```

`**isaacsim.sensors.physx.examples**` provides interactive examples for PhysX raycast based sensors in Isaac Sim. It demonstrates how to create sensor scenes, configure sensor behavior, add collision objects, and inspect sensor output through example UI panels. The examples focus on LIDAR, Generic Range Sensor, and LightBeam sensor workflows.

## Concepts

### PhysX raycast based sensing

The examples use PhysX line tracing or ray casting to detect collisions and measure distances. This makes the sensors useful for testing LIDAR-like behavior, custom scanning patterns, and simple beam-based detection against physics collision objects.

A valid physics scene is required for the demonstrations because the sensor output depends on collision queries. Some examples create the required `PhysicsScene` and configure the stage automatically.

### Sensor output

The examples expose sensor output in different forms depending on the sensor type:

- LIDAR data, including depth, azimuth, and zenith measurements.
- Generic Range Sensor point cloud data from custom scanning patterns.
- LightBeam hit status, linear depth, and hit position data.

## Functionality

### PhysX LIDAR example

The LIDAR example demonstrates how to load and configure a PhysX LIDAR sensor. The UI includes controls for creating a sensor, spawning obstacles, and viewing the live data stream.

Users can test common LIDAR configuration values such as field of view, resolution, range, and rotation rate. The example also visualizes LIDAR rays and detection points in the viewport, making it easier to understand how the sensor reacts to scene geometry.

### Generic Range Sensor example

The Generic Range Sensor example shows how to create a PhysX based range sensor with custom scanning patterns. It includes controls for loading the sensor, creating an obstacle scene, setting a custom sensor pattern, streaming data, and saving pattern visualization images.

The example supports both continuous streaming and repeating mode for predefined patterns. It also processes point cloud data to filter points that hit a wall surface and extracts the relevant Y-Z coordinates for pattern analysis.

### LightBeam sensor example

The LightBeam example creates a simple scene with a movable cube and a LightBeam sensor. The sensor continuously checks for hits and reports live data such as beam hit status, linear depth, and hit positions.

The demo visualizes 5 light beam rays with color-coded display. Users can move the cube in the scene and observe how the beam data changes as the target intersects or moves away from the rays.

## UI Components

### Examples browser entries

The extension registers its sensor demonstrations with the Isaac Sim examples browser. Each example is presented as an interactive entry that opens a focused UI for the selected sensor workflow.

### Command panels

The LIDAR and Generic Range Sensor examples provide command panels for creating sensors, spawning test obstacles, and controlling data output. These panels are intended to guide users through a complete sensor setup and testing workflow.

### Data display areas

The LIDAR example displays live sensor measurements in a formatted table. The LightBeam example displays live beam hit, depth, and position information while the simulation runs.

### Viewport visualization

The examples use viewport visualization to make sensor behavior visible. LIDAR rays, detection points, Generic Range Sensor point cloud patterns, and LightBeam rays can be inspected directly in the scene.

## Relationships

`**isaacsim.sensors.physx.examples**` demonstrates sensor functionality from `**isaacsim.sensors.physx**`. The examples use PhysX based ray casting and line tracing to generate distance and hit data.

The extension also registers its demonstrations with `**isaacsim.examples.browser**`, which is how users access the individual sensor examples from the examples browser UI.

## Requirements

The sensor examples depend on physics collision queries. Scenes used by the examples must include collision objects and a `PhysicsScene` for the raycast based sensors to produce meaningful hit data.
