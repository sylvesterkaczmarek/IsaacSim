# Overview

The ROS 2 Bridge extension enables communication between Isaac Sim and ROS 2 systems. It provides an integration layer for publishing and subscribing to ROS 2 topics and services through OmniGraph nodes and Action Graphs. This enables ROS 2 applications to interact with simulated environments in real time.

**Important**: ROS 2 publishers, subscribers, and services are active only during simulation playback. If `ROS_DISTRO` is unset, the Isaac Sim launchers configure the bundled ROS 2 libraries before startup. Source a system ROS 2 installation or workspace before launching Isaac Sim when the workflow requires custom message packages, external package resolution, or a non-default ROS environment.

## Functionality

The extension enables bidirectional data exchange between Isaac Sim and ROS 2 systems by providing ROS 2 message handling within the simulation environment. Communication flows through OmniGraph nodes, which can be assembled into Action Graphs for sensor publishing, command subscription, and service workflows.

### OmniGraph Integration

The bridge operates through OmniGraph nodes that handle ROS 2 communication. Each node can act as either a publisher or subscriber, converting data between Isaac Sim's internal representation and ROS 2 message formats. These nodes can be connected in Action Graphs to create robotics workflows, from sensor simulation to actuator control.

### Activation Behavior

ROS 2 communication nodes remain dormant until simulation playback begins. When simulation playback starts, configured publishers and subscribers activate and begin processing messages. This limits ROS 2 traffic to active simulation and prevents unintended messages during scene setup or editing. When simulation is paused or stopped, ROS 2 communication ceases until playback resumes.

## Integration

The extension consolidates multiple ROS 2-related extensions into a unified bridge:

- **isaacsim.ros2.core** provides the fundamental ROS 2 communication layer and message handling infrastructure, serving as the foundation for all ROS 2 operations
- **isaacsim.ros2.nodes** supplies the OmniGraph node implementations that perform ROS 2 publishing, subscribing, and service calls
- **isaacsim.ros2.examples** includes example scenes and Action Graphs demonstrating common ROS 2 integration patterns
- **isaacsim.ros2.ui** offers UI components for configuring and monitoring ROS 2 connections within the Isaac Sim interface

All ROS 2 Bridge settings are centrally defined in the isaacsim.ros2.core extension, providing a single configuration point for the entire bridge system.

## Considerations

### ROS 2 Environment Setup

The bridge requires a ROS 2 runtime before Isaac Sim starts. The Isaac Sim launchers configure the bundled ROS 2 libraries automatically when `ROS_DISTRO` is unset. Source a system ROS 2 installation or workspace in the parent terminal when the workflow requires custom message packages, external package resolution, a custom RMW implementation, or a non-default ROS distribution.
