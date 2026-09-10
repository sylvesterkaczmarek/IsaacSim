# Overview

The `isaacsim.ros2.nodes` extension provides OmniGraph nodes that enable ROS 2 communication within Isaac Sim. This extension bridges robotics simulation with ROS 2 systems by offering nodes for publishing, subscribing, and handling services that can be connected in Action Graphs. It relies on `isaacsim.ros2.core` for the underlying ROS 2 functionality and integrates sensor data from camera and physics sensors into the ROS 2 ecosystem.

## Functionality

The extension registers OmniGraph nodes that handle ROS 2 message passing, services, and ros2_control setup. These nodes can be placed in Action Graphs to stream sensor data, receive commands, interact with ROS 2 topics and services, and configure an in-process ros2_control ControllerManager. The nodes inherit ROS distribution and core settings from `isaacsim.ros2.core`, which centralizes all ROS 2 Bridge configuration under `/exts/isaacsim.ros2.bridge/*`. This allows the nodes to automatically align with the configured ROS 2 environment at runtime without requiring separate configuration.

TF publishers aggregate compatible transform submissions by default so ROS 2 consumers receive one coherent transform snapshot per simulation tick. Publisher nodes that share the same ROS domain, topic, static or dynamic mode, and Quality of Service (QoS) profile contribute to one merged `tf2_msgs/msg/TFMessage` after OmniGraph evaluation instead of publishing one message per node. This avoids partial TF trees from multi-node graphs and reduces ROS 2 publisher and message overhead. Disable this behavior for debugging with `/exts/isaacsim.ros2.nodes/tfAggregation/enabled=false`.

## Integration

The extension depends on `isaacsim.ros2.core` for the ROS 2 backend, `isaacsim.ros2.control` for ControllerManager integration, `isaacsim.sensors.experimental.physics` for sensor data access, `isaacsim.sensors.physics.nodes` for physics sensor OmniGraph nodes, and `omni.graph` for the OmniGraph framework. It uses `isaacsim.core.experimental.utils` for stage and prim utilities, `isaacsim.core.experimental.objects` and `isaacsim.core.experimental.prims` for scene object creation in tests, and `isaacsim.core.rendering_manager` and `isaacsim.core.simulation_manager` for viewport and physics management. C++ publisher implementations for images and point clouds provide optimized data transfer for high-bandwidth sensor streams.
