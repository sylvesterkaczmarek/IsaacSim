# Overview

The isaacsim.robot_motion.pink.examples extension provides interactive demonstrations of the PINK inverse kinematics library within Isaac Sim on Linux x86_64. Each example is available from **Window > Examples > Robotics Examples** under **Motion Generation > PINK**, with controls for loading scenes, running scenarios, and observing IK-controlled robot behavior.

## Examples

### IK Controller

Demonstrates reactive end-effector tracking using {class}`PinkIKController <isaacsim.robot_motion.pink.PinkIKController>` with a Franka Panda robot following a movable target cube. This is the PINK analog to the cuMotion RMPflow example.

### Multi-Task

Demonstrates simultaneous weighted tasks: a FrameTask for end-effector tracking, a PostureTask for joint regularization toward a preferred pose, and a DampingTask for velocity smoothing. Shows how task weights affect the trade-off between tracking accuracy and motion smoothness.
