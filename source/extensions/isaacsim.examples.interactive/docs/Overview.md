# Overview

The isaacsim.examples.interactive extension provides interactive Isaac Sim examples for scene setup, robot teleoperation, OmniGraph input, and multi-robot coordination. Each example includes a programmatic sample class and an interactive user interface.

```{image} ../../../../source/extensions/isaacsim.examples.interactive/data/preview.png
---
align: center
---
```


## Key Components

### Input Device Integration

**{class}`KayaGamepad <isaacsim.examples.interactive.kaya_gamepad.KayaGamepad>`** provides gamepad control integration for the NVIDIA Kaya robot, demonstrating how to connect physical input devices with robotic systems in simulation for teleoperation scenarios.

**{class}`OmnigraphKeyboard <isaacsim.examples.interactive.omnigraph_keyboard.OmnigraphKeyboard>`** shows keyboard input processing through Omni Graph nodes, allowing users to control object properties (like cube size) using keyboard inputs, illustrating the integration between user input and graph-based programming.

### Multi-Robot Coordination

**{class}`RoboParty <isaacsim.examples.interactive.robo_party.RoboParty>`** demonstrates concurrent operation of multiple robot types including Franka manipulators, UR10 arms, Kaya holonomic robots, and Jetbot differential drive robots, each performing specialized tasks simultaneously.

### Educational and Template Examples

**{class}`HelloWorld <isaacsim.examples.interactive.hello_world.HelloWorld>`** serves as the foundational template demonstrating basic Isaac Sim sample structure and scene setup patterns.

**{class}`GettingStarted <isaacsim.examples.interactive.getting_started.GettingStarted>`** and **{class}`GettingStartedRobot <isaacsim.examples.interactive.getting_started.GettingStartedRobot>`** provide step-by-step tutorials for Isaac Sim fundamentals, covering scene creation, physics setup, robot integration, and basic simulation concepts.

## Functionality

Each example follows a consistent architecture pattern with scene setup, interactive UI controls, and proper cleanup handling. The examples support both programmatic access through their respective sample classes and interactive operation through integrated user interfaces.

Multi-robot examples showcase coordination strategies and workspace management.

## Integration

The extension integrates with the examples browser system for easy access. Examples register themselves with the browser system and provide consistent UI patterns through shared base classes.
