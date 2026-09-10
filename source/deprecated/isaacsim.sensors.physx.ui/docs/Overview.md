# Overview

```{deprecated} 6.0.0
This extension is deprecated in favor of `isaacsim.sensors.physics.ui`.
```

`**isaacsim.sensors.physx.ui**` provides UI components for Isaac Sim PhysX-raycast-based sensor simulation. It focuses on user-facing controls such as menu options for working with physics-based sensor features, including sensor types such as lidar. The extension is intended for users who need access to PhysX sensor simulation controls through the application UI rather than through scripting.

## Functionality

The extension adds UI-facing functionality around PhysX sensor simulation workflows. Its main role is to expose sensor-related actions through menus and UI components so users can access common sensor operations from the interface.

The functionality is centered on:

- PhysX-raycast-based sensor simulation controls.
- UI menu options for sensor-related workflows.
- Support for Isaac Sim sensor workflows, including lidar-related use cases.

## UI Components

### Sensor menu options

The extension provides menu-oriented UI components for PhysX sensor simulation. These menu options give users access to sensor functionality without needing to call Python APIs directly.

The menu entries are intended to connect the user interface with the underlying PhysX sensor simulation extension.

The extension is focused on exposing controls through the UI. It does not define a separate standalone window or custom scripting interface.

## Relationships

`**isaacsim.sensors.physx.ui**` is built around the PhysX sensor simulation functionality provided by `**isaacsim.sensors.physx**`. The UI extension contributes menu and interface components, while the underlying sensor extension provides the simulation behavior for PhysX-raycast-based sensors.
