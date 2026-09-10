# Overview

This extension provides shared utility functions for asset importers. It includes:

- **Collision from Visuals**: Apply collision APIs using visual geometry.
- **Mesh Merge Helpers**: Merge mesh groups via Scene Optimizer.
- **Self-Collision Utilities**: Enable articulation self-collision flags.
- **URDF/MJCF Conversion Helpers**: Convert joint and actuator attributes between URDF, MJCF, and PhysX.
- **PhysX to MuJoCo/Newton Conversion**: Author `MjcJointAPI`, `MjcActuator`, `NewtonMimicAPI`, and `NewtonArticulationRootAPI` on a PhysX asset (see `physx_asset_to_mjc`).
- **Asset Structure Profiles**: Run Asset Transformer profiles for packaging.

## Dependencies

This extension depends on:
- `isaacsim.asset.transformer`: Asset structure profile execution.
- `omni.scene.optimizer.core`: Mesh merge and scene optimization utilities.
- `omni.usd.schema.mujoco` and `omni.usd.schema.newton`: MuJoCo (`mjcPhysics`) and Newton schemas used by the PhysX to MuJoCo/Newton conversion.
