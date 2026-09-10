# Overview

```{deprecated} 6.0.0
This extension is deprecated in favor of the Core Experimental extensions: `isaacsim.core.experimental.*`.
```

The `isaacsim.core.api` extension provides the legacy Isaac Sim core layer for controlling simulation state and physics scenes. It also supports working with USD objects, physics materials, and visual materials in Isaac Sim workflows.

New work should use the Core Experimental extensions under `isaacsim.core.experimental.*`. Use `isaacsim.core.api` only when maintaining existing workflows that still depend on the older core behavior.

## Key Components

### Simulation control

- {class}`World <isaacsim.core.api.World>` is the main entry point for most workflows. It extends {class}`SimulationContext <isaacsim.core.api.SimulationContext>` with a managed {class}`Scene <isaacsim.core.api.scenes.Scene>`, task registration, observations, metrics, and a data logger.
- {class}`SimulationContext <isaacsim.core.api.SimulationContext>` controls the simulation lifecycle (play, pause, stop, step, reset) and manages physics, stage, timeline, and render callbacks.
- {class}`PhysicsContext <isaacsim.core.api.PhysicsContext>` wraps the USD physics scene and exposes solver, GPU, gravity, and timestep settings.

```python
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World()
world.scene.add_default_ground_plane()
cube = world.scene.add(DynamicCuboid(prim_path="/World/Cube", name="cube"))

world.reset()
for _ in range(100):
    world.step(render=True)
```

### Scene helpers

The extension groups higher-level building blocks into submodules:

- `isaacsim.core.api.objects`: visual, fixed, and dynamic primitive shapes such as {class}`DynamicCuboid <isaacsim.core.api.objects.DynamicCuboid>`, `VisualSphere`, `FixedCylinder`, `DynamicCone`, `VisualCapsule`, and {class}`GroundPlane <isaacsim.core.api.objects.GroundPlane>`.
- `isaacsim.core.api.materials`: visual and physics material wrappers including {class}`VisualMaterial <isaacsim.core.api.materials.VisualMaterial>`, `PreviewSurface`, `OmniPBR`, `OmniGlass`, {class}`PhysicsMaterial <isaacsim.core.api.materials.PhysicsMaterial>`, `ParticleMaterial`, and `DeformableMaterial` (with matching view classes).
- `isaacsim.core.api.robots`: {class}`Robot <isaacsim.core.api.robots.Robot>` and {class}`RobotView <isaacsim.core.api.robots.RobotView>`.
- `isaacsim.core.api.controllers`: {class}`BaseController <isaacsim.core.api.controllers.BaseController>`, {class}`ArticulationController <isaacsim.core.api.controllers.ArticulationController>`, and {class}`BaseGripperController <isaacsim.core.api.controllers.BaseGripperController>`.
- `isaacsim.core.api.tasks`: {class}`BaseTask <isaacsim.core.api.tasks.BaseTask>` and ready-made tasks such as `FollowTarget`, `PickPlace`, and `Stacking`.
- `isaacsim.core.api.scenes`: {class}`Scene <isaacsim.core.api.scenes.Scene>` and {class}`SceneRegistry <isaacsim.core.api.scenes.SceneRegistry>`.
- `isaacsim.core.api.sensors`: {class}`BaseSensor <isaacsim.core.api.sensors.BaseSensor>` and {class}`RigidContactView <isaacsim.core.api.sensors.RigidContactView>`.
- `isaacsim.core.api.loggers`: {class}`DataLogger <isaacsim.core.api.loggers.DataLogger>` for recording simulation data.

Prim-level state wrappers used by these helpers (rigid bodies, articulations, geometry, and so on) live in the `isaacsim.core.prims` extension.
