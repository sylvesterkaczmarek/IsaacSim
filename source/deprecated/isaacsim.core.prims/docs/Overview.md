# Overview

```{deprecated} 6.0.0
This extension is deprecated in favor of `isaacsim.core.experimental.prims`.
```

The `isaacsim.core.prims` extension provides high-level Python wrappers for reading and writing state on USD prims used in simulation. It gives you a consistent way to work with transforms, geometry, rigid bodies, articulations, SDF shapes, and particle systems without directly managing every USD or PhysX schema detail.

The module supports both single-prim wrappers, such as {class}`SingleRigidPrim <isaacsim.core.prims.SingleRigidPrim>`, and view-style wrappers, such as {class}`RigidPrim <isaacsim.core.prims.RigidPrim>`, that operate on one or more prims matched by a path expression. This makes it useful for both simple scene objects and batched simulation workflows with cloned environments.

The public classes form the following inheritance relationships:

<div align="center">

```{mermaid}
graph TD
    %% Inheritance relationships
    XFormPrim --> GeometryPrim
    XFormPrim --> Articulation
    XFormPrim --> RigidPrim
    GeometryPrim --> SdfShapePrim
    _SinglePrimWrapper --> SingleXFormPrim
    _SinglePrimWrapper --> SingleGeometryPrim
    _SinglePrimWrapper --> SingleArticulation
    _SinglePrimWrapper --> SingleRigidPrim
```

</div>

## Concepts

### Single Prim vs Prim View

The module uses two common access patterns:

- {class}`SingleXFormPrim <isaacsim.core.prims.SingleXFormPrim>`, {class}`SingleGeometryPrim <isaacsim.core.prims.SingleGeometryPrim>`, {class}`SingleRigidPrim <isaacsim.core.prims.SingleRigidPrim>`, {class}`SingleArticulation <isaacsim.core.prims.SingleArticulation>`, and {class}`SingleParticleSystem <isaacsim.core.prims.SingleParticleSystem>` wrap one prim at a specific `prim_path`.
- {class}`XFormPrim <isaacsim.core.prims.XFormPrim>`, {class}`GeometryPrim <isaacsim.core.prims.GeometryPrim>`, {class}`RigidPrim <isaacsim.core.prims.RigidPrim>`, {class}`Articulation <isaacsim.core.prims.Articulation>`, {class}`SdfShapePrim <isaacsim.core.prims.SdfShapePrim>`, and {class}`ParticleSystem <isaacsim.core.prims.ParticleSystem>` wrap one or more prims using `prim_paths_expr`.

View classes are designed for batched operations. Most methods accept `indices` so you can read or update only part of the wrapped prim set.

```python
from isaacsim.core.prims import RigidPrim

rigids = RigidPrim(
    prim_paths_expr="/World/envs/env.*/Cube",
    name="cube_view",
)

rigids.initialize()

# Move only selected rigid bodies
rigids.set_world_poses(
    positions=[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
    indices=[0, 2],
)
```

### Path Expressions

Most view classes use `prim_paths_expr` to match prims in the stage. A normal prim path can wrap one prim, while a regex-style expression can wrap many cloned prims.

For example, `"/World/envs/env.*/panda"` can match several robot instances under cloned environments. Some classes also accept a list of path expressions.

### Initialization

Physics-backed classes such as {class}`RigidPrim <isaacsim.core.prims.RigidPrim>`, {class}`GeometryPrim <isaacsim.core.prims.GeometryPrim>`, {class}`Articulation <isaacsim.core.prims.Articulation>`, {class}`SdfShapePrim <isaacsim.core.prims.SdfShapePrim>`, and {class}`ParticleSystem <isaacsim.core.prims.ParticleSystem>` must be initialized before methods that depend on the PhysX tensor API can be used. Their `initialize()` methods accept an optional `omni.physics.tensors.SimulationView`.

If the object is added to a scene that handles reset, initialization may happen during reset. After a hard reset, such as stopping and playing the timeline, these objects may need to be initialized again before physics methods are used.

### Data Shapes

Most batched APIs use array-like inputs with shapes based on the number of wrapped prims. The module commonly accepts `numpy.ndarray`, `torch.Tensor`, and `warp.array` where supported.

Common conventions include:

- Positions: `(N, 3)`
- Orientations: `(N, 4)`, scalar-first quaternion `(w, x, y, z)`
- Linear or angular velocities: `(N, 3)`
- Combined linear and angular velocities: `(N, 6)`
- Joint values for articulations: `(N, num_dof)`

## Functionality

### Transform State

{class}`XFormPrim <isaacsim.core.prims.XFormPrim>` and {class}`SingleXFormPrim <isaacsim.core.prims.SingleXFormPrim>` provide the base transform workflow. They manage world poses, local poses, scale, visibility, default state, and visual materials.

{class}`XFormPrim <isaacsim.core.prims.XFormPrim>` can read and write poses through USD, and some methods expose a `usd` flag for choosing USD or Fabric-backed data access where supported.

```python
from isaacsim.core.prims import XFormPrim
import numpy as np

prims = XFormPrim("/World/envs/env.*", name="xforms")

positions = np.zeros((5, 3))
positions[:, 0] = np.arange(5)

orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (5, 1))
prims.set_world_poses(positions=positions, orientations=orientations)

current_positions, current_orientations = prims.get_world_poses()
```

### Geometry and Collision

{class}`GeometryPrim <isaacsim.core.prims.GeometryPrim>` and {class}`SingleGeometryPrim <isaacsim.core.prims.SingleGeometryPrim>` extend transform handling with geometry and collision-related properties. They can enable or disable collision, apply collision APIs, configure collision approximation, and apply physics materials.

They also support contact-force reporting when contact tracking is configured during construction.

```python
from isaacsim.core.prims import GeometryPrim

geoms = GeometryPrim(
    prim_paths_expr="/World/envs/env.*/Obstacle",
    collisions=[True] * 5,
    track_contact_forces=True,
)

geoms.initialize()

geoms.set_collision_approximations(["convexHull"] * 5)
forces = geoms.get_net_contact_forces()
```

{class}`SingleGeometryPrim <isaacsim.core.prims.SingleGeometryPrim>` exposes the same ideas for one prim, using scalar methods such as `set_contact_offset()`, `set_collision_enabled()`, and `apply_physics_material()`.

### Rigid Bodies

{class}`RigidPrim <isaacsim.core.prims.RigidPrim>` and {class}`SingleRigidPrim <isaacsim.core.prims.SingleRigidPrim>` wrap prims with the Rigid Body API. If the Rigid Body API is not already applied, the wrapper applies it during initialization.

Rigid body APIs cover:

- World and local poses
- Linear and angular velocities
- Forces and torques
- Mass, density, inertia, and center of mass
- Gravity and rigid body physics toggles
- Sleep thresholds
- Default and current dynamic state
- Contact and friction data when configured

```python
from isaacsim.core.prims import RigidPrim
import numpy as np

rigids = RigidPrim("/World/envs/env.*/Box", name="box_view")
rigids.initialize()

# Set velocity for every wrapped rigid body
velocities = np.zeros((rigids.count, 6))
velocities[:, 0] = 1.0
rigids.set_velocities(velocities)

# Apply force to selected bodies
rigids.apply_forces(
    forces=np.tile(np.array([0.0, 0.0, 100.0]), (2, 1)),
    indices=np.array([0, 3]),
)
```

### Articulations

{class}`Articulation <isaacsim.core.prims.Articulation>` and {class}`SingleArticulation <isaacsim.core.prims.SingleArticulation>` wrap prims that have the Root {class}`Articulation <isaacsim.core.prims.Articulation>` API applied. They are the main interfaces for robot-like structures with joints, links, drives, and articulated dynamics.

They provide APIs for:

- DOF, joint, body, and link metadata
- Joint position, velocity, and effort state
- Joint position and velocity targets
- `ArticulationActions` or `ArticulationAction` based control
- PD gains, max efforts, max joint velocities, and effort modes
- Solver iteration counts, stabilization, sleep thresholds, and self-collision flags
- Jacobians, mass matrices, gravity forces, and Coriolis and centrifugal forces
- Body mass, inertia, center of mass, and gravity settings
- Fixed tendon properties for articulations that use fixed tendons

```python
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
import numpy as np

robot = SingleArticulation("/World/envs/env_0/panda", name="panda")
robot.initialize()

# Move all joints using an articulation action
action = ArticulationAction(
    joint_positions=np.array([0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8, 0.04, 0.04])
)
robot.apply_action(action)

# Query joint state
state = robot.get_joints_state()
positions = state.positions
velocities = state.velocities
```

For batched robots, use {class}`Articulation <isaacsim.core.prims.Articulation>` with a path expression and optional `indices` or `joint_indices` to target specific articulations and joints.

### SDF Shapes

{class}`SdfShapePrim <isaacsim.core.prims.SdfShapePrim>` extends {class}`GeometryPrim <isaacsim.core.prims.GeometryPrim>` for mesh geometry prims that provide a Signed Distance Field. It creates a SDF shape view and can query SDF values and gradients for local-space points.

It also exposes SDF collision settings such as margin, narrow band thickness, subgrid resolution, and resolution.

```python
from isaacsim.core.prims import SdfShapePrim
import numpy as np

sdf_view = SdfShapePrim(
    prim_paths_expr="/World/envs/env.*/Mesh",
    num_query_points=16,
)

sdf_view.initialize()

points = np.zeros((sdf_view.num_shapes, sdf_view.num_query_points, 3))
sdf_and_gradients = sdf_view.get_sdf_and_gradients(points)
```

The returned SDF query data stores the SDF value in the first component and the gradient in the last three components.

### Particle Systems

{class}`ParticleSystem <isaacsim.core.prims.ParticleSystem>` and {class}`SingleParticleSystem <isaacsim.core.prims.SingleParticleSystem>` wrap PhysX particle systems. They provide access to particle system solver and collision parameters, including contact offsets, rest offsets, CCD, wind, max velocity, neighborhood size, and self-collision settings.

Particle systems use GPU-accelerated position-based dynamics. CPU simulation of particles is not supported, and particle system solver parameters cannot be changed once the scene is playing.

```python
from isaacsim.core.prims import SingleParticleSystem

particles = SingleParticleSystem(
    prim_path="/World/ParticleSystem",
    particle_contact_offset=0.05,
    solid_rest_offset=0.025,
    fluid_rest_offset=0.025,
    enable_ccd=True,
)

particles.set_wind([1.0, 0.0, 0.0])
particles.set_max_velocity(10.0)
```

{class}`ParticleSystem <isaacsim.core.prims.ParticleSystem>` provides the same style of operations for multiple particle systems matched by `prim_paths_expr`.

## Key Components

### {class}`XFormPrim <isaacsim.core.prims.XFormPrim>`

Base view for transformable prims. Use it when you need batched pose, scale, visibility, default-state, or visual-material operations without requiring rigid-body or articulation behavior.

### {class}`GeometryPrim <isaacsim.core.prims.GeometryPrim>`

Geometry view for one or more geometry prims. Use it for collision setup, collision approximation, contact offsets, physics materials, and contact-force reporting on geometry.

### {class}`RigidPrim <isaacsim.core.prims.RigidPrim>`

Rigid body view for one or more rigid prims. Use it for physics state, velocities, forces, masses, inertias, gravity, and contact data.

### {class}`Articulation <isaacsim.core.prims.Articulation>`

{class}`Articulation <isaacsim.core.prims.Articulation>` view for one or more articulated prims. Use it for batched robot state, joint control, dynamics queries, solver settings, and body-level articulation properties.

### {class}`SdfShapePrim <isaacsim.core.prims.SdfShapePrim>`

Specialized geometry view for querying Signed Distance Field values and gradients from mesh geometry prims.

### {class}`ParticleSystem <isaacsim.core.prims.ParticleSystem>`

View for one or more PhysX particle systems. Use it to configure particle solver, collision, material, and wind parameters.

### Single-Prim Wrappers

{class}`SingleXFormPrim <isaacsim.core.prims.SingleXFormPrim>`, {class}`SingleGeometryPrim <isaacsim.core.prims.SingleGeometryPrim>`, {class}`SingleRigidPrim <isaacsim.core.prims.SingleRigidPrim>`, {class}`SingleArticulation <isaacsim.core.prims.SingleArticulation>`, and {class}`SingleParticleSystem <isaacsim.core.prims.SingleParticleSystem>` provide scalar-style access to one prim. They are convenient when you are working with a single object or robot and do not need batched indexing.

## Usage Examples

### Wrap Cloned Rigid Bodies

```python
from isaacsim.core.prims import RigidPrim
import numpy as np

rigids = RigidPrim(
    prim_paths_expr="/World/envs/env.*/Cube",
    name="cube_view",
    masses=np.full(5, 1.0),
)

rigids.initialize()

positions, orientations = rigids.get_world_poses()
rigids.set_linear_velocities(np.zeros((5, 3)))
```

### Control Multiple Articulations

```python
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationActions
import numpy as np

robots = Articulation("/World/envs/env.*/panda", name="panda_view")
robots.initialize()

joint_positions = np.zeros((robots.count, robots.num_dof))
actions = ArticulationActions(joint_positions=joint_positions)

robots.apply_action(actions)
```

### Configure Collision on Geometry

```python
from isaacsim.core.prims import SingleGeometryPrim

geom = SingleGeometryPrim(
    prim_path="/World/Obstacle",
    collision=True,
)

geom.initialize()
geom.set_collision_approximation("convexHull")
geom.set_contact_offset(0.02)
geom.set_rest_offset(0.01)
```

## Relationships

Physics-backed classes use `omni.physics.tensors.SimulationView` in their `initialize()` methods to create PhysX tensor views. Several APIs also use state containers from `isaacsim.core.utils.types`, such as `DynamicState`, `DynamicsViewState`, `JointsState`, `ArticulationAction`, and `ArticulationActions`.

Material-related methods use material classes from `isaacsim.core.api.materials`, including visual materials, `PhysicsMaterial`, and `ParticleMaterial`. The wrappers also expose USD and PhysX schema objects where relevant, such as `UsdGeom.Gprim` for geometry and `PhysxSchema.PhysxParticleSystem` for particle systems.
