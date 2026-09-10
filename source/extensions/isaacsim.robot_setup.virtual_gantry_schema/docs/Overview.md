# Virtual Gantry Schema

Codeless USD schema defining the `IsaacVirtualGantry` prim type: an `Xform`
whose world transform is the anchor of a one-sided spring-damper "rope" that
suspends one articulation body.

Kept separate from `isaacsim.robot.schema` so that schema stays limited to
describing robots. The extension loads at `order = -100` — the schema tier —
so the prim type is registered before the extensions that depend on it.

The runtime that drives these prims is `isaacsim.robot_setup.virtual_gantry`;
the Create menu and property panel are in
`isaacsim.robot_setup.virtual_gantry.ui`.

```python
from isaacsim.robot_setup.virtual_gantry_schema import Attributes, CreateVirtualGantry

prim = CreateVirtualGantry(stage, "/World/VirtualGantry")
prim.GetAttribute(Attributes.GANTRY_STIFFNESS.name).Set(5000.0)
```
