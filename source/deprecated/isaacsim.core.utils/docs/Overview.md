# Overview

```{deprecated} 6.0.0
This extension is deprecated in favor of `isaacsim.core.experimental.utils`.
```

The `isaacsim.core.utils` extension provides shared utility functions used across Isaac Sim workflows. It collects helpers for common USD, stage, prim, physics, math, transform, rotation, mesh, bounds, semantics, rendering, and carb tasks so that extension authors and tools can rely on consistent behavior across the simulation stack.

The extension is a utility layer rather than a standalone tool or UI. Its modules are typically imported directly, for example `from isaacsim.core.utils.stage import add_reference_to_stage`.

## Key Components

### Utility submodules

Utilities are grouped by domain and accessed as `isaacsim.core.utils.<name>`:

- **USD and stage**: `stage`, `prims`, `semantics`, `xforms`, `fabric`, `render_product`
- **Math and transforms**: `math`, `rotations`, `transformations`, `distance_metrics`, and backend-specific helpers under `numpy`, `torch`, and `warp`
- **Physics and geometry**: `physics`, `articulations`, `bounds`, `collisions`, `mesh`, `deformable_mesh_utils`
- **Application and data helpers**: `carb`, `extensions`, `string`, `constants`, `types`, `viewports`, `camera_utils`, `interops`, `random`

```python
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.prims import get_prim_at_path

add_reference_to_stage(usd_path="/path/to/asset.usd", prim_path="/World/Asset")
stage = get_current_stage()
prim = get_prim_at_path("/World/Asset")
```

### Commands

The `commands` submodule registers `omni.kit` commands for common prim operations, including {class}`IsaacSimSpawnPrim <isaacsim.core.utils.commands.IsaacSimSpawnPrim>`, {class}`IsaacSimTeleportPrim <isaacsim.core.utils.commands.IsaacSimTeleportPrim>`, {class}`IsaacSimScalePrim <isaacsim.core.utils.commands.IsaacSimScalePrim>`, and {class}`IsaacSimDestroyPrim <isaacsim.core.utils.commands.IsaacSimDestroyPrim>`.

### C++ interface

The extension includes a Carbonite plugin and pybind11 bindings exposed as `isaacsim.core.utils.bindings._isaac_utils`. The bindings provide a `math` submodule with vector and quaternion operations (such as `dot`, `cross`, `normalize`, `lerp`, and `slerp`) and a `transforms` submodule (`set_transform`, `set_scale`), and back the prim-path matching helper used by the Python utilities.

## Relationships

`isaacsim.core.utils` is a shared utility dependency for simulation-oriented extensions. It declares dependencies on USD and physics systems, including PhysX, USD physics schemas, semantic schema support, and USD metrics assembly, which align with the USD and physics utilities it provides.
