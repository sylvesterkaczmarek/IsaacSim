# Commands
Public command API for module **isaacsim.core.utils**:

- [IsaacSimDestroyPrim](#isaacsimdestroyprim)
- [IsaacSimScalePrim](#isaacsimscaleprim)
- [IsaacSimSpawnPrim](#isaacsimspawnprim)
- [IsaacSimTeleportPrim](#isaacsimteleportprim)


## IsaacSimDestroyPrim
Command to delete a prim. This variant has less overhead than other commands as it doesn't store an undo operation.

Typical usage example:

.. code-block:: python

omni.kit.commands.execute(
"IsaacSimDestroyPrim",
prim_path="/World/Prim",
)

### Arguments
- prim_path: Path to the prim to delete.

### Usage

```python
import omni.kit.commands
import omni.usd
from pxr import UsdGeom

# Get the current USD stage
stage = omni.usd.get_context().get_stage()

# Create a prim to delete
prim_path = "/World/PrimToDelete"
UsdGeom.Xform.Define(stage, prim_path)

# Delete the prim using the IsaacSimDestroyPrim command
omni.kit.commands.execute(
    "IsaacSimDestroyPrim",
    prim_path=prim_path,
)
```

## IsaacSimScalePrim
Command to set a scale of a prim.

Typical usage example:

.. code-block:: python

omni.kit.commands.execute(
"IsaacSimScalePrim",
prim_path="/World/Prim",
scale=(1.5, 1.5, 1.5),
)

### Arguments
- prim_path: Path to the prim to scale.
- scale: Scale values for x, y, and z axes.

### Usage

```python
import omni.kit.commands

# Scale an existing prim at this path.
prim_path = "/World/Prim"

# Set the prim scale along the X, Y, and Z axes.
scale = (1.5, 1.5, 1.5)

omni.kit.commands.execute(
    "IsaacSimScalePrim",
    prim_path=prim_path,
    scale=scale,
)
```

## IsaacSimSpawnPrim
Command to spawn a new prim in the stage and set its transform. This uses dynamic_control to properly handle physics objects and articulation.

Typical usage example:

.. code-block:: python

omni.kit.commands.execute(
"IsaacSimSpawnPrim",
usd_path="/path/to/file.usd",
prim_path="/World/Prim",
translation=(0, 0, 0),
rotation=(0, 0, 0, 1),
)

### Arguments
- usd_path: Path to the USD file to reference.
- prim_path: Path where the prim will be created in the stage.
- translation: Translation vector for the prim's position.
- rotation: Rotation quaternion for the prim's orientation.

### Usage

```python
import omni.kit.commands

# Path to the USD file to spawn.
# Replace this with a valid local path or Omniverse/Nucleus USD path.
usd_path = "/path/to/asset.usd"

# Spawn the USD asset at /World/SpawnedPrim and set its transform.
omni.kit.commands.execute(
    "IsaacSimSpawnPrim",
    usd_path=usd_path,
    prim_path="/World/SpawnedPrim",
    translation=(0.0, 0.0, 0.5),  # position: (x, y, z)
    rotation=(0.0, 0.0, 0.0, 1.0),  # quaternion: (x, y, z, w)
)
```

## IsaacSimTeleportPrim
Command to set a transform of a prim. This uses dynamic_control to properly handle physics objects and articulation.

Typical usage example:

.. code-block:: python

omni.kit.commands.execute(
"IsaacSimTeleportPrim",
prim_path="/World/Prim",
translation=(0, 0, 0),
rotation=(0, 0, 0, 1),
)

### Arguments
- prim_path: Path to the prim to teleport.
- translation: Translation vector as (x, y, z).
- rotation: Rotation quaternion as (x, y, z, w).

### Usage

```python
import omni.kit.commands

# Teleport an existing prim to a new world transform.
# translation is (x, y, z)
# rotation is a quaternion (x, y, z, w)
omni.kit.commands.execute(
    "IsaacSimTeleportPrim",
    prim_path="/World/Prim",
    translation=(1.0, 0.0, 0.5),
    rotation=(0.0, 0.0, 0.0, 1.0),
)
```

