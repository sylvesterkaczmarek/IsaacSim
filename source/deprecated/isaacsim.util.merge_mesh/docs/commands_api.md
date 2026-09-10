# Commands
Public command API for module **isaacsim.util.merge_mesh**:

- [MergeMeshesCommand](#mergemeshescommand)


## MergeMeshesCommand
Command to merge selected meshes and all children of given prims into a single mesh.

This command provides options to control the merge behavior:

- Clear Parent Transform: Sets the mesh origin at world origin, otherwise origin is
the same as the first element.
- Deactivate source assets: Sets source prims to Inactive after performing the merge operation.
- Combine Materials: Redirects all assets materials to a given folder, and every geomsubset
that shares a same material name uses the same material, each geom subset uses the original
material from the source assets.

### Arguments
- source: List of prim paths to merge.
- clear_transform: If True, sets the merged mesh origin at world origin.
- deactivate_source: If True, deactivates source prims after merging.
- combine_materials: If True, redirects all materials to a single folder
- materials_destination: The prim path where combined materials will be stored.

### Usage

```python
import omni.kit.commands

# Existing mesh prims, or parent prims that contain mesh children, to merge.
source_prims = ["/World/Cube", "/World/Cone", "/World/ManyToruses"]

# Merge the meshes into a single mesh.
# The merged mesh will be created under "/Merged/" using the first source prim's name.
success, merged_mesh_path = omni.kit.commands.execute(
    "MergeMeshesCommand",
    source=source_prims,
    clear_transform=False,        # Keep the merged mesh origin based on the first source prim
    deactivate_source=True,       # Deactivate the original source prims after merging
    combine_materials=True,       # Reuse materials with the same names in one destination folder
    materials_destination="/World/Looks",
)

print(f"Merged mesh created at: {merged_mesh_path}")
```

