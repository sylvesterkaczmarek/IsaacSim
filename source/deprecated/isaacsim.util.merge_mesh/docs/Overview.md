# Overview

```{deprecated} 6.0.0
This extension has been deprecated and replaced by the Scene Optimizer.
```

`**isaacsim.util.merge_mesh**` provides tools for combining multiple USD mesh prims into a single output mesh prim. It is useful when you want to reduce a selection of meshes into one mesh while preserving mesh data such as geometry, material assignments, geometry subsets, and texture coordinates.

The module supports both direct utility usage through {class}`MeshMerger <isaacsim.util.merge_mesh.MeshMerger>` and command-based usage through {class}`MergeMeshesCommand <isaacsim.util.merge_mesh.MergeMeshesCommand>`. The command form is the recommended entry point when you want the merge operation to participate in the Kit undo system.

## Concepts

### Mesh Selection

The merge operation starts from a list of selected prim paths. `MeshMerger.update_selection()` traverses the selected prims and their children, finds visible meshes, and computes merge statistics such as:

- Total meshes to merge
- Total geometry subsets
- Total unique materials

This allows callers or UI code to inspect what will be merged before running the operation.

### Output Mesh

The merged mesh is written to a single output prim path. When setting `MeshMerger.output_mesh`, the path is adjusted to avoid conflicts with existing prims.

The merge operation combines mesh geometry into this output mesh, including points, normals, texture coordinates, material assignments, and geometry subsets.

### Transform Handling

The merge can control where the merged mesh origin is placed.

- If `clear_parent_xform` or `clear_transform` is `True`, the merged mesh origin is placed at the world origin.
- If it is `False`, the merged mesh keeps an origin based on the source hierarchy behavior described by the command.

This is useful when deciding whether the merged result should preserve scene placement context or be normalized around the world origin.

### Source Prim Handling

The merge can optionally deactivate the original source prims after creating the merged mesh. This keeps the original data in the stage while hiding it from active scene evaluation.

`MergeMeshesCommand.undo()` can reactivate those sources when the command is undone.

### Material Combining

The merge can optionally combine materials into a single destination path, such as `/World/Looks`. When enabled, materials are redirected into that destination, and geometry subsets that share matching material names can use the same material.

Copied material connections can be fixed with `MeshMerger.fix_material_sources()` so shader output connections point to the new copied shader paths.

## Functionality

`**isaacsim.util.merge_mesh**` focuses on creating a single USD mesh from multiple input meshes while preserving the information needed for rendering and material assignment.

Key behavior includes:

- Traversing selected prims and child prims to find visible meshes
- Merging mesh geometry into one output mesh
- Preserving geometry subsets
- Preserving or combining material assignments
- Copying materials to a shared destination when requested
- Optionally deactivating source prims after merge
- Supporting undo through {class}`MergeMeshesCommand <isaacsim.util.merge_mesh.MergeMeshesCommand>`

## Key Components

### {class}`MeshMerger <isaacsim.util.merge_mesh.MeshMerger>`

{class}`MeshMerger <isaacsim.util.merge_mesh.MeshMerger>` is the lower-level utility class that performs selection analysis and mesh merging.

Create it with a USD stage:

```python
from isaacsim.util.merge_mesh import MeshMerger

mesh_merger = MeshMerger(stage)
```

Typical usage is to configure merge options, update the selection, then run the merge:

```python
from isaacsim.util.merge_mesh import MeshMerger

mesh_merger = MeshMerger(stage)

mesh_merger.clear_parent_xform = False
mesh_merger.deactivate_source = True
mesh_merger.combine_materials = True
mesh_merger.materials_destination = "/World/Looks"
mesh_merger.output_mesh = "/World/MergedMesh"

mesh_merger.update_selection([
    "/World/Cube",
    "/World/Cone",
    "/World/ManyToruses",
])

print(mesh_merger.total_meshes)
print(mesh_merger.total_subsets)
print(mesh_merger.total_materials)

mesh_merger.merge_meshes()
```

{class}`MeshMerger <isaacsim.util.merge_mesh.MeshMerger>` also provides helper methods used for undo-style cleanup:

- `reactivate_sources()` reactivates source prims that were deactivated during merge.
- `remove_created_materials()` removes materials created during the merge operation.

### {class}`MergeMeshesCommand <isaacsim.util.merge_mesh.MergeMeshesCommand>`

{class}`MergeMeshesCommand <isaacsim.util.merge_mesh.MergeMeshesCommand>` wraps the merge operation as a Kit command. Use this when you want the merge to be undoable through the command system.

```python
import omni.kit.commands

result, merged_prim_path = omni.kit.commands.execute(
    "MergeMeshesCommand",
    source=[
        "/World/Cube",
        "/World/Cone",
        "/World/ManyToruses",
    ],
    clear_transform=False,
    deactivate_source=True,
    combine_materials=True,
    materials_destination="/World/Looks",
)

print(merged_prim_path)
```

The command supports the same main merge controls:

- `source`: list of prim paths to merge
- `clear_transform`: place the merged mesh origin at the world origin
- `deactivate_source`: deactivate source prims after merging
- `combine_materials`: redirect materials into a shared destination
- `materials_destination`: destination prim path for combined materials

Undoing the command reactivates source prims if needed, removes the merged mesh, and removes materials created during the merge.

## Relationships

{class}`MergeMeshesCommand <isaacsim.util.merge_mesh.MergeMeshesCommand>` inherits from `**omni.kit.commands.Command**`, so it can be executed with `**omni.kit.commands.execute**()` and participate in the Kit undo flow. Use {class}`MeshMerger <isaacsim.util.merge_mesh.MeshMerger>` when you need direct programmatic control, and use {class}`MergeMeshesCommand <isaacsim.util.merge_mesh.MergeMeshesCommand>` when the merge should behave like an undoable application operation.
