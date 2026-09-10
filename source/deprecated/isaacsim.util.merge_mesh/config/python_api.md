# Public API for module isaacsim.util.merge_mesh:

## Classes

- class MeshMerger(object)
  - def __init__(self, stage: object)
  - [property] def total_meshes(self) -> int
  - [property] def total_subsets(self) -> int
  - [property] def total_materials(self) -> int
  - [property] def meshes_to_merge(self) -> list
  - [property] def clear_parent_xform(self) -> bool
  - [clear_parent_xform.setter] def clear_parent_xform(self, value: bool)
  - [property] def deactivate_source(self) -> bool
  - [deactivate_source.setter] def deactivate_source(self, value: bool)
  - [property] def selected_objects(self) -> list
  - [property] def combine_materials(self) -> bool
  - [combine_materials.setter] def combine_materials(self, value: bool)
  - [property] def materials_destination(self) -> str
  - [materials_destination.setter] def materials_destination(self, value: str)
  - [property] def output_mesh(self) -> str
  - [output_mesh.setter] def output_mesh(self, value: str)
  - def fix_material_sources(self, mat: object)
  - def update_selection(self, selection: list, stage: object = None)
  - def merge_meshes(self)
  - def reactivate_sources(self)
  - def remove_created_materials(self)

- class MergeMeshesCommand(omni.kit.commands.Command)
  - def __init__(self, source: str, clear_transform: bool = False, deactivate_source: bool = False, combine_materials: bool = False, materials_destination: str = '/World/Looks')
  - def do(self) -> str
  - def undo(self)
