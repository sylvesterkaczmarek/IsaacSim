# Changelog

## [Unreleased]

### Added
- Add C++ and Python wrappers for the most common USD objects: `Stage`, `Prim`, `Xform`, `Camera`, `Mesh`,
  geometry shapes (`Sphere`, `Cube`, `Capsule`, `Cone`, `Cylinder`, `Plane`),
  and lights (`SphereLight`, `DiskLight`, `RectLight`, `CylinderLight`, `DistantLight`, `DomeLight`).

### Fixed
- Avoid GCC 13 false-positive diagnostics when generating sphere meshes.
