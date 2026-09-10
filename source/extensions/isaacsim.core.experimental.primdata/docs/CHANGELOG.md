# Changelog

## [0.3.6] - 2026-07-20
### Changed
- Renamed C++ headers from `.h` to `.hpp`; update downstream include directives.

## [0.3.5] - 2026-07-15
### Changed
- Rename internal ``physx*`` fields and helpers to engine-neutral names.
- Create articulation and rigid-body tensor views for both PhysX and Newton (previously PhysX-only).
- Gate DOF field callbacks on ``maxDofs > 0`` so 0-DOF articulations still expose root transforms.

### Fixed
- Detect and recreate invalid simulation views, releasing stale tensor handles on physics reset.
- Leave ``dof_efforts`` unavailable on Newton since it does not expose ``getDofProjectedJointForces``, preventing a semantically different actuation force or valid zero from being reported as measured joint effort.
- Match ``getDofIndex`` by exact full path first, then fall back to leaf-name match.
- Extract DOF names from the tensor-reported path string directly.

## [0.3.4] - 2026-07-05
### Fixed
- Allocate PhysX rigid-body and articulation field buffers on the simulation view's resolved device when the reader is initialized with automatic device selection.

## [0.3.3] - 2026-05-21
### Fixed
- Release all reader-owned stage resources when the owning USD context begins closing to avoid retaining `UsdStage` references after close.
- Reinitialize the reader after stage-close invalidation and create PhysX simulation views lazily when PhysX-backed views are requested.

### Added
- Add C++ regression coverage for stage-close cleanup and reader reinitialization, including a fail pattern for unexpected USD stage reference count warnings.

## [0.3.2] - 2026-05-20
### Fixed
- Release the stage handle on stage close

## [0.3.1] - 2026-04-22
### Changed
- Query active physics engine at runtime via shared utility and pass it to `createSimulationView` instead of only supporting PhysX

## [0.3.0] - 2026-04-20
### Added
- Implement enableContactReporting() to apply PhysxContactReportAPI to rigid body prims
- Implement getContactReport() to query full contact report from IPhysxSimulation and filter by body paths
- Add SdfPathToken.hpp include and static_asserts verifying contact event type constants match PhysX enums
- Add m_contactEvents and m_contactPoints member vectors to PrimDataReaderImpl

## [0.2.0] - 2026-04-03
### Added
- Bulk PhysX tensor reading for world transforms of physics prims (rigid bodies and articulation links) when engine is PhysX, significantly improving performance over per-prim Fabric queries

### Changed
- Non-physics prims (xforms, cameras) continue to use Fabric/USDRT computeWorldXformNoCache as fallback

## [0.1.1] - 2026-03-24
### Changed
- Replaced Readme.md with Overview.md

## [0.1.0] - 2026-03-20
### Added
- Added the new `isaacsim.core.experimental.primdata` extension scaffold.
- Added a native plugin target for hosting `IPrimDataReader` provider implementations.
