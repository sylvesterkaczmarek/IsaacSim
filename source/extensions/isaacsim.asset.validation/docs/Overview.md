# Overview

The isaacsim.asset.validation extension provisions the SimReady foundation validation tiers and the `simready-validate` framework inside Isaac Sim, so Isaac Sim content can be validated against the shared SimReady requirements.

The concrete validation rules for physics, joints, drives, robot schema, and materials that previously lived in this extension have been migrated to the SimReady foundation tiers and are now the single source of truth:

- **`simready-foundation-tier-core`**: general physics/geometry rules — rigid bodies, colliders, driven joints, articulation.
- **`simready-foundation-tier-isaac`**: Isaac-specific rules — robot core (naming, schema, physics layering) and robot materials.

## Functionality

The extension prebundles the two tier wheels and imports their capability hubs during startup. The tier decorators register their rules with the shared `usd_validation_nvidia` registry used by `omni.asset_validator.core`, making the rules available in the **Window > Asset Validator** interface and through `simready-validate`.
This extension registers `RGBSensorUsdRule` for authored RGB sensor USD assets.

Some rules (for example the non-adjacent collision-mesh check) run a live PhysX step and therefore rely on the Isaac/Kit physics runtime, which is why this extension keeps the physics and robot-schema Kit dependencies.
