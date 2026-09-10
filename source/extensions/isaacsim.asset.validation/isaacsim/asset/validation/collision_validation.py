# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime-dependent SimReady validators hosted by Isaac Sim.

The BA_002 non-adjacent collision-mesh check lives here rather than in the
SimReady ``simready-foundation-tier-core`` wheel because it requires a *live*
PhysX simulation step (carb, usdrt, omni.physics, omni.physx) to collect the
initial contact pairs. Those runtimes only exist inside the Isaac Sim / Kit
process, so the core tier deliberately omits the checker to avoid shipping a
rule that would silently pass in a standalone USD-only environment.

The rule is still registered against the core tier's ``BA_002`` requirement
(``simready.foundation.tier_core.requirements``) using the SimReady validation
framework (``usd_validation_nvidia``). Importing this module registers the rule
with the in-process ``usd_validation_nvidia`` registry; the Isaac Sim extension
imports it on startup so ``simready-validate`` discovers it while running inside
Kit.
"""

from typing import Any

import simready.foundation.tier_core.requirements as cap
import usd_validation_nvidia
from pxr import Sdf, Usd, UsdPhysics

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None

# Live PhysX contact detection needs the Isaac/Kit physics runtime. These are
# always present inside Isaac Sim; guard them so that importing this module in a
# non-Kit context (e.g. static analysis, docs build) does not hard-fail.
try:
    import carb
    import usdrt
    from omni.physics.core import ContactEventType, get_physics_simulation_interface
    from omni.physx.bindings._physx import SETTING_UPDATE_TO_USD
    from pxr import PhysicsSchemaTools, UsdUtils

    _PHYSICS_RUNTIME_AVAILABLE = True
except Exception:  # pragma: no cover - depends on runtime availability
    _PHYSICS_RUNTIME_AVAILABLE = False


def get_initial_collider_pairs(stage: Usd.Stage) -> set[tuple[str, str]]:
    """Get all collider pairs that are in contact in the physics simulation.

    This function performs a single physics simulation step and collects all collider pairs
    that are in contact. It temporarily modifies physics settings to ensure accurate contact
    detection and restores them after completion.

    The function:
    1. Creates a temporary session layer for contact reporting
    2. Enables contact reporting for all rigid bodies
    3. Runs a single physics simulation step
    4. Collects all collider pairs that are in contact
    5. Restores original physics settings

    Args:
        stage: The USD stage containing the physics scene to analyze.

    Returns:
        A set of tuples, where each tuple contains the paths of two colliders that are
            in contact. The paths in each tuple are sorted alphabetically to ensure
            consistent ordering regardless of which collider initiated the contact.

    Note:
        This function temporarily modifies physics settings and runs a simulation step.
        The original settings are restored after the function completes.
    """
    unique_collider_pairs = set()  # Use a set to store unique collider pairs

    # Live contact detection requires the Isaac/Kit physics runtime. When it is
    # not importable we cannot run a simulation step, so report no pairs.
    if not _PHYSICS_RUNTIME_AVAILABLE or PhysxSchema is None:
        return unique_collider_pairs

    def on_contact_event(contact_headers: Any, contact_data: Any, friction_anchors: Any) -> None:
        for contact_header in contact_headers:
            if contact_header.type == ContactEventType.CONTACT_FOUND:
                collider0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0))
                collider1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider1))
                # Store as a tuple, ensuring consistent ordering
                pair = tuple(sorted([collider0, collider1]))
                unique_collider_pairs.add(pair)

    session_sub_layer = Sdf.Layer.CreateAnonymous()
    stage.GetSessionLayer().subLayerPaths.append(session_sub_layer.identifier)
    old_layer = stage.GetEditTarget().GetLayer()
    stage.SetEditTarget(Usd.EditTarget(session_sub_layer))

    # Added this to avoid stage not in cache error
    stageCache = UsdUtils.StageCache.Get()
    stageCache.Insert(stage)  # Register the stage

    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    usdrtStage = usdrt.Usd.Stage.Attach(stage_id)
    prim_paths = usdrtStage.GetPrimsWithAppliedAPIName("PhysicsRigidBodyAPI")

    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(str(prim_path))
        if prim:
            contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            contact_report_api.CreateThresholdAttr().Set(0)

    settings = carb.settings.get_settings()
    write_usd = settings.get_as_bool(SETTING_UPDATE_TO_USD)
    write_fabric = settings.get_as_bool("/physics/fabricEnabled")

    settings.set(SETTING_UPDATE_TO_USD, False)
    settings.set("/physics/fabricEnabled", False)

    initial_attach = False
    if get_physics_simulation_interface().get_attached_stage() != stage_id:
        get_physics_simulation_interface().initialize(stage_id)
        initial_attach = True

    contact_report_sub = get_physics_simulation_interface().subscribe_physics_contact_report_events(on_contact_event)

    get_physics_simulation_interface().simulate(1.0 / 60.0, 0.0)

    if contact_report_sub:
        contact_report_sub = None

    if initial_attach:
        get_physics_simulation_interface().close()

    settings.set(SETTING_UPDATE_TO_USD, write_usd)
    settings.set("/physics/fabricEnabled", write_fabric)

    stage.SetEditTarget(old_layer)

    stage.GetSessionLayer().subLayerPaths.remove(session_sub_layer.identifier)
    session_sub_layer = None

    return unique_collider_pairs


def compute_adjacent_mesh_dict(stage: Usd.Stage) -> dict:
    """Compute a dictionary mapping body paths to lists of adjacent body paths.

    Args:
        stage: The USD stage to analyze.

    Returns:
        A dictionary mapping body paths to lists of adjacent body paths.
    """
    # Traverse through the joints, log every pair of connected bodies
    defaultPrim = stage.GetDefaultPrim()
    if not defaultPrim or not defaultPrim.IsValid():
        return {}

    if PhysxSchema is None:
        raise RuntimeError("PhysxSchema is not available in this environment")

    adjacent_mesh_matrix = {}

    for prim in stage.Traverse():
        if prim.HasAPI(PhysxSchema.PhysxJointAPI):
            joint = UsdPhysics.Joint(prim)
            body0_targets = joint.GetBody0Rel().GetTargets()
            if not body0_targets:
                continue
            body0 = body0_targets[0]
            body1_targets = joint.GetBody1Rel().GetTargets()
            if not body1_targets:
                continue
            body1 = body1_targets[0]

            # body0 and body1 are adjacent, log into joint dict
            if body0 not in adjacent_mesh_matrix:
                adjacent_mesh_matrix[body0] = []
            if body1 not in adjacent_mesh_matrix:
                adjacent_mesh_matrix[body1] = []
            adjacent_mesh_matrix[body0].append(body1)
            adjacent_mesh_matrix[body1].append(body0)

    return adjacent_mesh_matrix


def _find_rigid_body_ancestor(prim: Usd.Prim) -> Sdf.Path:
    """Walk up to the nearest ancestor (inclusive) carrying UsdPhysics.RigidBodyAPI.

    Returns ``Sdf.Path.emptyPath`` if no such ancestor exists. Used by
    :class:`NonAdjacentCollisionMeshesDoNotClash` to key adjacency-dict lookups
    on the rigid body that owns each collider, rather than the collider's direct
    parent -- which fails whenever collision meshes live nested under a link
    (e.g. ``/Robot/link0/collisions/mesh_0``).

    Args:
        prim: The prim (typically a collider) to search upward from.

    Returns:
        Path of the nearest ancestor (or ``prim`` itself) with
        ``UsdPhysics.RigidBodyAPI`` applied, or ``Sdf.Path.emptyPath`` if none.
    """
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current.GetPath()
        current = current.GetParent()
    return Sdf.Path.emptyPath


@usd_validation_nvidia.register_rule("BaseArticulation")
@usd_validation_nvidia.register_requirements(cap.BaseArticulationRequirements.BA_002, override=True)
class NonAdjacentCollisionMeshesDoNotClash(usd_validation_nvidia.BaseRuleChecker):
    """Validates that non-adjacent collision meshes don't intersect.

    This rule checks that collision meshes that aren't connected by joints don't
    intersect each other, which can cause unstable physics simulation. It relies
    on a live PhysX simulation step (see :func:`get_initial_collider_pairs`) and
    therefore only works inside the Isaac Sim / Kit runtime, which is why it is
    hosted by the ``isaacsim.asset.validation`` extension rather than the core
    SimReady tier wheel.
    """

    def CheckStage(self, stage: Usd.Stage) -> None:  # noqa: N802
        """Check for intersecting non-adjacent collision meshes.

        Args:
            stage: The USD stage to validate.
        """
        if PhysxSchema is None:
            self._AddFailedCheck(
                requirement=cap.BaseArticulationRequirements.BA_002,
                message="PhysxSchema is not available in this environment; cannot check non-adjacent collision meshes",
            )
            return
        self.adjacent_mesh_matrix = compute_adjacent_mesh_dict(stage)  # keyed on rigid-body (joint target) paths
        self.collisions_pairs = get_initial_collider_pairs(stage)  # tuples of collider Sdf paths in contact
        self._check_pairs(stage)

    def _check_pairs(self, stage: Usd.Stage) -> None:
        """Run the inner pair-filter loop.

        Factored out of :meth:`CheckStage` so tests can inject collision pairs
        without running a live PhysX simulation step.

        Two filters:

        - Pairs outside the ``stage.GetDefaultPrim()`` subtree are skipped. Ground
          planes and other environment scaffolding shipped alongside the robot are
          not the robot's non-adjacency problem to report.
        - Adjacency is looked up on the rigid body that *owns* each collider (via
          :func:`_find_rigid_body_ancestor`), not on the collider's immediate
          parent. Collision meshes nested deeper than one level under a link
          (e.g. ``/link/collisions/mesh_N``) would otherwise never match the
          adjacency dict's rigid-body keys.

        Args:
            stage: The USD stage being validated. Used for path->prim lookups.
        """
        default_prim = stage.GetDefaultPrim()
        if not default_prim or not default_prim.IsValid():
            return
        default_prim_path = default_prim.GetPath()

        for collision_pair in self.collisions_pairs:
            body0_prim = stage.GetPrimAtPath(collision_pair[0])
            body1_prim = stage.GetPrimAtPath(collision_pair[1])
            if not body0_prim or not body1_prim:
                continue

            # Both prims must live under defaultPrim.
            if not (
                body0_prim.GetPath().HasPrefix(default_prim_path) and body1_prim.GetPath().HasPrefix(default_prim_path)
            ):
                continue

            # Walk to the rigid-body ancestor for adjacency lookup.
            body0_rb = _find_rigid_body_ancestor(body0_prim)
            body1_rb = _find_rigid_body_ancestor(body1_prim)
            if body0_rb.isEmpty or body1_rb.isEmpty:
                # Orphan collider without a rigid-body ancestor. Not this validator's job.
                continue

            # check if the two bodies are adjacent
            if body1_rb in self.adjacent_mesh_matrix.get(body0_rb, []):
                continue
            self._AddFailedCheck(
                requirement=cap.BaseArticulationRequirements.BA_002,
                message=f"Colliding meshes {body0_prim.GetPath()} and {body1_prim.GetPath()} are not adjacent",
                at=body0_prim,
            )
