# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Newton property query interface compatible with PhysX property query API.

This module provides a Newton-compatible implementation of the PhysX property query
interface, allowing the same code to work with both backends.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import carb
from isaacsim.physics.newton.impl._warning_logging import _log_python_warnings
from pxr import Sdf, Usd, UsdPhysics, UsdUtils

if TYPE_CHECKING:
    from .tensors.articulation_view import NewtonArticulationView


class NewtonPropertyQueryArticulationLink:
    """Articulation link information compatible with PhysX response.

    Matches the interface of omni.physx.bindings._physx.PhysxPropertyQueryArticulationLink.

    Args:
        rigid_body: Encoded USD path of the rigid body.
        joint: Encoded USD path of the joint.
        joint_dof: Number of DOFs for this joint.
    """

    def __init__(self, rigid_body: int = 0, joint: int = 0, joint_dof: int = 0) -> None:
        self._rigid_body = rigid_body
        self._joint = joint
        self._joint_dof = joint_dof

    @property
    def rigid_body(self) -> int:
        """Encoded USD path of the rigid body.

        Returns:
            Encoded USD path of the rigid body.
        """
        return self._rigid_body

    @property
    def rigid_body_name(self) -> str:
        """USD path of the rigid body as string.

        Returns:
            USD path of the rigid body as string.
        """
        if self._rigid_body:
            from pxr import PhysicsSchemaTools

            return PhysicsSchemaTools.intToSdfPath(self._rigid_body).pathString
        return ""

    @property
    def joint(self) -> int:
        """Encoded USD path of the joint.

        Returns:
            Encoded USD path of the joint.
        """
        return self._joint

    @property
    def joint_name(self) -> str:
        """USD path of the joint as string.

        Returns:
            USD path of the joint as string.
        """
        if self._joint:
            from pxr import PhysicsSchemaTools

            return PhysicsSchemaTools.intToSdfPath(self._joint).pathString
        return ""

    @property
    def joint_dof(self) -> int:
        """Number of DOFs for this joint.

        Returns:
            Number of DOFs for this joint.
        """
        return self._joint_dof


class NewtonPropertyQueryArticulationResponse:
    """Articulation query response compatible with PhysX response.

    Matches the interface of omni.physx.bindings._physx.PhysxPropertyQueryArticulationResponse.

    Args:
        result: Query result code.
        stage_id: USD stage ID.
        path_id: Encoded USD prim path.
        links: List of articulation links.
    """

    def __init__(
        self,
        result: int = 0,
        stage_id: int = 0,
        path_id: int = 0,
        links: list[NewtonPropertyQueryArticulationLink] | None = None,
    ) -> None:
        self.result = result
        self.stage_id = stage_id
        self.path_id = path_id
        self.links = links if links is not None else []


class NewtonPropertyQueryInterface:
    """Newton implementation of property query interface."""

    _instance: NewtonPropertyQueryInterface | None = None
    """Singleton instance of the Newton property query interface."""

    def __new__(cls) -> NewtonPropertyQueryInterface:
        """Return singleton instance.

        Returns:
            The singleton instance of NewtonPropertyQueryInterface.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._articulation_cache = {}  # type: ignore[has-type]
            cls._instance._descriptor_cache = {}  # type: ignore[has-type]
            cls._instance._stage_event_subscription = None  # type: ignore[has-type]
        return cls._instance

    def _invalidate_cache_for_stage(self, stage_id: int) -> None:
        """Drop cached articulation responses for ``stage_id``.

        Args:
            stage_id: USD stage cache ID whose cached responses should be dropped.
        """
        self._articulation_cache = {
            key: value for key, value in self._articulation_cache.items() if key[0] != stage_id
        }  # type: ignore[has-type]
        self._descriptor_cache = {
            key: value for key, value in self._descriptor_cache.items() if key[0] != stage_id
        }  # type: ignore[has-type]

    def _ensure_stage_event_subscription(self) -> None:
        """Subscribe once to USD stage CLOSED events to drop stale cache entries."""
        if self._stage_event_subscription is not None:  # type: ignore[has-type]
            return
        try:
            import carb.eventdispatcher
            import omni.usd
        except ImportError:
            return
        usd_context = omni.usd.get_context()
        if usd_context is None:
            return

        def _on_stage_closed(event: object) -> None:
            current_stage = usd_context.get_stage()
            if current_stage is None:
                self._articulation_cache.clear()
                self._descriptor_cache.clear()
                return
            current_stage_id = UsdUtils.StageCache.Get().GetId(current_stage).ToLongInt()
            self._invalidate_cache_for_stage(current_stage_id)

        self._stage_event_subscription = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=usd_context.stage_event_name(omni.usd.StageEventType.CLOSED),
            on_event=_on_stage_closed,
            observer_name="NewtonPropertyQueryInterface stage cache invalidator",
        )

    def query_prim(
        self,
        stage_id: int,
        query_mode: int,
        prim_id: int,
        timeout_ms: int = -1,
        finished_fn: Callable | None = None,
        rigid_body_fn: Callable | None = None,
        collider_fn: Callable | None = None,
        articulation_fn: Callable | None = None,
    ) -> None:
        """Query articulation properties from Newton backend.

        Args:
            stage_id: USD stage ID.
            query_mode: Query mode (1 = QUERY_ARTICULATION).
            prim_id: Encoded USD prim path.
            timeout_ms: Timeout in milliseconds (-1 for no timeout).
            finished_fn: Callback when query is finished.
            rigid_body_fn: Callback for rigid body results.
            collider_fn: Callback for collider results.
            articulation_fn: Callback for articulation results.
        """
        # Only handle articulation queries
        if query_mode != 1:  # QUERY_ARTICULATION
            if finished_fn:
                finished_fn()
            return

        if articulation_fn is None:
            if finished_fn:
                finished_fn()
            return

        # Get the stage and prim path
        from pxr import PhysicsSchemaTools

        stage = UsdUtils.StageCache.Get().Find(Usd.StageCache.Id.FromLongInt(stage_id))
        if not stage:
            response = NewtonPropertyQueryArticulationResponse(result=3)  # ERROR_INVALID_USD_STAGE
            articulation_fn(response)
            if finished_fn:
                finished_fn()
            return

        prim_path = PhysicsSchemaTools.intToSdfPath(prim_id)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            response = NewtonPropertyQueryArticulationResponse(result=4)  # ERROR_INVALID_USD_PRIM
            articulation_fn(response)
            if finished_fn:
                finished_fn()
            return

        # Get Newton simulation view
        from isaacsim.core.simulation_manager import SimulationManager

        physics_sim_view = SimulationManager._physics_sim_view__warp
        if physics_sim_view is None or not physics_sim_view.is_valid:
            # No live sim view: invoke Newton's parser to compute the canonical
            # link/joint enumeration, so pre-physics metadata matches what the
            # articulation view will report once physics is initialized.
            self._ensure_stage_event_subscription()
            cache_key = (stage_id, prim_path.pathString)
            cached = self._articulation_cache.get(cache_key)
            if cached is not None:
                response = NewtonPropertyQueryArticulationResponse(
                    result=0,
                    stage_id=stage_id,
                    path_id=prim_id,
                    links=cached,
                )
                articulation_fn(response)
                if finished_fn:
                    finished_fn()
                return
            try:
                links = self._build_articulation_links_from_usd(stage, prim)
                self._articulation_cache[cache_key] = links
                response = NewtonPropertyQueryArticulationResponse(
                    result=0,  # VALID
                    stage_id=stage_id,
                    path_id=prim_id,
                    links=links,
                )
                articulation_fn(response)
            except Exception as e:
                carb.log_error(f"Failed to query articulation from USD: {e}")
                response = NewtonPropertyQueryArticulationResponse(result=6)  # ERROR_RUNTIME
                articulation_fn(response)

            if finished_fn:
                finished_fn()
            return

        # Create articulation view
        articulation_view = physics_sim_view.create_articulation_view(prim_path.pathString)

        if articulation_view is None or articulation_view.count == 0:
            response = NewtonPropertyQueryArticulationResponse(result=6)  # ERROR_RUNTIME
            articulation_fn(response)
            if finished_fn:
                finished_fn()
            return

        # Build the response from Newton's articulation metadata
        try:
            links = self._build_articulation_links(stage, articulation_view)

            response = NewtonPropertyQueryArticulationResponse(
                result=0,  # VALID
                stage_id=stage_id,
                path_id=prim_id,
                links=links,
            )
            articulation_fn(response)

        except Exception as e:
            carb.log_error(f"Newton property query failed: {e}")
            import traceback

            traceback.print_exc()
            response = NewtonPropertyQueryArticulationResponse(result=6)  # ERROR_RUNTIME
            articulation_fn(response)

        if finished_fn:
            finished_fn()

    def _build_articulation_links(
        self, stage: Usd.Stage, articulation_view: "NewtonArticulationView"
    ) -> list[NewtonPropertyQueryArticulationLink]:
        """Build articulation link list matching PhysX format.

        PhysX returns one entry per link, where each link may have an associated
        inbound joint. The mapping is: link[i] is connected by joint[i].

        Args:
            stage: USD stage.
            articulation_view: Newton articulation view.

        Returns:
            List of articulation links in PhysX-compatible format.
        """
        from pxr import PhysicsSchemaTools

        links = []  # type: ignore[var-annotated]
        # Get paths for the first articulation (homogeneous view, all have same structure)
        all_link_paths = articulation_view.link_paths
        all_dof_paths = articulation_view.dof_paths

        if not all_link_paths or len(all_link_paths) == 0:
            return links

        link_paths = all_link_paths[0]
        dof_paths = all_dof_paths[0] if all_dof_paths else []
        joint_paths = list(dict.fromkeys(p for p in dof_paths if p))

        # Create a mapping from joint paths to DOF count by counting DOFs
        # In Newton, each DOF path corresponds to exactly one joint
        joint_dof_count_map = {}  # type: ignore[var-annotated]
        for dof_path in dof_paths:
            # Count this DOF for its joint
            joint_dof_count_map[dof_path] = joint_dof_count_map.get(dof_path, 0) + 1

        # Build link entries
        # In Newton's structure, links and joints are aligned (each joint connects to a child link)
        for i, link_path in enumerate(link_paths):
            link = NewtonPropertyQueryArticulationLink()

            # Encode the link path
            link_sdf_path = Sdf.Path(link_path)
            link._rigid_body = PhysicsSchemaTools.sdfPathToInt(link_sdf_path)

            # Map link to its inbound joint
            # joint[i] connects to link[i]
            if i < len(joint_paths):
                joint_path = joint_paths[i]
                if joint_path:
                    joint_sdf_path = Sdf.Path(joint_path)
                    link._joint = PhysicsSchemaTools.sdfPathToInt(joint_sdf_path)
                    # Get DOF count from our mapping
                    link._joint_dof = joint_dof_count_map.get(joint_path, 0)

            links.append(link)

        return links

    def _build_articulation_links_from_usd(
        self, stage: Usd.Stage, articulation_prim: Usd.Prim
    ) -> list[NewtonPropertyQueryArticulationLink]:
        """Build articulation link list by parsing USD with Newton's own importer.

        Used when the physics simulation view is not available. Runs
        ``newton.ModelBuilder.add_usd`` confined to the articulation subtree so the
        link/joint enumeration matches what ``NewtonArticulationView`` will produce
        once physics is initialized (including the effect of
        ``collapse_fixed_joints`` and the ``"dfs"`` joint ordering).

        Visual shapes, sites, mesh approximation, and MuJoCo option parsing are
        skipped to keep this query inexpensive.

        Args:
            stage: USD stage.
            articulation_prim: USD prim passed to ``query_prim``.

        Returns:
            List of articulation links in PhysX-compatible format.
        """
        import newton
        from newton.usd import SchemaResolverNewton, SchemaResolverPhysx  # type: ignore[attr-defined]
        from pxr import PhysicsSchemaTools

        collapse_fixed_joints = self._get_collapse_fixed_joints_setting()
        articulation_path = articulation_prim.GetPath().pathString
        root_path = self._resolve_add_usd_root_path(stage, articulation_prim)

        builder = newton.ModelBuilder()
        with _log_python_warnings():
            builder.add_usd(
                source=stage,
                root_path=root_path,
                collapse_fixed_joints=collapse_fixed_joints,
                schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
                only_load_enabled_rigid_bodies=True,
                load_visual_shapes=False,
                load_sites=False,
                skip_mesh_approximation=True,
                parse_mujoco_options=False,
                verbose=False,
            )

        # Mirror NewtonArticulationView's filtering rules so pre-physics metadata
        # matches what the runtime view publishes:
        #   * Skip the first joint of every articulation (the FREE joint of a
        #     floating-base articulation or the implicit fixed-base root joint).
        #   * Keep only joint types the runtime view enumerates as articulation
        #     joints; everything else (FIXED, FREE, DISTANCE, CABLE, ...) is dropped.
        # See ``_build_articulations_helper`` in ``isaacsim/physics/newton/impl/tensors/backend.py``.
        articulation_index = self._find_builder_articulation_index(builder.articulation_label, articulation_path)
        joint_start = builder.articulation_start[articulation_index]
        if articulation_index + 1 < len(builder.articulation_start):
            joint_end = builder.articulation_start[articulation_index + 1]
        else:
            joint_end = builder.joint_count

        included_joint_types = {
            newton.JointType.PRISMATIC,
            newton.JointType.REVOLUTE,
            newton.JointType.BALL,
            newton.JointType.D6,
        }

        body_to_inbound_joint: dict[int, int] = {}
        articulation_body_indices: set[int] = set()
        for joint_index in range(joint_start, joint_end):
            parent_body = builder.joint_parent[joint_index]
            child_body = builder.joint_child[joint_index]
            if parent_body >= 0:
                articulation_body_indices.add(parent_body)
            if child_body < 0:
                continue
            articulation_body_indices.add(child_body)
            if joint_index == joint_start:
                continue
            if builder.joint_type[joint_index] not in included_joint_types:
                continue
            body_to_inbound_joint.setdefault(child_body, joint_index)

        links: list[NewtonPropertyQueryArticulationLink] = []
        for body_index, body_label in enumerate(builder.body_label):
            if body_index not in articulation_body_indices:
                continue
            link = NewtonPropertyQueryArticulationLink()
            link._rigid_body = PhysicsSchemaTools.sdfPathToInt(Sdf.Path(body_label))
            joint_index = body_to_inbound_joint.get(body_index)  # type: ignore[assignment]
            if joint_index is not None:
                linear_axes, angular_axes = builder.joint_dof_dim[joint_index]
                joint_label = builder.joint_label[joint_index]
                link._joint = PhysicsSchemaTools.sdfPathToInt(Sdf.Path(joint_label))
                link._joint_dof = linear_axes + angular_axes
            links.append(link)
        return links

    def _resolve_add_usd_root_path(self, stage: Usd.Stage, articulation_prim: Usd.Prim) -> str:
        """Return the ``add_usd`` root path for the articulation of ``articulation_prim``.

        The USD physics parser owns articulation membership. Prefer the nearest
        ``IsaacRobotAPI`` ancestor as a narrow parser scope, then fall back to the
        stage pseudo-root for generic USD layouts where articulated bodies and
        joints may live outside the robot namespace.

        Args:
            stage: USD stage containing the articulation.
            articulation_prim: USD prim passed to ``query_prim``.

        Returns:
            USD path string to use as the ``add_usd`` root.
        """
        articulation_path = articulation_prim.GetPath().pathString
        scope_paths = self._descriptor_scope_paths(articulation_prim)
        for scope_path in scope_paths:
            bodies, joints = self._query_articulation_descriptor(stage, scope_path, articulation_path)
            if not bodies:
                continue
            return self._common_ancestor_path({articulation_path} | bodies | joints)

        carb.log_warn(
            f"Newton property query: USD physics parser did not report articulated bodies for "
            f"'{articulation_path}'; falling back to legacy root resolution"
        )
        return self._resolve_add_usd_root_path_legacy(articulation_prim)

    @staticmethod
    def _resolve_add_usd_root_path_legacy(articulation_prim: Usd.Prim) -> str:
        """Return the legacy parent-of-joint ``add_usd`` root path.

        When ``ArticulationRootAPI`` is authored on a ``UsdPhysics.Joint`` (a fixed
        "root joint" that anchors a fixed-base articulation), the joint prim itself
        has no rigid-body subtree, so ``newton.ModelBuilder.add_usd`` rooted there
        parses nothing. The enclosing (parent) prim is the container that holds the
        articulation's links, so use it instead. For every other case the prim is
        used unchanged.

        Args:
            articulation_prim: USD prim passed to ``query_prim``.

        Returns:
            USD path string to use as the ``add_usd`` root.
        """
        if articulation_prim.IsA(UsdPhysics.Joint):
            parent = articulation_prim.GetParent()
            if parent and parent.IsValid() and not parent.IsPseudoRoot():
                return parent.GetPath().pathString
        return articulation_prim.GetPath().pathString

    @staticmethod
    def _descriptor_scope_paths(articulation_prim: Usd.Prim) -> list[Sdf.Path]:
        """Return parser scopes to try for an articulation descriptor.

        The robot schema scope is a performance optimization for typical Isaac
        assets. The stage pseudo-root remains the correctness fallback for generic
        USD articulations whose relationships cross namespace boundaries.

        Args:
            articulation_prim: USD prim passed to ``query_prim``.

        Returns:
            Candidate parser scope paths in priority order.
        """
        scope_paths: list[Sdf.Path] = []
        current = articulation_prim
        while current is not None and current.IsValid() and not current.IsPseudoRoot():
            if NewtonPropertyQueryInterface._has_isaac_robot_api(current):
                scope_paths.append(current.GetPath())
                break
            current = current.GetParent()
        if Sdf.Path.absoluteRootPath not in scope_paths:
            scope_paths.append(Sdf.Path.absoluteRootPath)
        return scope_paths

    @staticmethod
    def _has_isaac_robot_api(prim: Usd.Prim) -> bool:
        """Return whether ``prim`` authors ``IsaacRobotAPI``.

        Args:
            prim: USD prim to inspect.

        Returns:
            True if ``IsaacRobotAPI`` is applied to ``prim``.
        """
        if prim.HasAPI("IsaacRobotAPI"):
            return True
        api_schemas = prim.GetMetadata("apiSchemas")
        return api_schemas is not None and "IsaacRobotAPI" in api_schemas.GetAddedOrExplicitItems()

    def _query_articulation_descriptor(
        self, stage: Usd.Stage, scope_path: Sdf.Path, articulation_path: str
    ) -> tuple[set[str], set[str]]:
        """Return parser-reported body and joint membership for an articulation.

        Args:
            stage: USD stage to inspect.
            scope_path: Parser scope passed to ``UsdPhysics.LoadUsdPhysicsFromRange``.
            articulation_path: Path of the prim with ``ArticulationRootAPI``.

        Returns:
            Sets of articulated body paths and joint paths. Both sets are empty if
            the scoped parse cannot report the requested articulation.
        """
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        cache_key = (stage_id, scope_path.pathString)
        descriptors = self._descriptor_cache.get(cache_key)  # type: ignore[has-type]
        if descriptors is None:
            descriptors = self._query_articulation_descriptors(stage, scope_path)
            self._descriptor_cache[cache_key] = descriptors  # type: ignore[has-type]
        return descriptors.get(articulation_path, (set(), set()))

    @staticmethod
    def _query_articulation_descriptors(stage: Usd.Stage, scope_path: Sdf.Path) -> dict[str, tuple[set[str], set[str]]]:
        """Return articulation descriptors keyed by articulation root path.

        Args:
            stage: USD stage to inspect.
            scope_path: Parser scope passed to ``UsdPhysics.LoadUsdPhysicsFromRange``.

        Returns:
            Mapping from articulation root path to parser-reported body and joint
            membership.
        """
        try:
            ret_dict = UsdPhysics.LoadUsdPhysicsFromRange(stage, [scope_path], excludePaths=[])
        except Exception as exc:  # noqa: BLE001 - parser errors fall back to legacy root resolution
            carb.log_warn(f"Newton property query: LoadUsdPhysicsFromRange failed for '{scope_path}': {exc}")
            return {}

        articulation_data = ret_dict.get(UsdPhysics.ObjectType.Articulation)
        if articulation_data is None:
            return {}

        descriptors: dict[str, tuple[set[str], set[str]]] = {}
        paths, descriptions = articulation_data
        for path, description in zip(paths, descriptions):
            bodies = {path.pathString for path in description.articulatedBodies if path != Sdf.Path.emptyPath}
            joints = {path.pathString for path in description.articulatedJoints if path != Sdf.Path.emptyPath}
            descriptors[path.pathString] = (bodies, joints)
        return descriptors

    @staticmethod
    def _common_ancestor_path(paths: set[str]) -> str:
        """Return the deepest common ancestor path of ``paths``.

        Args:
            paths: Non-empty set of absolute USD prim path strings.

        Returns:
            USD path string of the deepest prim enclosing every input path.
        """
        common = Sdf.Path(next(iter(paths)))
        for path in paths:
            common = common.GetCommonPrefix(Sdf.Path(path))
        return common.pathString

    @staticmethod
    def _find_builder_articulation_index(articulation_labels: list[str], articulation_path: str) -> int:
        """Return the builder articulation index for ``articulation_path``.

        Args:
            articulation_labels: Labels from ``newton.ModelBuilder``.
            articulation_path: Path of the queried articulation root.

        Returns:
            Matching builder articulation index.

        Raises:
            RuntimeError: If no unique articulation matches.
        """
        indices = [index for index, label in enumerate(articulation_labels) if label == articulation_path]
        if not indices:
            raise RuntimeError(f"Newton did not import the articulation rooted at {articulation_path}")
        if len(indices) > 1:
            raise RuntimeError(f"Newton imported multiple articulations rooted at {articulation_path}")
        return indices[0]

    @staticmethod
    def _get_collapse_fixed_joints_setting() -> bool:
        """Return the ``collapse_fixed_joints`` flag from the active ``NewtonStage`` config.

        Returns:
            True if fixed joints should be collapsed, False otherwise.
        """
        try:
            from . import extension as newton_extension
        except ImportError:
            return False
        active_stage = getattr(newton_extension, "_newton_stage", None)
        if active_stage is None:
            return False
        cfg = getattr(active_stage, "cfg", None)
        if cfg is None:
            return False
        return bool(getattr(cfg, "collapse_fixed_joints", False))


def get_newton_property_query_interface() -> NewtonPropertyQueryInterface:
    """Get the Newton property query interface singleton.

    Returns:
        Newton property query interface instance.
    """
    return NewtonPropertyQueryInterface()
