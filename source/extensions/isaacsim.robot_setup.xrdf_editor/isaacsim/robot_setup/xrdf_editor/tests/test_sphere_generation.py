# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for link/mesh attribution in sphere_generation."""

from __future__ import annotations

import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
from isaacsim.robot_setup.xrdf_editor import sphere_generation
from pxr import Gf, UsdGeom

_ROBOT_PATH = "/World/robot"

# Minimal cube geometry. `find_link_meshes` only records prims that are a
# `UsdGeom.Mesh` with authored points, so `UsdGeom.Cube` will not do.
_CUBE_POINTS = [
    (-0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5),
    (-0.5, -0.5, 0.5),
    (0.5, -0.5, 0.5),
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
]
_CUBE_INDICES = [0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1, 1, 5, 6, 2, 2, 6, 7, 3, 3, 7, 4, 0]
_CUBE_COUNTS = [4] * 6


def _define_mesh(path: str) -> None:
    """Author a unit cube mesh at ``path``."""
    stage = stage_utils.get_current_stage()
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in _CUBE_POINTS])
    mesh.GetFaceVertexIndicesAttr().Set(_CUBE_INDICES)
    mesh.GetFaceVertexCountsAttr().Set(_CUBE_COUNTS)


class TestFindLinkMeshes(omni.kit.test.AsyncTestCase):
    """Test that meshes are attributed to the correct link."""

    async def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        await stage_utils.create_new_stage_async()
        stage_utils.define_prim(_ROBOT_PATH, "Xform")

    def _define_link(self, link_subpath: str) -> None:
        """Author a link Xform with one mesh under it."""
        stage_utils.define_prim(_ROBOT_PATH + link_subpath, "Xform")
        _define_mesh(_ROBOT_PATH + link_subpath + "/visuals")

    def _find(self, *link_subpaths: str) -> dict[str, list[str]]:
        """Run attribution for links identified by subpath under the robot root."""
        link_paths = [_ROBOT_PATH + subpath for subpath in link_subpaths]
        return dict(sphere_generation.find_link_meshes(stage_utils.get_current_stage(), _ROBOT_PATH, link_paths))

    # -------------------------------------------------------------------------
    # Flat hierarchies
    # -------------------------------------------------------------------------

    async def test_flat_links_each_own_their_mesh(self) -> None:
        """Every link parented directly under the root keeps its own mesh."""
        for subpath in ("/base_link", "/link1", "/link2"):
            self._define_link(subpath)

        self.assertEqual(
            self._find("/base_link", "/link1", "/link2"),
            {"/base_link": ["/visuals"], "/link1": ["/visuals"], "/link2": ["/visuals"]},
        )

    async def test_prefix_colliding_link_names_are_kept_apart(self) -> None:
        """`link1` must not absorb the meshes of `link10` or `link1_tip`."""
        for subpath in ("/link1", "/link10", "/link1_tip"):
            self._define_link(subpath)

        self.assertEqual(
            self._find("/link1", "/link10", "/link1_tip"),
            {"/link1": ["/visuals"], "/link10": ["/visuals"], "/link1_tip": ["/visuals"]},
        )

    async def test_link_authored_directly_as_a_mesh(self) -> None:
        """A link that is itself a mesh is recorded with an empty mesh subpath."""
        _define_mesh(_ROBOT_PATH + "/link1")

        self.assertEqual(self._find("/link1"), {"/link1": [""]})

    # -------------------------------------------------------------------------
    # Nested links
    # -------------------------------------------------------------------------

    async def test_nested_link_is_discovered(self) -> None:
        """A link nested inside another link is offered as a link in its own right.

        Attributing by the first matching path component instead drops the nested
        link entirely, so it can never be selected in the panel.
        """
        self._define_link("/base_link")
        self._define_link("/base_link/arm_link")

        self.assertIn("/base_link/arm_link", self._find("/base_link", "/base_link/arm_link"))

    async def test_nested_link_mesh_is_not_absorbed_by_parent(self) -> None:
        """The parent link keeps only its own mesh, not the nested link's.

        When the nested mesh is folded into the parent, spheres generated for the
        parent are fitted around geometry that moves with the child instead.
        """
        self._define_link("/base_link")
        self._define_link("/base_link/arm_link")

        self.assertEqual(
            self._find("/base_link", "/base_link/arm_link"),
            {"/base_link": ["/visuals"], "/base_link/arm_link": ["/visuals"]},
        )

    async def test_nested_link_authored_as_a_mesh_is_not_absorbed_by_parent(self) -> None:
        """A nested link that is itself a mesh keeps its geometry.

        Name-based attribution that prefers an ancestor match hands this mesh to
        the enclosing link, so the nested link disappears exactly as it does in
        the Xform case.
        """
        self._define_link("/base_link")
        _define_mesh(_ROBOT_PATH + "/base_link/arm_link")

        self.assertEqual(
            self._find("/base_link", "/base_link/arm_link"),
            {"/base_link": ["/visuals"], "/base_link/arm_link": [""]},
        )

    async def test_deeply_nested_link_attributed_to_innermost_link(self) -> None:
        """Attribution picks the innermost enclosing link, not an intermediate one."""
        self._define_link("/base_link")
        self._define_link("/base_link/arm_link")
        self._define_link("/base_link/arm_link/tool_link")

        result = self._find("/base_link", "/base_link/arm_link", "/base_link/arm_link/tool_link")

        self.assertEqual(result.get("/base_link/arm_link/tool_link"), ["/visuals"])
        self.assertEqual(result.get("/base_link/arm_link"), ["/visuals"])
        self.assertEqual(result.get("/base_link"), ["/visuals"])

    async def test_two_links_sharing_a_prim_name_stay_separate(self) -> None:
        """Links whose prims share a name under different parents keep their own meshes."""
        self._define_link("/arm_a")
        self._define_link("/arm_a/tool")
        self._define_link("/arm_b")
        self._define_link("/arm_b/tool")

        result = self._find("/arm_a", "/arm_a/tool", "/arm_b", "/arm_b/tool")

        self.assertEqual(result.get("/arm_a/tool"), ["/visuals"])
        self.assertEqual(result.get("/arm_b/tool"), ["/visuals"])

    # -------------------------------------------------------------------------
    # Prims that are not links
    # -------------------------------------------------------------------------

    async def test_mesh_under_intermediate_xform_belongs_to_enclosing_link(self) -> None:
        """A non-link grouping Xform does not break attribution."""
        stage_utils.define_prim(_ROBOT_PATH + "/base_link", "Xform")
        stage_utils.define_prim(_ROBOT_PATH + "/base_link/geometry", "Xform")
        _define_mesh(_ROBOT_PATH + "/base_link/geometry/visuals")

        self.assertEqual(self._find("/base_link"), {"/base_link": ["/geometry/visuals"]})

    async def test_mesh_named_after_another_link_stays_with_its_own_link(self) -> None:
        """A mesh that merely shares a link's name is not treated as that link.

        `link2` is a real link elsewhere on the robot, so matching by name hands
        `/link1/link2` to it even though that prim is only a mesh.
        """
        stage_utils.define_prim(_ROBOT_PATH + "/link1", "Xform")
        _define_mesh(_ROBOT_PATH + "/link1/link2")
        self._define_link("/link2")

        result = self._find("/link1", "/link2")

        self.assertEqual(result.get("/link1"), ["/link2"])
        self.assertEqual(result.get("/link2"), ["/visuals"])

    async def test_grouping_xform_named_after_another_link_is_not_a_link(self) -> None:
        """An intermediate Xform sharing a link's name does not capture the mesh."""
        stage_utils.define_prim(_ROBOT_PATH + "/base_link", "Xform")
        stage_utils.define_prim(_ROBOT_PATH + "/base_link/arm_link", "Xform")
        _define_mesh(_ROBOT_PATH + "/base_link/arm_link/visuals")
        self._define_link("/arm_link")

        result = self._find("/base_link", "/arm_link")

        self.assertEqual(result.get("/base_link"), ["/arm_link/visuals"])
        self.assertEqual(result.get("/arm_link"), ["/visuals"])

    async def test_mesh_outside_any_link_is_skipped(self) -> None:
        """A mesh with no enclosing link is dropped rather than misattributed."""
        _define_mesh(_ROBOT_PATH + "/stray_mesh")

        self.assertEqual(self._find("/link1"), {})

    async def test_articulation_root_with_trailing_slash(self) -> None:
        """A trailing slash on the articulation root does not shift the path slicing."""
        self._define_link("/base_link")

        link_paths = [_ROBOT_PATH + "/base_link"]
        result = sphere_generation.find_link_meshes(stage_utils.get_current_stage(), _ROBOT_PATH + "/", link_paths)

        self.assertEqual(dict(result), {"/base_link": ["/visuals"]})
