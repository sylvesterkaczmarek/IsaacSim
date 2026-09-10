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

"""Tests for CollisionSphereEditor."""

import io
import os
import tempfile

import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.test
from isaacsim.robot_setup.xrdf_editor.collision_sphere_editor import CollisionSphereEditor

_ROBOT_PATH = "/World/robot"
_LINK1_PATH = "/World/robot/link1"
_LINK2_PATH = "/World/robot/link2"
# `link1` is a strict string prefix of both of these, so any sphere lookup that compares raw
# path prefixes without a boundary will match them when it was asked for `link1` alone.
_LINK10_PATH = "/World/robot/link10"
_LINK1_TIP_PATH = "/World/robot/link1_tip"
# Prefix-colliding articulation root, for the export guards.
_ROBOT2_PATH = "/World/robot2"
_ROBOT2_LINK1_PATH = "/World/robot2/link1"
# `arm_link` is a link in its own right that lives inside another link. Any sphere lookup
# scoped by subtree rather than by owning link treats its spheres as belonging to
# `base_link`, and any export key derived by slicing the path writes "base_link/arm_link".
_BASE_LINK_PATH = "/World/robot/base_link"
_ARM_LINK_PATH = "/World/robot/base_link/arm_link"


def _radius_of(editor: CollisionSphereEditor, sphere_path: str) -> float:
    """Read a sphere radius as a plain float.

    `set_radii` can leave the stored radii as a nested array, which makes `assertAlmostEqual`
    raise `TypeError` instead of reporting the value it saw. Flattening keeps a wrong radius
    an assertion failure rather than an error, and requiring a single element stops an
    unexpected shape from being reported as a passing scalar.
    """
    radii = np.ravel(editor.path_2_spheres[sphere_path].get_radii().numpy())
    if radii.size != 1:
        raise AssertionError(f"Expected one radius for {sphere_path}, got shape {radii.shape}")
    return float(radii[0])


def _color_of(editor: CollisionSphereEditor, sphere_path: str) -> np.ndarray:
    """Read a sphere display color as a flat RGB array."""
    color = editor.path_2_spheres[sphere_path].geoms[0].GetDisplayColorAttr().Get()
    return np.ravel(np.array(color))[:3]


class TestCollisionSphereEditor(omni.kit.test.AsyncTestCase):
    """Test CollisionSphereEditor API."""

    async def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        await stage_utils.create_new_stage_async()
        stage_utils.define_prim(_ROBOT_PATH, "Xform")
        stage_utils.define_prim(_LINK1_PATH, "Xform")
        stage_utils.define_prim(_LINK2_PATH, "Xform")
        stage_utils.define_prim(_LINK10_PATH, "Xform")
        stage_utils.define_prim(_LINK1_TIP_PATH, "Xform")
        stage_utils.define_prim(_BASE_LINK_PATH, "Xform")
        stage_utils.define_prim(_ARM_LINK_PATH, "Xform")
        self.editor = CollisionSphereEditor()
        # `EditorState` supplies this from the articulation whenever one is
        # selected, so exercising the editor without it would test a
        # configuration the UI never produces.
        self.editor.set_link_names(
            {
                _LINK1_PATH: "link1",
                _LINK2_PATH: "link2",
                _LINK10_PATH: "link10",
                _LINK1_TIP_PATH: "link1_tip",
                _BASE_LINK_PATH: "base_link",
                _ARM_LINK_PATH: "arm_link",
            }
        )

    async def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.editor.on_shutdown()
        super().tearDown()

    # -------------------------------------------------------------------------
    # add_sphere / delete_sphere
    # -------------------------------------------------------------------------

    async def test_add_sphere_returns_valid_path(self) -> None:
        """Test add sphere returns valid path."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.array([0.1, 0.0, 0.0]), 0.05)
        self.assertIsNotNone(sphere_path)
        self.assertIn(sphere_path, self.editor.path_2_spheres)
        prim = prim_utils.get_prim_at_path(sphere_path)
        self.assertTrue(prim and prim.IsValid())

    async def test_add_sphere_path_nested_under_link(self) -> None:
        """Test add sphere path nested under link."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.assertTrue(sphere_path.startswith(_LINK1_PATH))

    async def test_add_multiple_spheres_unique_paths(self) -> None:
        """Test add multiple spheres unique paths."""
        path1 = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        path2 = self.editor.add_sphere(_LINK1_PATH, np.array([0.1, 0.0, 0.0]), 0.05)
        self.assertNotEqual(path1, path2)
        self.assertEqual(len(self.editor.path_2_spheres), 2)

    async def test_delete_sphere_removes_from_dict(self) -> None:
        """Test delete sphere removes from dict."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.delete_sphere(sphere_path)
        self.assertNotIn(sphere_path, self.editor.path_2_spheres)

    async def test_delete_sphere_removes_prim(self) -> None:
        """Test delete sphere removes prim."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.delete_sphere(sphere_path)
        prim = prim_utils.get_prim_at_path(sphere_path)
        self.assertFalse(prim and prim.IsValid())

    async def test_delete_nonexistent_sphere_is_safe(self) -> None:
        """Test delete nonexistent sphere is safe."""
        # Should not raise even when the path is not tracked.
        self.editor.delete_sphere("/World/robot/link1/nonexistent_sphere")

    # -------------------------------------------------------------------------
    # clear_spheres / clear_link_spheres
    # -------------------------------------------------------------------------

    async def test_clear_spheres_removes_all(self) -> None:
        """Test clear spheres removes all."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.add_sphere(_LINK2_PATH, np.ones(3) * 0.1, 0.05)
        self.editor.clear_spheres()
        self.assertEqual(len(self.editor.path_2_spheres), 0)

    async def test_clear_spheres_on_empty_editor_is_safe(self) -> None:
        """Test clear spheres on empty editor is safe."""
        self.editor.clear_spheres()
        self.assertEqual(len(self.editor.path_2_spheres), 0)

    async def test_clear_link_spheres_removes_only_target_link(self) -> None:
        """Test clear link spheres removes only target link."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.add_sphere(_LINK1_PATH, np.array([0.1, 0.0, 0.0]), 0.05)
        link2_path = self.editor.add_sphere(_LINK2_PATH, np.zeros(3), 0.05)
        self.editor.clear_link_spheres(_LINK1_PATH)
        remaining = list(self.editor.path_2_spheres.keys())
        self.assertEqual(remaining, [link2_path])

    async def test_clear_link_spheres_keeps_prefix_sibling_links(self) -> None:
        """Test clear link spheres keeps links whose names extend the target name."""
        # Regression: an unanchored prefix comparison made `link1` also match `link10` and
        # `link1_tip`, silently deleting their spheres.
        link1_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        link10_path = self.editor.add_sphere(_LINK10_PATH, np.zeros(3), 0.05)
        link1_tip_path = self.editor.add_sphere(_LINK1_TIP_PATH, np.zeros(3), 0.05)

        self.editor.clear_link_spheres(_LINK1_PATH)

        self.assertNotIn(link1_path, self.editor.path_2_spheres)
        self.assertIn(link10_path, self.editor.path_2_spheres)
        self.assertIn(link1_tip_path, self.editor.path_2_spheres)
        for surviving_path in (link10_path, link1_tip_path):
            prim = prim_utils.get_prim_at_path(surviving_path)
            self.assertTrue(prim and prim.IsValid(), f"{surviving_path} should not have been deleted")

    async def test_clear_link_spheres_tolerates_trailing_slash(self) -> None:
        """Test clear link spheres accepts a link path with a trailing slash."""
        link1_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        link10_path = self.editor.add_sphere(_LINK10_PATH, np.zeros(3), 0.05)

        self.editor.clear_link_spheres(_LINK1_PATH + "/")

        self.assertNotIn(link1_path, self.editor.path_2_spheres)
        self.assertIn(link10_path, self.editor.path_2_spheres)

    async def test_clear_link_spheres_keeps_nested_link_spheres(self) -> None:
        """Test clearing a link leaves the spheres of a link nested inside it."""
        # Subtree matching deletes the nested link's spheres too, even though they belong to
        # a link that moves independently of the one the user asked to clear.
        base_sphere = self.editor.add_sphere(_BASE_LINK_PATH, np.zeros(3), 0.05)
        arm_sphere = self.editor.add_sphere(_ARM_LINK_PATH, np.zeros(3), 0.05)

        self.editor.clear_link_spheres(_BASE_LINK_PATH)

        self.assertNotIn(base_sphere, self.editor.path_2_spheres)
        self.assertIn(arm_sphere, self.editor.path_2_spheres)
        prim = prim_utils.get_prim_at_path(arm_sphere)
        self.assertTrue(prim and prim.IsValid(), f"{arm_sphere} should not have been deleted")

    async def test_clear_link_spheres_on_nested_link_keeps_parent_spheres(self) -> None:
        """Test clearing a nested link leaves its parent link's spheres."""
        base_sphere = self.editor.add_sphere(_BASE_LINK_PATH, np.zeros(3), 0.05)
        arm_sphere = self.editor.add_sphere(_ARM_LINK_PATH, np.zeros(3), 0.05)

        self.editor.clear_link_spheres(_ARM_LINK_PATH)

        self.assertNotIn(arm_sphere, self.editor.path_2_spheres)
        self.assertIn(base_sphere, self.editor.path_2_spheres)

    # -------------------------------------------------------------------------
    # get_sphere_names_by_link
    # -------------------------------------------------------------------------

    async def test_get_sphere_names_by_link_returns_correct_count(self) -> None:
        """Test get sphere names by link returns correct count."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.add_sphere(_LINK1_PATH, np.array([0.1, 0.0, 0.0]), 0.05)
        self.editor.add_sphere(_LINK2_PATH, np.zeros(3), 0.05)
        names = self.editor.get_sphere_names_by_link(_LINK1_PATH)
        self.assertEqual(len(names), 2)

    async def test_get_sphere_names_by_link_are_relative(self) -> None:
        """Test get sphere names by link are relative."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        names = self.editor.get_sphere_names_by_link(_LINK1_PATH)
        for name in names:
            self.assertTrue(name.startswith("/"), f"Expected relative path starting with '/', got: {name}")

    async def test_get_sphere_names_by_link_empty_when_none(self) -> None:
        """Test get sphere names by link empty when none."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        names = self.editor.get_sphere_names_by_link(_LINK2_PATH)
        self.assertEqual(names, [])

    # -------------------------------------------------------------------------
    # scale_spheres
    # -------------------------------------------------------------------------

    async def test_scale_spheres_doubles_radius(self) -> None:
        """Test scale spheres doubles radius."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.1)
        self.editor.scale_spheres(_LINK1_PATH, 2.0)
        radius = self.editor.path_2_spheres[sphere_path].get_radii().numpy()[0]
        self.assertAlmostEqual(radius, 0.2, places=4)

    async def test_scale_spheres_only_affects_target_link(self) -> None:
        """Test scale spheres leaves an unrelated link untouched."""
        path1 = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.1)
        path2 = self.editor.add_sphere(_LINK2_PATH, np.zeros(3), 0.1)
        self.editor.scale_spheres(_LINK1_PATH, 3.0)
        radius1 = self.editor.path_2_spheres[path1].get_radii().numpy()[0]
        radius2 = self.editor.path_2_spheres[path2].get_radii().numpy()[0]
        self.assertAlmostEqual(radius1, 0.3, places=4)
        self.assertAlmostEqual(radius2, 0.1, places=4)

    async def test_scale_spheres_keeps_prefix_sibling_links_unscaled(self) -> None:
        """Test scale spheres skips links whose names extend the target name."""
        # `link1` / `link2` fixtures cannot catch this: they share no prefix, so the assertion
        # holds even with an unanchored comparison. These names can fail.
        target_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.10)
        link10_path = self.editor.add_sphere(_LINK10_PATH, np.zeros(3), 0.07)
        link1_tip_path = self.editor.add_sphere(_LINK1_TIP_PATH, np.zeros(3), 0.09)

        self.editor.scale_spheres(_LINK1_PATH, 3.0)

        self.assertAlmostEqual(_radius_of(self.editor, target_path), 0.30, places=4)
        self.assertAlmostEqual(_radius_of(self.editor, link10_path), 0.07, places=4)
        self.assertAlmostEqual(_radius_of(self.editor, link1_tip_path), 0.09, places=4)

    async def test_scale_spheres_scales_whole_articulation_subtree(self) -> None:
        """Test scale spheres applied to an articulation root scales every nested link."""
        # `_on_scale_all_spheres` passes the articulation root, so subtree semantics must
        # survive the boundary-anchored matching.
        link1_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.10)
        link2_path = self.editor.add_sphere(_LINK2_PATH, np.zeros(3), 0.20)

        self.editor.scale_spheres(_ROBOT_PATH, 2.0)

        self.assertAlmostEqual(_radius_of(self.editor, link1_path), 0.20, places=4)
        self.assertAlmostEqual(_radius_of(self.editor, link2_path), 0.40, places=4)

    async def test_scale_spheres_does_not_cross_prefix_colliding_articulations(self) -> None:
        """Test scaling one articulation root leaves a prefix-colliding sibling robot alone."""
        stage_utils.define_prim(_ROBOT2_PATH, "Xform")
        stage_utils.define_prim(_ROBOT2_LINK1_PATH, "Xform")
        robot_sphere = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.10)
        robot2_sphere = self.editor.add_sphere(_ROBOT2_LINK1_PATH, np.zeros(3), 0.05)

        self.editor.scale_spheres(_ROBOT_PATH, 2.0)

        self.assertAlmostEqual(_radius_of(self.editor, robot_sphere), 0.20, places=4)
        self.assertAlmostEqual(_radius_of(self.editor, robot2_sphere), 0.05, places=4)

    async def test_scale_spheres_records_operation(self) -> None:
        """Test scale spheres records operation."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.1)
        ops_before = len(self.editor._operations)
        self.editor.scale_spheres(_LINK1_PATH, 2.0)
        self.assertEqual(len(self.editor._operations), ops_before + 1)

    async def test_scale_spheres_scales_nested_links_from_articulation_root(self) -> None:
        """Test Scale All still reaches links nested inside another link."""
        base_sphere = self.editor.add_sphere(_BASE_LINK_PATH, np.zeros(3), 0.10)
        arm_sphere = self.editor.add_sphere(_ARM_LINK_PATH, np.zeros(3), 0.20)

        self.editor.scale_spheres(_ROBOT_PATH, 2.0)

        self.assertAlmostEqual(_radius_of(self.editor, base_sphere), 0.20, places=4)
        self.assertAlmostEqual(_radius_of(self.editor, arm_sphere), 0.40, places=4)

    # -------------------------------------------------------------------------
    # scale_link_spheres
    # -------------------------------------------------------------------------

    async def test_scale_link_spheres_doubles_radius(self) -> None:
        """Test scale link spheres scales the target link."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.1)
        self.editor.scale_link_spheres(_LINK1_PATH, 2.0)
        self.assertAlmostEqual(_radius_of(self.editor, sphere_path), 0.2, places=4)

    async def test_scale_link_spheres_keeps_nested_link_unscaled(self) -> None:
        """Test scaling a link leaves a link nested inside it at its authored radius."""
        # This is the per-link Scale button, which must not resize a nested link the user
        # did not select. `scale_spheres` keeps subtree semantics for Scale All.
        base_sphere = self.editor.add_sphere(_BASE_LINK_PATH, np.zeros(3), 0.10)
        arm_sphere = self.editor.add_sphere(_ARM_LINK_PATH, np.zeros(3), 0.20)

        self.editor.scale_link_spheres(_BASE_LINK_PATH, 3.0)

        self.assertAlmostEqual(_radius_of(self.editor, base_sphere), 0.30, places=4)
        self.assertAlmostEqual(_radius_of(self.editor, arm_sphere), 0.20, places=4)

    async def test_scale_link_spheres_keeps_prefix_sibling_links_unscaled(self) -> None:
        """Test scale link spheres skips links whose names extend the target name."""
        target_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.10)
        link10_path = self.editor.add_sphere(_LINK10_PATH, np.zeros(3), 0.07)

        self.editor.scale_link_spheres(_LINK1_PATH, 3.0)

        self.assertAlmostEqual(_radius_of(self.editor, target_path), 0.30, places=4)
        self.assertAlmostEqual(_radius_of(self.editor, link10_path), 0.07, places=4)

    async def test_scale_link_spheres_tolerates_trailing_slash(self) -> None:
        """Test scale link spheres accepts a link path with a trailing slash."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.1)
        self.editor.scale_link_spheres(_LINK1_PATH + "/", 2.0)
        self.assertAlmostEqual(_radius_of(self.editor, sphere_path), 0.2, places=4)

    async def test_scale_link_spheres_records_operation(self) -> None:
        """Test scale link spheres records one undo entry."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.1)
        ops_before = len(self.editor._operations)
        self.editor.scale_link_spheres(_LINK1_PATH, 2.0)
        self.assertEqual(len(self.editor._operations), ops_before + 1)

    # -------------------------------------------------------------------------
    # undo / redo
    # -------------------------------------------------------------------------

    async def test_undo_add_removes_sphere(self) -> None:
        """Test undo add removes sphere."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.undo()
        self.assertNotIn(sphere_path, self.editor.path_2_spheres)
        prim = prim_utils.get_prim_at_path(sphere_path)
        self.assertFalse(prim and prim.IsValid())

    async def test_undo_add_populates_redo_stack(self) -> None:
        """Test undo add populates redo stack."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.undo()
        self.assertEqual(len(self.editor._redo), 1)

    async def test_redo_after_undo_add_restores_sphere(self) -> None:
        """Test redo after undo add restores sphere."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.undo()
        self.editor.redo()
        self.assertIn(sphere_path, self.editor.path_2_spheres)
        prim = prim_utils.get_prim_at_path(sphere_path)
        self.assertTrue(prim and prim.IsValid())

    async def test_undo_scale_restores_original_radius(self) -> None:
        """Test undo scale restores original radius."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.1)
        self.editor.scale_spheres(_LINK1_PATH, 3.0)
        self.editor.undo()
        radius = self.editor.path_2_spheres[sphere_path].get_radii().numpy()[0]
        self.assertAlmostEqual(radius, 0.1, places=4)

    async def test_undo_clear_spheres_restores_sphere(self) -> None:
        """Test undo clear spheres restores sphere."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.clear_spheres()
        self.editor.undo()
        self.assertIn(sphere_path, self.editor.path_2_spheres)
        prim = prim_utils.get_prim_at_path(sphere_path)
        self.assertTrue(prim and prim.IsValid())

    async def test_undo_on_empty_stack_is_safe(self) -> None:
        """Test undo on empty stack is safe."""
        self.editor.undo()
        self.assertEqual(len(self.editor.path_2_spheres), 0)

    async def test_redo_on_empty_stack_is_safe(self) -> None:
        """Test redo on empty stack is safe."""
        self.editor.redo()
        self.assertEqual(len(self.editor.path_2_spheres), 0)

    async def test_add_sphere_clears_redo_stack(self) -> None:
        """Test add sphere clears redo stack."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.undo()
        self.assertEqual(len(self.editor._redo), 1)
        self.editor.add_sphere(_LINK1_PATH, np.array([0.1, 0.0, 0.0]), 0.05)
        # New add should clear the redo stack
        self.assertEqual(len(self.editor._redo), 0)

    # -------------------------------------------------------------------------
    # interpolate_spheres
    # -------------------------------------------------------------------------

    async def test_interpolate_spheres_creates_correct_count(self) -> None:
        """Test interpolate spheres creates correct count."""
        path1 = self.editor.add_sphere(_LINK1_PATH, np.array([0.0, 0.0, 0.0]), 0.05)
        path2 = self.editor.add_sphere(_LINK1_PATH, np.array([0.6, 0.0, 0.0]), 0.05)
        self.editor.interpolate_spheres(path1, path2, num_spheres=3)
        self.assertEqual(len(self.editor.path_2_spheres), 5)  # 2 original + 3 interpolated

    async def test_interpolate_spheres_positions_between_endpoints(self) -> None:
        """Test interpolate spheres positions between endpoints."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        path1 = self.editor.add_sphere(_LINK1_PATH, p0, 0.05)
        path2 = self.editor.add_sphere(_LINK1_PATH, p1, 0.05)
        self.editor.interpolate_spheres(path1, path2, num_spheres=1)
        # Collect interpolated sphere paths (those that are not path1 or path2)
        interp_paths = [p for p in self.editor.path_2_spheres if p not in (path1, path2)]
        self.assertEqual(len(interp_paths), 1)
        interp_sphere = self.editor.path_2_spheres[interp_paths[0]]
        center = interp_sphere.get_local_poses()[0].numpy()[0]
        # Midpoint should be between 0 and 1 on X axis
        self.assertGreater(center[0], 0.0)
        self.assertLess(center[0], 1.0)

    # -------------------------------------------------------------------------
    # set_sphere_colors
    # -------------------------------------------------------------------------

    async def test_set_sphere_colors_updates_filter(self) -> None:
        """Test set sphere colors updates filter."""
        self.editor.set_sphere_colors(_LINK1_PATH)
        self.assertEqual(self.editor.filter, _LINK1_PATH)

    async def test_set_sphere_colors_updates_color_in(self) -> None:
        """Test set sphere colors updates color in."""
        color = np.array([1.0, 0.0, 0.0])
        self.editor.set_sphere_colors(_LINK1_PATH, color_in=color)
        np.testing.assert_array_equal(self.editor.filter_in_sphere_color, color)

    async def test_set_sphere_colors_updates_color_out(self) -> None:
        """Test set sphere colors updates color out."""
        color = np.array([0.0, 1.0, 0.0])
        self.editor.set_sphere_colors(_LINK1_PATH, color_out=color)
        np.testing.assert_array_equal(self.editor.filter_out_sphere_color, color)

    async def test_set_sphere_colors_skips_prefix_sibling_links(self) -> None:
        """Test set sphere colors leaves links whose names extend the target name unselected."""
        # Selecting a link colors its spheres so the user can see which ones they are about to
        # clear or scale, so a prefix sibling colored as selected misstates that.
        target_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        link10_path = self.editor.add_sphere(_LINK10_PATH, np.zeros(3), 0.05)
        link1_tip_path = self.editor.add_sphere(_LINK1_TIP_PATH, np.zeros(3), 0.05)

        self.editor.set_sphere_colors(_LINK1_PATH)

        np.testing.assert_allclose(_color_of(self.editor, target_path), self.editor.filter_in_sphere_color, atol=1e-6)
        for sibling_path in (link10_path, link1_tip_path):
            np.testing.assert_allclose(
                _color_of(self.editor, sibling_path), self.editor.filter_out_sphere_color, atol=1e-6
            )

    async def test_set_sphere_colors_empty_filter_selects_every_sphere(self) -> None:
        """Test the default empty filter still colors every sphere as selected."""
        link1_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        link2_path = self.editor.add_sphere(_LINK2_PATH, np.zeros(3), 0.05)

        self.editor.set_sphere_colors("")

        for sphere_path in (link1_path, link2_path):
            np.testing.assert_allclose(
                _color_of(self.editor, sphere_path), self.editor.filter_in_sphere_color, atol=1e-6
            )

    async def test_add_sphere_colors_prefix_sibling_as_unselected(self) -> None:
        """Test a sphere added under a prefix sibling is not colored as the selected link."""
        self.editor.set_sphere_colors(_LINK1_PATH)

        target_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        sibling_path = self.editor.add_sphere(_LINK10_PATH, np.zeros(3), 0.05)

        np.testing.assert_allclose(_color_of(self.editor, target_path), self.editor.filter_in_sphere_color, atol=1e-6)
        np.testing.assert_allclose(_color_of(self.editor, sibling_path), self.editor.filter_out_sphere_color, atol=1e-6)

    async def test_set_sphere_colors_marks_nested_link_as_unselected(self) -> None:
        """Test selecting a link does not claim the spheres of a link nested inside it."""
        base_sphere = self.editor.add_sphere(_BASE_LINK_PATH, np.zeros(3), 0.05)
        arm_sphere = self.editor.add_sphere(_ARM_LINK_PATH, np.zeros(3), 0.05)

        self.editor.set_sphere_colors(_BASE_LINK_PATH)

        np.testing.assert_allclose(_color_of(self.editor, base_sphere), self.editor.filter_in_sphere_color, atol=1e-6)
        np.testing.assert_allclose(_color_of(self.editor, arm_sphere), self.editor.filter_out_sphere_color, atol=1e-6)

    async def test_set_sphere_colors_selects_nested_link_spheres(self) -> None:
        """Test selecting a nested link colors its own spheres as selected."""
        base_sphere = self.editor.add_sphere(_BASE_LINK_PATH, np.zeros(3), 0.05)
        arm_sphere = self.editor.add_sphere(_ARM_LINK_PATH, np.zeros(3), 0.05)

        self.editor.set_sphere_colors(_ARM_LINK_PATH)

        np.testing.assert_allclose(_color_of(self.editor, arm_sphere), self.editor.filter_in_sphere_color, atol=1e-6)
        np.testing.assert_allclose(_color_of(self.editor, base_sphere), self.editor.filter_out_sphere_color, atol=1e-6)

    # -------------------------------------------------------------------------
    # write_spheres_to_dict
    # -------------------------------------------------------------------------

    async def test_write_spheres_to_dict_populates_link_key(self) -> None:
        """Test write spheres to dict populates link key."""
        self.editor.add_sphere(_LINK1_PATH, np.array([0.1, 0.2, 0.3]), 0.05)
        link_to_spheres = {}
        self.editor.write_spheres_to_dict(_ROBOT_PATH, link_to_spheres)
        self.assertIn("link1", link_to_spheres)

    async def test_write_spheres_to_dict_correct_radius(self) -> None:
        """Test write spheres to dict correct radius."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.07)
        link_to_spheres = {}
        self.editor.write_spheres_to_dict(_ROBOT_PATH, link_to_spheres)
        sphere_data = link_to_spheres["link1"][0]
        self.assertAlmostEqual(sphere_data["radius"], 0.07, places=2)

    async def test_write_spheres_to_dict_multiple_links(self) -> None:
        """Test write spheres to dict multiple links."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.add_sphere(_LINK2_PATH, np.zeros(3), 0.05)
        link_to_spheres = {}
        self.editor.write_spheres_to_dict(_ROBOT_PATH, link_to_spheres)
        self.assertIn("link1", link_to_spheres)
        self.assertIn("link2", link_to_spheres)

    # -------------------------------------------------------------------------
    # load_xrdf_spheres
    # -------------------------------------------------------------------------

    async def test_load_xrdf_spheres_v1_creates_spheres(self) -> None:
        """Test load xrdf spheres v1 creates spheres."""
        parsed = {
            "format_version": 1.0,
            "collision": {"geometry": "default"},
            "geometry": {
                "default": {
                    "spheres": {
                        "link1": [{"center": [0.0, 0.0, 0.1], "radius": 0.05}],
                    }
                }
            },
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)
        self.assertEqual(len(self.editor.path_2_spheres), 1)
        sphere_path = next(iter(self.editor.path_2_spheres))
        self.assertTrue(sphere_path.startswith(_LINK1_PATH))

    async def test_load_xrdf_spheres_v2_creates_spheres(self) -> None:
        """Test load xrdf spheres v2 creates spheres."""
        parsed = {
            "format_version": 2.0,
            "world_collision": {"geometry": "default"},
            "geometry": {
                "default": {
                    "spheres": {
                        "link1": [{"center": [0.0, 0.0, 0.1], "radius": 0.05}],
                        "link2": [{"center": [0.1, 0.0, 0.0], "radius": 0.04}],
                    }
                }
            },
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)
        self.assertEqual(len(self.editor.path_2_spheres), 2)

    async def test_load_xrdf_spheres_clears_existing_spheres(self) -> None:
        """Test load xrdf spheres clears existing spheres."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        parsed = {
            "format_version": 1.0,
            "collision": {"geometry": "default"},
            "geometry": {
                "default": {
                    "spheres": {
                        "link2": [{"center": [0.1, 0.0, 0.0], "radius": 0.04}],
                    }
                }
            },
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)
        # Only the freshly loaded sphere should remain
        self.assertEqual(len(self.editor.path_2_spheres), 1)
        sphere_path = next(iter(self.editor.path_2_spheres))
        self.assertTrue(sphere_path.startswith(_LINK2_PATH))

    async def test_load_xrdf_spheres_resets_undo_history(self) -> None:
        """Test load xrdf spheres resets undo history."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        parsed = {
            "format_version": 1.0,
            "collision": {"geometry": "default"},
            "geometry": {
                "default": {
                    "spheres": {
                        "link1": [{"center": [0.0, 0.0, 0.1], "radius": 0.05}],
                    }
                }
            },
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)
        # After load, only the single ADD operation from the load should exist
        self.assertEqual(len(self.editor._operations), 1)
        self.assertEqual(self.editor._operations[0][0], "ADD")

    async def test_load_xrdf_spheres_missing_collision_key_is_safe(self) -> None:
        """Test load xrdf spheres missing collision key is safe."""
        parsed = {"format_version": 1.0}
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)
        self.assertEqual(len(self.editor.path_2_spheres), 0)

    # -------------------------------------------------------------------------
    # on_shutdown
    # -------------------------------------------------------------------------

    async def test_on_shutdown_removes_all_spheres(self) -> None:
        """Test on shutdown removes all spheres."""
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.add_sphere(_LINK2_PATH, np.ones(3) * 0.1, 0.05)
        self.editor.on_shutdown()
        self.assertEqual(len(self.editor.path_2_spheres), 0)

    # -------------------------------------------------------------------------
    # clear_preview
    # -------------------------------------------------------------------------

    async def test_clear_preview_does_not_affect_regular_spheres(self) -> None:
        """Test clear preview does not affect regular spheres."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.clear_preview()
        self.assertIn(sphere_path, self.editor.path_2_spheres)
        self.assertEqual(len(self.editor._preview_spheres), 0)

    # -------------------------------------------------------------------------
    # save_spheres / load_spheres round-trip (Lula robot description YAML)
    # -------------------------------------------------------------------------

    async def test_save_spheres_load_spheres_round_trip_preserves_data(self) -> None:
        """Test save spheres load spheres round trip preserves data."""
        # Authored data — distinct centers and radii so each sphere is uniquely
        # identifiable after the load.
        authored = [
            (_LINK1_PATH, np.array([0.10, 0.20, 0.30]), 0.050),
            (_LINK1_PATH, np.array([0.40, 0.50, 0.60]), 0.070),
            (_LINK2_PATH, np.array([0.01, 0.02, 0.03]), 0.030),
        ]
        for link_path, center, radius in authored:
            self.editor.add_sphere(link_path, center, radius)

        tmp_dir = tempfile.mkdtemp()
        try:
            yaml_path = os.path.join(tmp_dir, "robot_description.yaml")
            with open(yaml_path, "w") as f:
                self.editor.save_spheres(_ROBOT_PATH, f)

            self.editor.clear_spheres()
            self.assertEqual(len(self.editor.path_2_spheres), 0)

            self.editor.load_spheres(_ROBOT_PATH, yaml_path)

            self.assertEqual(len(self.editor.path_2_spheres), len(authored))

            # Reconstruct {link_path: [(center, radius), ...]} from the loaded
            # spheres so the round-trip can be checked independently of the
            # paths the editor assigns.
            loaded_by_link: dict[str, list[tuple[np.ndarray, float]]] = {}
            for sphere_path, sphere in self.editor.path_2_spheres.items():
                link_path = sphere_path.rsplit("/", 1)[0]
                center = sphere.get_local_poses()[0].numpy()[0]
                radius = float(sphere.get_radii().numpy()[0])
                loaded_by_link.setdefault(link_path, []).append((center, radius))

            for link_path, expected_center, expected_radius in authored:
                # save_spheres rounds to 5 decimals; compare with the same
                # precision.
                match = any(
                    np.allclose(c, expected_center, atol=1e-4) and abs(r - expected_radius) <= 1e-4
                    for c, r in loaded_by_link.get(link_path, [])
                )
                self.assertTrue(
                    match,
                    f"Round-trip lost sphere ({expected_center}, r={expected_radius}) on {link_path}; "
                    f"loaded={loaded_by_link.get(link_path)}",
                )
        finally:
            try:
                os.remove(yaml_path)
            except OSError:
                pass
            os.rmdir(tmp_dir)

    async def test_save_spheres_skips_spheres_outside_robot_path(self) -> None:
        """Test save spheres skips spheres outside robot path."""
        # Sphere under the robot path is included; a sphere created under a
        # foreign root is skipped (warning is logged, not asserted).
        stage_utils.define_prim("/World/other_robot", "Xform")
        stage_utils.define_prim("/World/other_robot/foreign_link", "Xform")
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.add_sphere("/World/other_robot/foreign_link", np.zeros(3), 0.05)

        buf = io.StringIO()
        self.editor.save_spheres(_ROBOT_PATH, buf)
        output = buf.getvalue()

        self.assertIn("link1", output)
        self.assertNotIn("foreign_link", output)
        self.assertNotIn("other_robot", output)

    async def test_save_spheres_skips_prefix_colliding_robot(self) -> None:
        """Test save spheres excludes a robot whose root extends the exported root name."""
        # Regression: `/World/robot2/link1/...` passed the containment guard for
        # `/World/robot`, and the surviving path slice emitted a corrupt `- /link1:` key.
        stage_utils.define_prim(_ROBOT2_PATH, "Xform")
        stage_utils.define_prim(_ROBOT2_LINK1_PATH, "Xform")
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.10)
        self.editor.add_sphere(_ROBOT2_LINK1_PATH, np.zeros(3), 0.05)

        buf = io.StringIO()
        self.editor.save_spheres(_ROBOT_PATH, buf)
        output = buf.getvalue()

        self.assertIn("  - link1:", output)
        self.assertNotIn("- /link1:", output)
        self.assertEqual(output.count("radius"), 1, "only the exported robot's sphere should be written")

    async def test_write_spheres_to_dict_skips_prefix_colliding_robot(self) -> None:
        """Test write spheres to dict excludes a robot whose root extends the exported root."""
        stage_utils.define_prim(_ROBOT2_PATH, "Xform")
        stage_utils.define_prim(_ROBOT2_LINK1_PATH, "Xform")
        self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.10)
        self.editor.add_sphere(_ROBOT2_LINK1_PATH, np.zeros(3), 0.05)

        link_to_spheres = {}
        self.editor.write_spheres_to_dict(_ROBOT_PATH, link_to_spheres)

        self.assertEqual(sorted(link_to_spheres.keys()), ["link1"])
        # Key set alone would still pass if both robots collapsed into one `link1` entry.
        self.assertEqual(len(link_to_spheres["link1"]), 1)
        self.assertAlmostEqual(link_to_spheres["link1"][0]["radius"], 0.10, places=4)

    # -------------------------------------------------------------------------
    # Nested links: export keys and round trip
    # -------------------------------------------------------------------------

    async def test_write_spheres_to_dict_keys_nested_link_by_name(self) -> None:
        """Test a nested link is exported under its link name, not its path fragment."""
        # Deriving the key by slicing the path relative to the articulation root writes
        # "base_link/arm_link", which does not name any link in the robot description.
        self.editor.add_sphere(_ARM_LINK_PATH, np.zeros(3), 0.05)

        link_to_spheres = {}
        self.editor.write_spheres_to_dict(_ROBOT_PATH, link_to_spheres)

        self.assertEqual(sorted(link_to_spheres.keys()), ["arm_link"])

    async def test_save_spheres_keys_nested_link_by_name(self) -> None:
        """Test the Lula export writes a nested link under its link name."""
        self.editor.add_sphere(_ARM_LINK_PATH, np.zeros(3), 0.05)

        buf = io.StringIO()
        self.editor.save_spheres(_ROBOT_PATH, buf)
        output = buf.getvalue()

        self.assertIn("  - arm_link:", output)
        self.assertNotIn("base_link/arm_link", output)

    async def test_write_spheres_to_dict_separates_parent_and_nested_link(self) -> None:
        """Test parent and nested link spheres are exported under separate keys."""
        self.editor.add_sphere(_BASE_LINK_PATH, np.zeros(3), 0.10)
        self.editor.add_sphere(_ARM_LINK_PATH, np.zeros(3), 0.05)

        link_to_spheres = {}
        self.editor.write_spheres_to_dict(_ROBOT_PATH, link_to_spheres)

        self.assertEqual(sorted(link_to_spheres.keys()), ["arm_link", "base_link"])
        self.assertAlmostEqual(link_to_spheres["base_link"][0]["radius"], 0.10, places=4)
        self.assertAlmostEqual(link_to_spheres["arm_link"][0]["radius"], 0.05, places=4)

    async def test_links_sharing_a_prim_name_export_under_distinct_keys(self) -> None:
        """Test two links whose prims share a name do not collapse into one export key.

        Prim names are not unique across an articulation: `/robot/arm_a/tool` and
        `/robot/arm_b/tool` are separate links that PhysX reports as `tool` and
        `tool_0`. Keying the export on the prim name merges both links' spheres
        into one entry, so one link loses its collision geometry on reload.
        """
        arm_a_tool = "/World/robot/arm_a/tool"
        arm_b_tool = "/World/robot/arm_b/tool"
        for path in ("/World/robot/arm_a", arm_a_tool, "/World/robot/arm_b", arm_b_tool):
            stage_utils.define_prim(path, "Xform")
        self.editor.set_link_names({arm_a_tool: "tool", arm_b_tool: "tool_0"})

        self.editor.add_sphere(arm_a_tool, np.zeros(3), 0.11)
        self.editor.add_sphere(arm_b_tool, np.zeros(3), 0.22)

        link_to_spheres = {}
        self.editor.write_spheres_to_dict(_ROBOT_PATH, link_to_spheres)

        self.assertEqual(sorted(link_to_spheres.keys()), ["tool", "tool_0"])
        self.assertAlmostEqual(link_to_spheres["tool"][0]["radius"], 0.11, places=4)
        self.assertAlmostEqual(link_to_spheres["tool_0"][0]["radius"], 0.22, places=4)

    async def test_links_sharing_a_prim_name_round_trip_to_their_own_link(self) -> None:
        """Test each link keeps its own spheres across a save/load cycle."""
        arm_a_tool = "/World/robot/arm_a/tool"
        arm_b_tool = "/World/robot/arm_b/tool"
        for path in ("/World/robot/arm_a", arm_a_tool, "/World/robot/arm_b", arm_b_tool):
            stage_utils.define_prim(path, "Xform")
        self.editor.set_link_names({arm_a_tool: "tool", arm_b_tool: "tool_0"})

        self.editor.add_sphere(arm_a_tool, np.zeros(3), 0.11)
        self.editor.add_sphere(arm_b_tool, np.zeros(3), 0.22)

        tmp_dir = tempfile.mkdtemp()
        try:
            yaml_path = os.path.join(tmp_dir, "robot_description.yaml")
            with open(yaml_path, "w") as f:
                self.editor.save_spheres(_ROBOT_PATH, f)

            self.editor.clear_spheres()
            self.editor.load_spheres(_ROBOT_PATH, yaml_path)

            radii_by_link: dict[str, list[float]] = {}
            for sphere_path in self.editor.path_2_spheres:
                radii_by_link.setdefault(sphere_path.rsplit("/", 1)[0], []).append(_radius_of(self.editor, sphere_path))

            self.assertEqual(sorted(radii_by_link), [arm_a_tool, arm_b_tool])
            self.assertAlmostEqual(radii_by_link[arm_a_tool][0], 0.11, places=4)
            self.assertAlmostEqual(radii_by_link[arm_b_tool][0], 0.22, places=4)
        finally:
            try:
                os.remove(yaml_path)
            except OSError:
                pass
            os.rmdir(tmp_dir)

    async def test_save_load_round_trip_restores_nested_link_spheres(self) -> None:
        """Test a nested link's spheres survive a save/load cycle on the same link.

        Exporting by link name is only useful if the importer can map that name back onto a
        link that is not a direct child of the articulation root.
        """
        self.editor.add_sphere(_BASE_LINK_PATH, np.array([0.10, 0.0, 0.0]), 0.10)
        self.editor.add_sphere(_ARM_LINK_PATH, np.array([0.0, 0.20, 0.0]), 0.05)

        tmp_dir = tempfile.mkdtemp()
        try:
            yaml_path = os.path.join(tmp_dir, "robot_description.yaml")
            with open(yaml_path, "w") as f:
                self.editor.save_spheres(_ROBOT_PATH, f)

            self.editor.clear_spheres()
            self.editor.load_spheres(_ROBOT_PATH, yaml_path)

            loaded_links = {path.rsplit("/", 1)[0] for path in self.editor.path_2_spheres}
            self.assertEqual(loaded_links, {_BASE_LINK_PATH, _ARM_LINK_PATH})
        finally:
            try:
                os.remove(yaml_path)
            except OSError:
                pass
            os.rmdir(tmp_dir)

    # -------------------------------------------------------------------------
    # load_spheres error handling (Lula robot description YAML)
    # -------------------------------------------------------------------------

    async def test_load_spheres_handles_malformed_yaml(self) -> None:
        """Test load spheres handles malformed yaml."""
        # Regression: previously raised UnboundLocalError because `parsed_file`
        # was never assigned on the YAMLError branch.
        tmp_dir = tempfile.mkdtemp()
        try:
            yaml_path = os.path.join(tmp_dir, "broken.yaml")
            with open(yaml_path, "w") as f:
                f.write("collision_spheres: [unterminated\n")

            # Should not raise.
            self.editor.load_spheres(_ROBOT_PATH, yaml_path)
            self.assertEqual(len(self.editor.path_2_spheres), 0)
        finally:
            try:
                os.remove(yaml_path)
            except OSError:
                pass
            os.rmdir(tmp_dir)

    async def test_load_spheres_handles_non_mapping_yaml_root(self) -> None:
        """Test load spheres handles non mapping yaml root."""
        # An empty or list-rooted YAML file is not a robot description; should
        # bail out cleanly rather than raising AttributeError on `.get()`.
        tmp_dir = tempfile.mkdtemp()
        try:
            yaml_path = os.path.join(tmp_dir, "list_root.yaml")
            with open(yaml_path, "w") as f:
                f.write("- this is a list, not a mapping\n")

            self.editor.load_spheres(_ROBOT_PATH, yaml_path)
            self.assertEqual(len(self.editor.path_2_spheres), 0)
        finally:
            try:
                os.remove(yaml_path)
            except OSError:
                pass
            os.rmdir(tmp_dir)

    # -------------------------------------------------------------------------
    # interpolate_spheres additional branches
    # -------------------------------------------------------------------------

    async def test_interpolate_spheres_skips_when_endpoints_under_different_links(self) -> None:
        """Test interpolate spheres skips when endpoints under different links."""
        # Regression: the function logged a warning then continued, producing
        # interpolated spheres on path1's link even though path2 lived on a
        # different link. After the fix it must return without adding spheres.
        path_link1 = self.editor.add_sphere(_LINK1_PATH, np.array([0.0, 0.0, 0.0]), 0.05)
        path_link2 = self.editor.add_sphere(_LINK2_PATH, np.array([1.0, 0.0, 0.0]), 0.05)
        count_before = len(self.editor.path_2_spheres)

        self.editor.interpolate_spheres(path_link1, path_link2, num_spheres=3)

        self.assertEqual(len(self.editor.path_2_spheres), count_before)

    async def test_interpolate_spheres_general_radius_branch(self) -> None:
        """Test interpolate spheres general radius branch."""
        # Exercises the `relative_offsets = (rads - rad_1) / (rad_2 - rad_1)`
        # branch — only reached when the two endpoint radii differ enough.
        path1 = self.editor.add_sphere(_LINK1_PATH, np.array([0.0, 0.0, 0.0]), 0.02)
        path2 = self.editor.add_sphere(_LINK1_PATH, np.array([1.0, 0.0, 0.0]), 0.20)

        self.editor.interpolate_spheres(path1, path2, num_spheres=2)

        # Two interpolated spheres added between the endpoints.
        interp_paths = [p for p in self.editor.path_2_spheres if p not in (path1, path2)]
        self.assertEqual(len(interp_paths), 2)

        # Their centers must lie strictly between the endpoints on X, and the
        # interpolated radii must lie strictly between the endpoint radii.
        for p in interp_paths:
            sphere = self.editor.path_2_spheres[p]
            center = sphere.get_local_poses()[0].numpy()[0]
            radius = float(sphere.get_radii().numpy()[0])
            self.assertGreater(center[0], 0.0)
            self.assertLess(center[0], 1.0)
            self.assertGreater(radius, 0.02)
            self.assertLess(radius, 0.20)

    async def test_interpolate_spheres_invalid_path_is_safe(self) -> None:
        """Test interpolate spheres invalid path is safe."""
        # Either invalid endpoint should early-return without raising.
        path_valid = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        count_before = len(self.editor.path_2_spheres)
        self.editor.interpolate_spheres(path_valid, "/World/robot/link1/does_not_exist", num_spheres=2)
        self.editor.interpolate_spheres("/World/robot/link1/does_not_exist", path_valid, num_spheres=2)
        self.assertEqual(len(self.editor.path_2_spheres), count_before)

    # -------------------------------------------------------------------------
    # load_xrdf_spheres — clone traversal in _get_sphere_list_from_xrdf_geometries
    # -------------------------------------------------------------------------

    async def test_load_xrdf_spheres_clone_single_level(self) -> None:
        """Test load xrdf spheres clone single level."""
        parsed = {
            "format_version": 1.0,
            "collision": {"geometry": "primary"},
            "geometry": {
                "primary": {
                    "spheres": {"link1": [{"center": [0.1, 0.0, 0.0], "radius": 0.05}]},
                    "clone": ["extra"],
                },
                "extra": {
                    "spheres": {"link2": [{"center": [0.2, 0.0, 0.0], "radius": 0.06}]},
                },
            },
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)

        # Spheres from both the primary group and its clone should be present.
        paths = list(self.editor.path_2_spheres.keys())
        self.assertEqual(len(paths), 2)
        self.assertTrue(any(p.startswith(_LINK1_PATH) for p in paths))
        self.assertTrue(any(p.startswith(_LINK2_PATH) for p in paths))

    async def test_load_xrdf_spheres_clone_transitive(self) -> None:
        """Test load xrdf spheres clone transitive."""
        # primary -> mid -> leaf chain. Leaf's spheres should land on the stage.
        parsed = {
            "format_version": 1.0,
            "collision": {"geometry": "primary"},
            "geometry": {
                "primary": {
                    "spheres": {"link1": [{"center": [0.1, 0.0, 0.0], "radius": 0.05}]},
                    "clone": ["mid"],
                },
                "mid": {
                    "spheres": {},
                    "clone": ["leaf"],
                },
                "leaf": {
                    "spheres": {"link2": [{"center": [0.2, 0.0, 0.0], "radius": 0.06}]},
                },
            },
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)

        paths = list(self.editor.path_2_spheres.keys())
        self.assertEqual(len(paths), 2)
        self.assertTrue(any(p.startswith(_LINK2_PATH) for p in paths))

    async def test_load_xrdf_spheres_clone_cycle_does_not_loop_forever(self) -> None:
        """Test load xrdf spheres clone cycle does not loop forever."""
        # primary -> other -> primary cycle. The handled_groups guard must
        # break the cycle (regression scaffolding: if the guard is ever
        # removed, this test will hang the runner).
        parsed = {
            "format_version": 1.0,
            "collision": {"geometry": "primary"},
            "geometry": {
                "primary": {
                    "spheres": {"link1": [{"center": [0.1, 0.0, 0.0], "radius": 0.05}]},
                    "clone": ["other"],
                },
                "other": {
                    "spheres": {"link2": [{"center": [0.2, 0.0, 0.0], "radius": 0.06}]},
                    "clone": ["primary"],
                },
            },
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)

        self.assertEqual(len(self.editor.path_2_spheres), 2)

    async def test_load_xrdf_spheres_clone_missing_target_is_safe(self) -> None:
        """Test load xrdf spheres clone missing target is safe."""
        # Referencing a non-existent clone group should be silently skipped,
        # not crash.
        parsed = {
            "format_version": 1.0,
            "collision": {"geometry": "primary"},
            "geometry": {
                "primary": {
                    "spheres": {"link1": [{"center": [0.1, 0.0, 0.0], "radius": 0.05}]},
                    "clone": ["nonexistent_group"],
                },
            },
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)
        self.assertEqual(len(self.editor.path_2_spheres), 1)

    # -------------------------------------------------------------------------
    # load_xrdf_spheres — format_version / buffer_distance branches
    # -------------------------------------------------------------------------

    async def test_load_xrdf_spheres_unsupported_format_version_is_safe(self) -> None:
        """Test load xrdf spheres unsupported format version is safe."""
        parsed = {
            "format_version": 3.0,
            "world_collision": {"geometry": "default"},
            "geometry": {
                "default": {"spheres": {"link1": [{"center": [0.0, 0.0, 0.0], "radius": 0.05}]}},
            },
        }
        # Must not raise and must not place any spheres on the stage.
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)
        self.assertEqual(len(self.editor.path_2_spheres), 0)

    async def test_load_xrdf_spheres_buffer_distance_inflates_only_targeted_link(self) -> None:
        """Test load xrdf spheres buffer distance inflates only targeted link."""
        # Regression: previously used a bare prefix compare, so a buffer
        # distance keyed on "link1" also affected sibling links whose names
        # *start* with "link1" (e.g. "link10").
        stage_utils.define_prim("/World/robot/link10", "Xform")

        base_radius_link1 = 0.05
        base_radius_link10 = 0.07
        buffer = 0.02

        parsed = {
            "format_version": 2.0,
            "world_collision": {
                "geometry": "default",
                "buffer_distance": {"link1": buffer},
            },
            "geometry": {
                "default": {
                    "spheres": {
                        "link1": [{"center": [0.0, 0.0, 0.0], "radius": base_radius_link1}],
                        "link10": [{"center": [0.0, 0.0, 0.0], "radius": base_radius_link10}],
                    }
                }
            },
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)

        link1_radii = []
        link10_radii = []
        for p, sphere in self.editor.path_2_spheres.items():
            if p.startswith("/World/robot/link1/"):
                link1_radii.append(float(sphere.get_radii().numpy()[0]))
            elif p.startswith("/World/robot/link10/"):
                link10_radii.append(float(sphere.get_radii().numpy()[0]))

        self.assertEqual(len(link1_radii), 1)
        self.assertEqual(len(link10_radii), 1)
        self.assertAlmostEqual(link1_radii[0], base_radius_link1 + buffer, places=4)
        # `link10` must be unaffected by the buffer keyed on `link1`.
        self.assertAlmostEqual(link10_radii[0], base_radius_link10, places=4)

    async def test_load_xrdf_spheres_prefers_the_link_over_a_same_named_prim(self) -> None:
        """Test a non-link prim sharing a link's name does not capture the spheres.

        Resolving a link key by searching the stage for a matching prim name can
        return a visual or grouping prim authored earlier in the subtree. Spheres
        then hang off geometry whose frame is not the link's, so the exported
        collision geometry is silently in the wrong place.
        """
        stage_utils.define_prim("/World/robot/Looks", "Scope")
        stage_utils.define_prim("/World/robot/Looks/arm_link", "Xform")

        parsed = {
            "format_version": 2.0,
            "world_collision": {"geometry": "default"},
            "geometry": {"default": {"spheres": {"arm_link": [{"center": [0.0, 0.0, 0.0], "radius": 0.05}]}}},
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)

        sphere_paths = list(self.editor.path_2_spheres)
        self.assertEqual(len(sphere_paths), 1)
        self.assertTrue(sphere_paths[0].startswith(_ARM_LINK_PATH + "/"))

    async def test_load_xrdf_spheres_rejects_a_key_naming_a_non_link_prim(self) -> None:
        """Test a key that resolves to a prim which is not a link is refused."""
        stage_utils.define_prim("/World/robot/Looks", "Scope")

        parsed = {
            "format_version": 2.0,
            "world_collision": {"geometry": "default"},
            "geometry": {"default": {"spheres": {"Looks": [{"center": [0.0, 0.0, 0.0], "radius": 0.05}]}}},
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)

        self.assertEqual(len(self.editor.path_2_spheres), 0)

    async def test_load_xrdf_spheres_accepts_legacy_path_fragment_key(self) -> None:
        """Test keys written before 3.7.0 as path fragments still resolve."""
        parsed = {
            "format_version": 2.0,
            "world_collision": {"geometry": "default"},
            "geometry": {"default": {"spheres": {"base_link/arm_link": [{"center": [0.0, 0.0, 0.0], "radius": 0.05}]}}},
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)

        sphere_paths = list(self.editor.path_2_spheres)
        self.assertEqual(len(sphere_paths), 1)
        self.assertTrue(sphere_paths[0].startswith(_ARM_LINK_PATH + "/"))

    async def test_load_xrdf_spheres_buffer_distance_applies_to_nested_link(self) -> None:
        """Test a buffer keyed on a nested link inflates that link, not its parent.

        The nested link's name does not join onto the articulation root, so a
        resolver that only joins paths cannot find it and drops the buffer with
        just a warning.
        """
        base_radius = 0.05
        nested_radius = 0.06
        buffer = 0.02

        parsed = {
            "format_version": 2.0,
            "world_collision": {
                "geometry": "default",
                "buffer_distance": {"arm_link": buffer},
            },
            "geometry": {
                "default": {
                    "spheres": {
                        "base_link": [{"center": [0.0, 0.0, 0.0], "radius": base_radius}],
                        "arm_link": [{"center": [0.0, 0.0, 0.0], "radius": nested_radius}],
                    }
                }
            },
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)

        radii_by_link: dict[str, float] = {}
        for p, sphere in self.editor.path_2_spheres.items():
            radii_by_link[p.rsplit("/", 1)[0]] = float(sphere.get_radii().numpy()[0])

        self.assertEqual(sorted(radii_by_link), sorted([_BASE_LINK_PATH, _ARM_LINK_PATH]))
        self.assertAlmostEqual(radii_by_link[_ARM_LINK_PATH], nested_radius + buffer, places=4)
        self.assertAlmostEqual(radii_by_link[_BASE_LINK_PATH], base_radius, places=4)

    async def test_load_xrdf_spheres_spheres_value_none_is_safe(self) -> None:
        """Test load xrdf spheres spheres value none is safe."""
        # `spheres: null` in YAML parses to None; loader must coerce to {}
        # instead of crashing on `for key, val in None.items()`.
        parsed = {
            "format_version": 1.0,
            "collision": {"geometry": "default"},
            "geometry": {"default": {"spheres": None}},
        }
        self.editor.load_xrdf_spheres(_ROBOT_PATH, parsed)
        self.assertEqual(len(self.editor.path_2_spheres), 0)

    # -------------------------------------------------------------------------
    # redo — DEL and SCALE branches (ADD is covered above)
    # -------------------------------------------------------------------------

    async def test_redo_after_undo_clear_spheres_re_deletes(self) -> None:
        """Test redo after undo clear spheres re deletes."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.05)
        self.editor.clear_spheres()
        # Undo restores the sphere; redo of that DEL op must remove it again.
        self.editor.undo()
        self.assertIn(sphere_path, self.editor.path_2_spheres)
        self.editor.redo()
        self.assertNotIn(sphere_path, self.editor.path_2_spheres)
        prim = prim_utils.get_prim_at_path(sphere_path)
        self.assertFalse(prim and prim.IsValid())

    async def test_redo_after_undo_scale_re_applies_scale(self) -> None:
        """Test redo after undo scale re applies scale."""
        sphere_path = self.editor.add_sphere(_LINK1_PATH, np.zeros(3), 0.1)
        self.editor.scale_spheres(_LINK1_PATH, 3.0)
        # Undo restores 0.1; redo of the SCALE op must re-apply the 3x factor
        # so the final radius is 0.3 again.
        self.editor.undo()
        self.assertAlmostEqual(
            float(self.editor.path_2_spheres[sphere_path].get_radii().numpy()[0]),
            0.1,
            places=4,
        )
        self.editor.redo()
        self.assertAlmostEqual(
            float(self.editor.path_2_spheres[sphere_path].get_radii().numpy()[0]),
            0.3,
            places=4,
        )

    # -------------------------------------------------------------------------
    # generate_spheres — non-triangle mesh rejection
    # -------------------------------------------------------------------------

    async def test_generate_spheres_rejects_non_triangle_mesh(self) -> None:
        """Test generate spheres rejects non triangle mesh."""
        # Quad mesh (vertex count 4 per face) must be rejected before Lula is
        # invoked. The function logs a warning and returns without adding any
        # spheres.
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        face_inds = np.array([0, 1, 2, 3], dtype=np.int32)
        vert_cts = np.array([4], dtype=np.int32)

        self.editor.generate_spheres(
            _LINK1_PATH,
            points,
            face_inds,
            vert_cts,
            num_spheres=4,
            radius_offset=0.0,
            is_preview=False,
        )

        self.assertEqual(len(self.editor.path_2_spheres), 0)
        self.assertEqual(len(self.editor._preview_spheres), 0)

    async def test_generate_spheres_rejects_mixed_face_topology(self) -> None:
        """Test generate spheres rejects mixed face topology."""
        # Mixed triangle + quad face counts is also rejected (the unique check
        # fails before the value check).
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=np.float32,
        )
        face_inds = np.array([0, 1, 4, 1, 2, 3, 4], dtype=np.int32)
        vert_cts = np.array([3, 4], dtype=np.int32)

        self.editor.generate_spheres(
            _LINK1_PATH,
            points,
            face_inds,
            vert_cts,
            num_spheres=4,
            radius_offset=0.0,
            is_preview=False,
        )

        self.assertEqual(len(self.editor.path_2_spheres), 0)
