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

"""Tests for variant routing rule behavior."""

import asyncio
import os
import re
import shutil
import tempfile

import omni.kit.test
from isaacsim.asset.transformer.rules.structure.variants import VariantRoutingRule
from isaacsim.asset.transformer.rules.utils import sanitize_prim_name
from pxr import Gf, Sdf, Usd

from .common import _TEST_DATA_DIR

# Relative path of the canonical UR10 robot asset inside the Isaac Sim assets
# folder. Tests use this path with ``get_assets_root_path_async`` to resolve
# the assets-root prefix at runtime (typically an ``omniverse://`` URL or the
# local SDK install). ``isaacsim.storage.native`` is a test-only dependency;
# import it lazily inside test methods so the tests module still imports
# cleanly when the extension is enabled outside the test runner.
_ISAAC_UR10_PATH = "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

_G1_USD = os.path.join(_TEST_DATA_DIR, "G1", "g1.usda")
_EXCLUDED_VARIANTS = ["none", "default", "physx"]
_ASSET_PATH_RE = re.compile(r"@([^@]+)@")

_EXPECTED_DEPENDENCIES = {
    "left_hand": {
        "inspire_left_base.usda",
        "inspire_left_hand.usda",
        "inspire_left_physics.usda",
        "inspire_left_robot.usda",
        "three_finger_hand_base_left.usda",
        "three_finger_hand_physics_left.usda",
        "three_fingers_left_hand_robot.usda",
        "three_fingers_left_hand.usda",
    },
    "right_hand": {
        "inspire_right_base.usda",
        "inspire_right_hand.usda",
        "inspire_right_physics.usda",
        "inspire_right_robot.usda",
        "three_finger_hand_base_right.usda",
        "three_finger_hand_physics_right.usda",
        "three_fingers_right_hand_robot.usda",
        "three_fingers_right_hand.usda",
    },
    "Thor": {
        "DefaultMaterial.mdl",
        "g1_29dof_NVBP_base.usda",
        "g1_29dof_NVBP_physics.usda",
        "g1_29dof_NVBP_robot.usda",
        "g1_29dof_NVBP_sensor.usda",
        "g1_29dof_NVBP.usda",
    },
}


class TestVariantRoutingRule(omni.kit.test.AsyncTestCase):
    """Async tests for the variant routing rule."""

    async def setUp(self) -> None:
        """Create a temporary directory for test output."""
        self._tmpdir = tempfile.mkdtemp()
        self._success = False

    async def tearDown(self) -> None:
        """Remove temporary directories after successful tests."""
        if self._success:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _variant_set_dir(self, variant_set_name: str) -> str:
        """Get the output directory for a variant set.

        Args:
            variant_set_name: Variant set name to sanitize.

        Returns:
            Absolute path to the variant set output directory.

        """
        return os.path.join(self._tmpdir, "payloads", sanitize_prim_name(variant_set_name))

    def _variant_file_path(self, variant_set_name: str, variant_name: str) -> str:
        """Get the output file path for a specific variant.

        Args:
            variant_set_name: Variant set containing the variant.
            variant_name: Variant option name.

        Returns:
            Absolute path to the variant USDA file.

        """
        variant_file = f"{sanitize_prim_name(variant_name).lower()}.usda"
        return os.path.join(self._variant_set_dir(variant_set_name), variant_file)

    def _layer_asset_paths(self, layer_path: str) -> list[str]:
        """Extract referenced asset paths from a layer.

        Args:
            layer_path: Path to the layer file to scan.

        Returns:
            List of asset paths referenced by the layer.

        """
        layer = Sdf.Layer.FindOrOpen(layer_path)
        self.assertIsNotNone(layer)
        return _ASSET_PATH_RE.findall(layer.ExportToString())

    def _assert_dependency_assets_exist(self, layer_path: str) -> None:
        """Assert dependency assets exist next to a layer.

        Args:
            layer_path: Path to the variant layer file.

        """
        layer_dir = os.path.dirname(layer_path)
        for asset_path in self._layer_asset_paths(layer_path):
            normalized = asset_path.replace("\\", "/")
            if "dependencies/" not in normalized:
                continue
            relative = normalized.lstrip("./")
            abs_path = os.path.normpath(os.path.join(layer_dir, relative))
            self.assertTrue(os.path.exists(abs_path), f"Missing dependency asset: {abs_path}")

    async def test_get_configuration_parameters(self) -> None:
        """Verify configuration parameters are exposed by the rule."""
        stage = Usd.Stage.Open(_G1_USD)
        rule = VariantRoutingRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="payloads",
            args={"input_stage_path": _G1_USD},
        )

        params = rule.get_configuration_parameters()

        self.assertEqual(len(params), 4)
        param_names = [p.name for p in params]
        self.assertIn("variant_sets", param_names)
        self.assertIn("case_insensitive", param_names)
        self.assertIn("collect_dependencies", param_names)
        self.assertIn("excluded_variants", param_names)
        self._success = True

    async def test_process_rule_creates_variants_and_dependencies(self) -> None:
        """Verify variant files and dependencies are generated."""
        stage = Usd.Stage.Open(_G1_USD)
        rule = VariantRoutingRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="payloads",
            args={
                "input_stage_path": _G1_USD,
                "params": {"excluded_variants": _EXCLUDED_VARIANTS},
            },
        )

        rule.process_rule()

        default_prim = stage.GetDefaultPrim()
        self.assertTrue(default_prim.IsValid())
        variant_sets = default_prim.GetVariantSets()
        variant_set_names = variant_sets.GetNames()
        self.assertGreater(len(variant_set_names), 0)

        for variant_set_name in variant_set_names:
            output_dir = self._variant_set_dir(variant_set_name)
            self.assertTrue(os.path.isdir(output_dir))
            variant_set = variant_sets.GetVariantSet(variant_set_name)
            for variant_name in variant_set.GetVariantNames():
                variant_path = self._variant_file_path(variant_set_name, variant_name)
                self.assertTrue(os.path.isfile(variant_path))
                self._assert_dependency_assets_exist(variant_path)

        for variant_set_name, expected_files in _EXPECTED_DEPENDENCIES.items():
            dependencies_dir = os.path.join(self._variant_set_dir(variant_set_name), "dependencies")
            self.assertTrue(os.path.isdir(dependencies_dir))
            self.assertEqual(set(os.listdir(dependencies_dir)), expected_files)

        self._success = True

    async def test_process_rule_no_default_prim_and_logging(self) -> None:
        """No-default-prim skip, affected-stages tracking, start/completion log entries."""
        failures = []

        # -- No default prim --
        no_prim_dir = tempfile.mkdtemp()
        try:
            layer_path = os.path.join(no_prim_dir, "empty.usda")
            empty_layer = Sdf.Layer.CreateNew(layer_path)
            empty_layer.Save()
            empty_stage = Usd.Stage.Open(layer_path)
            rule = VariantRoutingRule(
                source_stage=empty_stage,
                package_root=self._tmpdir,
                destination_path="payloads",
                args={"input_stage_path": layer_path},
            )
            rule.process_rule()
            log = rule.get_operation_log()
            if not any("No valid default prim" in m for m in log):
                failures.append("No-default-prim skip not logged")
        finally:
            shutil.rmtree(no_prim_dir, ignore_errors=True)

        # -- Logging and affected stages on real asset --
        stage = Usd.Stage.Open(_G1_USD)
        rule2 = VariantRoutingRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="payloads",
            args={
                "input_stage_path": _G1_USD,
                "params": {"excluded_variants": _EXCLUDED_VARIANTS},
            },
        )
        rule2.process_rule()

        log2 = rule2.get_operation_log()
        if not any("VariantRoutingRule start" in m for m in log2):
            failures.append("Missing start log entry")
        if not any("VariantRoutingRule completed" in m for m in log2):
            failures.append("Missing completion log entry")

        affected = rule2.get_affected_stages()
        if not affected:
            failures.append("No affected stages recorded")

        self.assertEqual(failures, [], "\n".join(failures))
        self._success = True

    def _make_payload_stage_with_root_xform(
        self,
        payload_path: str,
        *,
        translate: tuple[float, float, float],
        scale: tuple[float, float, float],
        op_order: tuple[str, ...] = ("xformOp:translate", "xformOp:scale"),
    ) -> None:
        """Author a USD asset whose default prim carries an authored xform stack.

        Used by the identity-strip tests to simulate the structural-artifact
        pose that source assets routinely place on their default prim.

        Args:
            payload_path: Output path for the authored USD asset.
            translate: ``xformOp:translate`` value to author on the root.
            scale: ``xformOp:scale`` value to author on the root.
            op_order: ``xformOpOrder`` token sequence to author on the root.

        """
        stage = Usd.Stage.CreateNew(payload_path)
        prim = stage.DefinePrim("/Robot", "Xform")
        stage.SetDefaultPrim(prim)
        translate_attr = prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3, custom=False)
        translate_attr.Set(Gf.Vec3d(*translate))
        scale_attr = prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3, custom=False)
        scale_attr.Set(Gf.Vec3f(*scale))
        order_attr = prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray, custom=False)
        order_attr.Set(list(op_order))
        # A child prim so the variant has real content to carry across.
        stage.DefinePrim("/Robot/Gripper", "Xform")
        stage.GetRootLayer().Save()

    def _make_wrapper_with_variant_payload(
        self,
        wrapper_path: str,
        payload_relative: str,
        *,
        variant_name: str = "long_suction",
    ) -> None:
        """Author a wrapper USD whose variant payloads ``payload_relative``.

        The wrapper mirrors the real Isaac Sim authoring pattern: a default
        prim with a variant set whose selected variant loads an external
        asset via a prepended payload.

        Args:
            wrapper_path: Output path for the authored wrapper USD.
            payload_relative: Relative payload path loaded by the variant.
            variant_name: Name of the variant to author and select.

        """
        wrapper = Usd.Stage.CreateNew(wrapper_path)
        wrapper_prim = wrapper.DefinePrim("/Robot", "Xform")
        wrapper.SetDefaultPrim(wrapper_prim)
        vset = wrapper_prim.GetVariantSets().AddVariantSet("Gripper")
        vset.AddVariant(variant_name)
        vset.SetVariantSelection(variant_name)
        with vset.GetVariantEditContext():
            wrapper_prim.GetPayloads().AddPayload(payload_relative)
        wrapper.GetRootLayer().Save()

    async def test_strips_identity_xform_from_variant_root(self) -> None:
        """Identity translate/scale on the source root must NOT appear in the variant file.

        The original report: a variant file generated from a source whose
        default prim authored ``xformOp:translate = (0, 0, 0)`` and
        ``xformOp:scale = (1, 1, 1)`` propagated that authored identity into
        the variant layer. When the variant was later referenced into a
        wrapping stage that authored its own pose, the variant's identity
        opinions silently overrode the wrapper.
        """
        case_dir = tempfile.mkdtemp()
        try:
            payload_path = os.path.join(case_dir, "long_suction.usda")
            self._make_payload_stage_with_root_xform(payload_path, translate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))

            wrapper_path = os.path.join(case_dir, "robot.usda")
            self._make_wrapper_with_variant_payload(wrapper_path, "./long_suction.usda")

            stage = Usd.Stage.Open(wrapper_path)
            rule = VariantRoutingRule(
                source_stage=stage,
                package_root=self._tmpdir,
                destination_path="payloads",
                args={"input_stage_path": wrapper_path},
            )
            rule.process_rule()

            variant_layer_path = self._variant_file_path("Gripper", "long_suction")
            self.assertTrue(os.path.isfile(variant_layer_path), variant_layer_path)
            variant_layer = Sdf.Layer.FindOrOpen(variant_layer_path)
            self.assertIsNotNone(variant_layer)
            root_spec = variant_layer.GetPrimAtPath("/Robot")
            self.assertIsNotNone(root_spec)
            attr_names = set(root_spec.attributes.keys())
            # Identity ops and the order list must all be stripped together.
            self.assertNotIn("xformOp:translate", attr_names)
            self.assertNotIn("xformOp:scale", attr_names)
            self.assertNotIn("xformOpOrder", attr_names)
            # Child content carrying the variant must survive the strip.
            self.assertIsNotNone(variant_layer.GetPrimAtPath("/Robot/Gripper"))
            self._success = True
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    async def test_keeps_non_identity_xform_on_variant_root(self) -> None:
        """A non-identity authored translate/scale is intentional and stays."""
        case_dir = tempfile.mkdtemp()
        try:
            payload_path = os.path.join(case_dir, "long_suction.usda")
            self._make_payload_stage_with_root_xform(payload_path, translate=(5.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))

            wrapper_path = os.path.join(case_dir, "robot.usda")
            self._make_wrapper_with_variant_payload(wrapper_path, "./long_suction.usda")

            stage = Usd.Stage.Open(wrapper_path)
            rule = VariantRoutingRule(
                source_stage=stage,
                package_root=self._tmpdir,
                destination_path="payloads",
                args={"input_stage_path": wrapper_path},
            )
            rule.process_rule()

            variant_layer_path = self._variant_file_path("Gripper", "long_suction")
            self.assertTrue(os.path.isfile(variant_layer_path))
            variant_layer = Sdf.Layer.FindOrOpen(variant_layer_path)
            root_spec = variant_layer.GetPrimAtPath("/Robot")
            self.assertIsNotNone(root_spec)
            attr_names = set(root_spec.attributes.keys())
            # Whole xform stack must be preserved when any op is non-identity.
            self.assertIn("xformOp:translate", attr_names)
            self.assertIn("xformOp:scale", attr_names)
            self.assertIn("xformOpOrder", attr_names)
            translate_value = root_spec.attributes["xformOp:translate"].default
            self.assertAlmostEqual(translate_value[0], 5.0)
            self._success = True
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    async def test_keeps_xform_when_variant_authors_delta(self) -> None:
        """If the variant itself authored an xformOp, it must survive even at identity.

        The variant delta is an explicit authoring choice (the variant
        author asked for the identity transform), so the strip's
        delta-authored exemption should preserve it.
        """
        case_dir = tempfile.mkdtemp()
        try:
            payload_path = os.path.join(case_dir, "long_suction.usda")
            self._make_payload_stage_with_root_xform(payload_path, translate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))

            wrapper_path = os.path.join(case_dir, "robot.usda")
            wrapper = Usd.Stage.CreateNew(wrapper_path)
            wrapper_prim = wrapper.DefinePrim("/Robot", "Xform")
            wrapper.SetDefaultPrim(wrapper_prim)
            vset = wrapper_prim.GetVariantSets().AddVariantSet("Gripper")
            vset.AddVariant("long_suction")
            vset.SetVariantSelection("long_suction")
            with vset.GetVariantEditContext():
                wrapper_prim.GetPayloads().AddPayload("./long_suction.usda")
                # Variant explicitly authors an identity translate as its own delta.
                translate_attr = wrapper_prim.CreateAttribute(
                    "xformOp:translate", Sdf.ValueTypeNames.Double3, custom=False
                )
                translate_attr.Set(Gf.Vec3d(0.0, 0.0, 0.0))
            wrapper.GetRootLayer().Save()

            stage = Usd.Stage.Open(wrapper_path)
            rule = VariantRoutingRule(
                source_stage=stage,
                package_root=self._tmpdir,
                destination_path="payloads",
                args={"input_stage_path": wrapper_path},
            )
            rule.process_rule()

            variant_layer_path = self._variant_file_path("Gripper", "long_suction")
            variant_layer = Sdf.Layer.FindOrOpen(variant_layer_path)
            root_spec = variant_layer.GetPrimAtPath("/Robot")
            attr_names = set(root_spec.attributes.keys())
            # Variant-authored attribute survives even at identity.
            self.assertIn("xformOp:translate", attr_names)
            self._success = True
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    async def test_strips_identity_xform_from_collected_dependency(self) -> None:
        """Identity xformOps on a collected dep's default prim must be stripped.

        Variant authoring pattern: ``over "ee_link" (prepend payload =
        @dep.usd@)``. The dep's default-prim transforms get remapped onto
        the payload target prim — if the dep authors an identity transform
        on its root, that identity then wins against the base layer's
        authored pose every time the variant is selected, snapping the prim
        to the origin. The collected copy of the dep (the one the variant
        actually references) must have its identity root xformOps stripped.
        """
        case_dir = tempfile.mkdtemp()
        try:
            # The collected dep: its root has identity translate+scale that
            # would leak onto the variant's payload target prim.
            dep_path = os.path.join(case_dir, "gripper.usda")
            dep_stage = Usd.Stage.CreateNew(dep_path)
            dep_prim = dep_stage.DefinePrim("/Gripper", "Xform")
            dep_stage.SetDefaultPrim(dep_prim)
            dep_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3, custom=False).Set(
                Gf.Vec3d(0.0, 0.0, 0.0)
            )
            dep_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3, custom=False).Set(
                Gf.Vec3f(1.0, 1.0, 1.0)
            )
            dep_prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray, custom=False).Set(
                ["xformOp:translate", "xformOp:scale"]
            )
            dep_stage.DefinePrim("/Gripper/Finger", "Xform")
            dep_stage.GetRootLayer().Save()

            # Wrapper authors the same pattern ur10.usd uses for grippers: the
            # variant payloads a dep onto a CHILD prim (the over "ee_link"),
            # not on the variant's root prim itself.
            wrapper_path = os.path.join(case_dir, "robot.usda")
            wrapper = Usd.Stage.CreateNew(wrapper_path)
            wrapper_prim = wrapper.DefinePrim("/Robot", "Xform")
            wrapper.SetDefaultPrim(wrapper_prim)
            wrapper.DefinePrim("/Robot/ee_link", "Xform")
            vset = wrapper_prim.GetVariantSets().AddVariantSet("Gripper")
            vset.AddVariant("long_suction")
            vset.SetVariantSelection("long_suction")
            with vset.GetVariantEditContext():
                ee_link_prim = wrapper.GetPrimAtPath("/Robot/ee_link")
                ee_link_prim.GetPayloads().AddPayload("./gripper.usda")
            wrapper.GetRootLayer().Save()

            stage = Usd.Stage.Open(wrapper_path)
            rule = VariantRoutingRule(
                source_stage=stage,
                package_root=self._tmpdir,
                destination_path="payloads",
                args={"input_stage_path": wrapper_path},
            )
            rule.process_rule()

            # The dep was copied into the per-variant-set dependencies dir.
            dep_collected = os.path.join(self._variant_set_dir("Gripper"), "dependencies", "gripper.usda")
            self.assertTrue(os.path.isfile(dep_collected), dep_collected)
            dep_layer = Sdf.Layer.FindOrOpen(dep_collected)
            root = dep_layer.GetPrimAtPath("/Gripper")
            self.assertIsNotNone(root)
            attr_names = set(root.attributes.keys())
            # Identity-only authoring must be stripped from the collected dep.
            self.assertNotIn("xformOp:translate", attr_names)
            self.assertNotIn("xformOp:scale", attr_names)
            self.assertNotIn("xformOpOrder", attr_names)
            # Child content must survive.
            self.assertIsNotNone(dep_layer.GetPrimAtPath("/Gripper/Finger"))
            self._success = True
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    async def test_keeps_xform_with_reset_xform_stack(self) -> None:
        """``!resetXformStack!`` is a semantic directive — never strip."""
        case_dir = tempfile.mkdtemp()
        try:
            payload_path = os.path.join(case_dir, "long_suction.usda")
            self._make_payload_stage_with_root_xform(
                payload_path,
                translate=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                op_order=("!resetXformStack!", "xformOp:translate", "xformOp:scale"),
            )

            wrapper_path = os.path.join(case_dir, "robot.usda")
            self._make_wrapper_with_variant_payload(wrapper_path, "./long_suction.usda")

            stage = Usd.Stage.Open(wrapper_path)
            rule = VariantRoutingRule(
                source_stage=stage,
                package_root=self._tmpdir,
                destination_path="payloads",
                args={"input_stage_path": wrapper_path},
            )
            rule.process_rule()

            variant_layer_path = self._variant_file_path("Gripper", "long_suction")
            variant_layer = Sdf.Layer.FindOrOpen(variant_layer_path)
            root_spec = variant_layer.GetPrimAtPath("/Robot")
            attr_names = set(root_spec.attributes.keys())
            self.assertIn("xformOpOrder", attr_names)
            order_value = root_spec.attributes["xformOpOrder"].default
            self.assertIn("!resetXformStack!", list(order_value))
            self._success = True
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    async def test_isaac_ur10_gripper_variant_strips_root_xformops(self) -> None:
        """End-to-end check against the standard Isaac UR10 asset.

        Loads ``Isaac/Robots/UniversalRobots/ur10/ur10.usd`` (the canonical
        sample where the Gripper variant payloads ``long_gripper.usd`` /
        ``short_gripper.usd`` onto ``/ur10/ee_link``) and asserts that the
        generated variant files and their collected dependencies carry NO
        identity-only ``xformOp:`` / ``xformOpOrder`` opinions on their
        default prim. Such opinions used to override the base layer's
        authored ``ee_link`` pose, snapping the prim to the origin every
        time the Gripper variant was selected.

        Skipped when the Isaac assets root cannot be resolved (offline /
        unconfigured CI). ``isaacsim.storage.native`` is wired in as a
        test-only dependency in extension.toml's ``[[test]]`` block.
        """
        from isaacsim.storage.native import get_assets_root_path_async

        assets_root = await get_assets_root_path_async()
        if not assets_root:
            self.skipTest("Isaac assets root not configured")
            return

        input_path = assets_root + _ISAAC_UR10_PATH

        # ``Usd.Stage.Open`` and ``VariantRoutingRule.process_rule`` are fully
        # synchronous and, for a remote assets root, do a large amount of
        # blocking ``omni.client`` I/O (dependency computation, stat, and
        # file reads). Running them directly on the async test would block the
        # Kit main-thread event loop that pumps ``omni.client`` callbacks, so
        # the remote fetches never make progress and the whole extension test
        # process is killed at its wall-clock timeout. Offload the blocking
        # work to a worker thread so the event loop keeps pumping and the
        # remote asset resolves. Local (on-disk) roots are unaffected.
        def _open_and_process() -> Usd.Stage | None:
            stage = Usd.Stage.Open(input_path)
            if stage is None or not stage.GetDefaultPrim().IsValid():
                return None
            rule = VariantRoutingRule(
                source_stage=stage,
                package_root=self._tmpdir,
                destination_path="payloads",
                args={
                    "input_stage_path": input_path,
                    "params": {"excluded_variants": _EXCLUDED_VARIANTS},
                },
            )
            rule.process_rule()
            return stage

        stage = await asyncio.get_event_loop().run_in_executor(None, _open_and_process)
        if stage is None:
            self.skipTest(f"could not open Isaac UR10 asset at {input_path}")
            return

        gripper_dir = os.path.join(self._tmpdir, "payloads", "Gripper")
        self.assertTrue(os.path.isdir(gripper_dir), gripper_dir)

        excluded_lower = {v.lower() for v in _EXCLUDED_VARIANTS}
        usd_exts = {".usd", ".usda", ".usdc"}
        failures: list[str] = []

        # 1) Each generated Gripper variant file's /ur10 root must be clean.
        for fname in sorted(os.listdir(gripper_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in usd_exts:
                continue
            if stem in excluded_lower:
                continue
            fpath = os.path.join(gripper_dir, fname)
            layer = Sdf.Layer.FindOrOpen(fpath)
            self.assertIsNotNone(layer, fpath)
            root = layer.GetPrimAtPath("/ur10")
            if root is None:
                failures.append(f"variant {fname}: /ur10 prim spec missing")
                continue
            leaked: list[str] = []
            for attr_name in root.attributes.keys():  # noqa: SIM118 -- Sdf proxy needs .keys() to yield names
                if attr_name.startswith("xformOp:") or attr_name == "xformOpOrder":
                    leaked.append(attr_name)
            if leaked:
                failures.append(f"variant {fname}: identity xformOps leaked on /ur10: {leaked}")

        # 2) Every collected dep used by the Gripper variants (the actual
        #    payload targets that override /ur10/ee_link) must also be clean.
        deps_dir = os.path.join(gripper_dir, "dependencies")
        self.assertTrue(os.path.isdir(deps_dir), deps_dir)
        for fname in sorted(os.listdir(deps_dir)):
            if os.path.splitext(fname)[1].lower() not in usd_exts:
                continue
            fpath = os.path.join(deps_dir, fname)
            layer = Sdf.Layer.FindOrOpen(fpath)
            self.assertIsNotNone(layer, fpath)
            default_prim_name = layer.defaultPrim
            if not default_prim_name:
                continue
            prim_spec = layer.GetPrimAtPath(Sdf.Path.absoluteRootPath.AppendChild(default_prim_name))
            if prim_spec is None:
                continue
            leaked = []
            for attr_name in prim_spec.attributes.keys():  # noqa: SIM118 -- Sdf proxy needs .keys() to yield names
                if attr_name.startswith("xformOp:") or attr_name == "xformOpOrder":
                    leaked.append(attr_name)
            if not leaked:
                continue
            # If the dep authors a real (non-identity) pose, keep it.
            non_identity = [n for n in leaked if not rule._xform_attr_is_identity(prim_spec.attributes[n])]
            if non_identity:
                continue
            failures.append(f"dep {fname}: identity xformOps leaked on /{default_prim_name}: {leaked}")

        self.assertEqual(failures, [], "\n".join(failures))
        self._success = True

    async def test_process_rule_options(self) -> None:
        """variant_sets filter, case_insensitive=False, collect_dependencies=False."""
        stage = Usd.Stage.Open(_G1_USD)
        default_prim = stage.GetDefaultPrim()
        all_vs_names = default_prim.GetVariantSets().GetNames()
        failures = []

        # -- Variant-set filter: only first set --
        filter_dir = tempfile.mkdtemp()
        try:
            target_vs = all_vs_names[0]
            rule_filter = VariantRoutingRule(
                source_stage=stage,
                package_root=filter_dir,
                destination_path="payloads",
                args={
                    "input_stage_path": _G1_USD,
                    "params": {
                        "variant_sets": [target_vs],
                        "excluded_variants": _EXCLUDED_VARIANTS,
                    },
                },
            )
            rule_filter.process_rule()
            output_base = os.path.join(filter_dir, "payloads")
            created_dirs = [d for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]
            if len(created_dirs) != 1:
                failures.append(f"variant_sets filter: expected 1 dir, got {len(created_dirs)}: {created_dirs}")
        finally:
            shutil.rmtree(filter_dir, ignore_errors=True)

        # -- case_insensitive=False: at least one filename with uppercase --
        case_dir = tempfile.mkdtemp()
        try:
            rule_case = VariantRoutingRule(
                source_stage=stage,
                package_root=case_dir,
                destination_path="payloads",
                args={
                    "input_stage_path": _G1_USD,
                    "params": {
                        "case_insensitive": False,
                        "excluded_variants": ["None", "Default", "PhysX"],
                    },
                },
            )
            rule_case.process_rule()
            found_upper = False
            for root, _dirs, files in os.walk(os.path.join(case_dir, "payloads")):
                for f in files:
                    if f.endswith(".usda") and f != f.lower():
                        found_upper = True
                        break
            if not found_upper:
                failures.append("case_insensitive=False: no uppercase filenames found")
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

        # -- collect_dependencies=False: no dependencies/ dirs --
        nodep_dir = tempfile.mkdtemp()
        try:
            rule_nodep = VariantRoutingRule(
                source_stage=stage,
                package_root=nodep_dir,
                destination_path="payloads",
                args={
                    "input_stage_path": _G1_USD,
                    "params": {
                        "collect_dependencies": False,
                        "excluded_variants": _EXCLUDED_VARIANTS,
                    },
                },
            )
            rule_nodep.process_rule()
            for vs_name in all_vs_names:
                dep_dir = os.path.join(nodep_dir, "payloads", sanitize_prim_name(vs_name), "dependencies")
                if os.path.isdir(dep_dir):
                    failures.append(f"collect_dependencies=False: {dep_dir} should not exist")
        finally:
            shutil.rmtree(nodep_dir, ignore_errors=True)

        self.assertEqual(failures, [], "\n".join(failures))
        self._success = True
