# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for urdf_synth: USD->URDF synthesis logic.

These cover the pure functions (XML transforms over ElementTree, USD inspection
over an in-memory stage). No simulation, GPU, or controller_manager required.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import deque
from collections.abc import Callable
from unittest import mock
from urllib.parse import unquote
from xml.etree import ElementTree as ET

import omni.kit.test
from isaacsim.ros2.control.urdf_synth import (
    HARDWARE_PLUGIN_NAME,
    HardwareBlock,
    JointEntry,
    _detect_command_kind,
    _extract_mimic_map,
    _fix_joint_limits,
    _fix_root_link_inertia,
    _is_finite_float,
    _prune_to_control_tree,
    _splice_sensor_overlay,
    _stable_export_dir,
    build_full_urdf,
    discover_articulations,
    hardware_block_to_xml,
    synthesize_hardware_block,
)
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


def _postprocessed_export(
    xml: str,
) -> Callable[[Usd.Stage, str, Callable[[ET.Element], None]], str]:
    def export(_stage: Usd.Stage, _target: str, postprocess: Callable[[ET.Element], None]) -> str:
        root = ET.fromstring(xml)
        postprocess(root)
        return ET.tostring(root, encoding="unicode")

    return export


def _urdf(links, joints):
    """Build a <robot> ElementTree from (name) links and (name,type,parent,child) joints."""
    robot = ET.Element("robot", {"name": "test"})
    for name in links:
        ET.SubElement(robot, "link", {"name": name})
    for name, jtype, parent, child in joints:
        j = ET.SubElement(robot, "joint", {"name": name, "type": jtype})
        ET.SubElement(j, "parent", {"link": parent})
        ET.SubElement(j, "child", {"link": child})
    return robot


def _drive_joint(stage, path, *, prismatic=False, stiffness=None, damping=None, drive=True):
    """Define a revolute/prismatic joint with an optional DriveAPI."""
    if prismatic:
        joint = UsdPhysics.PrismaticJoint.Define(stage, path)
        token = "linear"
    else:
        joint = UsdPhysics.RevoluteJoint.Define(stage, path)
        token = "angular"
    prim = joint.GetPrim()
    if drive:
        drv = UsdPhysics.DriveAPI.Apply(prim, token)
        if stiffness is not None:
            drv.CreateStiffnessAttr().Set(stiffness)
        if damping is not None:
            drv.CreateDampingAttr().Set(damping)
    return prim


class TestDetectCommandKind(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()

    async def test_position_when_stiffness_positive(self):
        prim = _drive_joint(self.stage, "/J", stiffness=100.0, damping=5.0)
        self.assertEqual(_detect_command_kind(prim), "position")  # stiffness wins over damping

    async def test_velocity_when_only_damping(self):
        prim = _drive_joint(self.stage, "/J", stiffness=0.0, damping=5.0)
        self.assertEqual(_detect_command_kind(prim), "velocity")

    async def test_velocity_when_stiffness_attr_absent(self):
        prim = _drive_joint(self.stage, "/J", damping=5.0)  # stiffness unauthored -> None -> 0
        self.assertEqual(_detect_command_kind(prim), "velocity")

    async def test_effort_when_no_gains(self):
        prim = _drive_joint(self.stage, "/J", stiffness=0.0, damping=0.0)
        self.assertEqual(_detect_command_kind(prim), "effort")

    async def test_effort_when_drive_present_but_gains_unset(self):
        prim = _drive_joint(self.stage, "/J")  # DriveAPI present, gains unset (treated as 0)
        self.assertEqual(_detect_command_kind(prim), "effort")

    async def test_none_without_driveapi(self):
        prim = _drive_joint(self.stage, "/J", drive=False)
        self.assertIsNone(_detect_command_kind(prim))

    async def test_none_for_non_articulated_joint(self):
        prim = UsdPhysics.FixedJoint.Define(self.stage, "/F").GetPrim()
        self.assertIsNone(_detect_command_kind(prim))

    async def test_prismatic_position(self):
        prim = _drive_joint(self.stage, "/J", prismatic=True, stiffness=50.0)
        self.assertEqual(_detect_command_kind(prim), "position")


class TestDiscoverArticulations(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()

    async def test_target_itself_is_root(self):
        prim = UsdGeom.Xform.Define(self.stage, "/World/Robot").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        found = discover_articulations(self.stage, "/World/Robot")
        self.assertEqual([str(p.GetPath()) for p in found], ["/World/Robot"])

    async def test_nested_root(self):
        UsdGeom.Xform.Define(self.stage, "/World")
        prim = UsdGeom.Xform.Define(self.stage, "/World/Robot/base").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        found = discover_articulations(self.stage, "/World")
        self.assertEqual([str(p.GetPath()) for p in found], ["/World/Robot/base"])

    async def test_invalid_path_raises(self):
        with self.assertRaises(ValueError):
            discover_articulations(self.stage, "/does/not/exist")

    async def test_no_articulation_raises(self):
        UsdGeom.Xform.Define(self.stage, "/World")
        with self.assertRaises(ValueError):
            discover_articulations(self.stage, "/World")

    async def test_multiple_roots_returned(self):
        UsdGeom.Xform.Define(self.stage, "/World")
        for name in ("A", "B"):
            p = UsdGeom.Xform.Define(self.stage, f"/World/{name}").GetPrim()
            UsdPhysics.ArticulationRootAPI.Apply(p)
        self.assertEqual(len(discover_articulations(self.stage, "/World")), 2)


class TestSynthesizeHardwareBlock(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()

    async def test_enumerates_articulated_joints_only(self):
        root = UsdGeom.Xform.Define(self.stage, "/World/Robot").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(root)
        _drive_joint(self.stage, "/World/Robot/rev", stiffness=10.0)
        _drive_joint(self.stage, "/World/Robot/pris", prismatic=True, stiffness=0.0, damping=2.0)
        UsdPhysics.FixedJoint.Define(self.stage, "/World/Robot/fixed")  # skipped

        block = synthesize_hardware_block(self.stage, root, joint_scope=root)
        self.assertEqual(block.articulation_prim_path, "/World/Robot")
        kinds = {j.name: j.command_kind for j in block.joints}
        self.assertEqual(kinds, {"rev": "position", "pris": "velocity"})

    async def test_empty_scope_yields_no_joints(self):
        root = UsdGeom.Xform.Define(self.stage, "/World/Robot").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(root)
        block = synthesize_hardware_block(self.stage, root, joint_scope=root)
        self.assertEqual(block.joints, [])

    async def test_physx_mimic_joint_without_drive_is_state_only(self):
        root = UsdGeom.Xform.Define(self.stage, "/World/Hand").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(root)
        source = _drive_joint(self.stage, "/World/Hand/source_joint", stiffness=10.0)
        mimic = _drive_joint(self.stage, "/World/Hand/mimic_joint", drive=False)
        mimic_api = PhysxSchema.PhysxMimicJointAPI.Apply(mimic, "rotZ")
        mimic_api.CreateReferenceJointRel().SetTargets([source.GetPath()])
        mimic_api.CreateGearingAttr().Set(-1.0)
        mimic_api.CreateOffsetAttr().Set(0.1)

        block = synthesize_hardware_block(self.stage, root, joint_scope=root)
        self.assertEqual(
            {j.name: j.command_kind for j in block.joints},
            {"source_joint": "position", "mimic_joint": None},
        )


class TestHardwareBlockToXml(omni.kit.test.AsyncTestCase):
    async def test_structure_and_interfaces(self):
        block = HardwareBlock(
            name="arm",
            articulation_prim_path="/World/Arm",
            joints=[JointEntry("j1", "position"), JointEntry("j2", None)],
        )
        el = hardware_block_to_xml(block)
        self.assertEqual(el.tag, "ros2_control")
        self.assertEqual(el.attrib, {"name": "arm", "type": "system"})
        self.assertEqual(el.findtext("hardware/plugin"), "isaacsim_ros2_control/IsaacSimSystem")
        self.assertEqual(el.find("hardware/param[@name='prim_path']").text, "/World/Arm")

        j1 = el.find("joint[@name='j1']")
        self.assertIsNotNone(j1.find("command_interface[@name='position']"))
        self.assertEqual({s.get("name") for s in j1.findall("state_interface")}, {"position", "velocity", "effort"})

        j2 = el.find("joint[@name='j2']")
        self.assertEqual(j2.findall("command_interface"), [])  # state-only joint

    async def test_driven_joint_exports_command_interface_superset(self):
        for kind in ("position", "velocity", "effort"):
            el = hardware_block_to_xml(
                HardwareBlock(name="arm", articulation_prim_path="/World/Arm", joints=[JointEntry("j1", kind)])
            )
            j1 = el.find("joint[@name='j1']")
            self.assertEqual(
                {c.get("name") for c in j1.findall("command_interface")},
                {"position", "velocity", "effort"},
                f"driven joint inferred as {kind!r} should still export all three command interfaces",
            )

    async def test_empty_joints_still_valid(self):
        el = hardware_block_to_xml(HardwareBlock(name="x", articulation_prim_path="/X"))
        self.assertIsNotNone(el.find("hardware"))
        self.assertEqual(el.findall("joint"), [])


class TestFixJointLimits(omni.kit.test.AsyncTestCase):
    async def test_adds_missing_limit(self):
        root = _urdf(["a", "b"], [("j", "revolute", "a", "b")])
        _fix_joint_limits(root)
        limit = root.find("joint/limit")
        self.assertEqual(limit.get("lower"), "-3.141592653589793")
        self.assertEqual(limit.get("upper"), "3.141592653589793")
        self.assertEqual(limit.get("effort"), "1000.0")
        self.assertEqual(limit.get("velocity"), "10.0")

    async def test_preserves_existing_and_fills_gap(self):
        root = _urdf(["a", "b"], [("j", "prismatic", "a", "b")])
        ET.SubElement(root.find("joint"), "limit", {"lower": "-0.2", "upper": "0.4", "effort": "7.0"})
        _fix_joint_limits(root)
        limit = root.find("joint/limit")
        self.assertEqual(limit.get("lower"), "-0.2")
        self.assertEqual(limit.get("upper"), "0.4")
        self.assertEqual(limit.get("effort"), "7.0")  # not overwritten
        self.assertEqual(limit.get("velocity"), "10.0")  # gap filled

    async def test_continuous_joint_gets_effort_velocity_limit(self):
        root = _urdf(["a", "b"], [("j", "continuous", "a", "b")])
        _fix_joint_limits(root)
        limit = root.find("joint/limit")
        self.assertEqual(limit.get("effort"), "1000.0")
        self.assertEqual(limit.get("velocity"), "10.0")
        self.assertIsNone(limit.get("lower"))
        self.assertIsNone(limit.get("upper"))

    async def test_replaces_nonfinite_bounds(self):
        root = _urdf(["a", "b"], [("j", "revolute", "a", "b")])
        ET.SubElement(root.find("joint"), "limit", {"lower": "-inf", "upper": "inf"})
        _fix_joint_limits(root)
        limit = root.find("joint/limit")
        self.assertEqual(limit.get("lower"), "-3.141592653589793")
        self.assertEqual(limit.get("upper"), "3.141592653589793")

    async def test_nonfinite_effort_is_sanitized_before_controller_manager_setup(self) -> None:
        """Verify that non-finite effort and velocity limits are replaced."""
        root = _urdf(["a", "b"], [("j", "revolute", "a", "b")])
        ET.SubElement(root.find("joint"), "limit", {"effort": "inf", "velocity": "-inf"})
        _fix_joint_limits(root)
        limit = root.find("joint/limit")
        self.assertEqual(limit.get("effort"), "1000.0")
        self.assertEqual(limit.get("velocity"), "10.0")

    async def test_ignores_fixed_joint(self):
        root = _urdf(["a", "b"], [("j", "fixed", "a", "b")])
        _fix_joint_limits(root)
        self.assertIsNone(root.find("joint/limit"))

    async def test_idempotent(self):
        root = _urdf(["a", "b"], [("j", "revolute", "a", "b")])
        _fix_joint_limits(root)
        _fix_joint_limits(root)
        self.assertEqual(len(root.find("joint").findall("limit")), 1)


class TestPruneToControlTree(omni.kit.test.AsyncTestCase):
    async def test_generated_world_anchor_does_not_replace_control_root(self):
        root = _urdf(
            ["world", "base", "link1"],
            [
                ("floating_base", "floating", "world", "link1"),
                ("joint1", "revolute", "base", "link1"),
            ],
        )
        _prune_to_control_tree(root, ["joint1"])
        self.assertEqual({l.get("name") for l in root.findall("link")}, {"base", "link1"})
        self.assertEqual({j.get("name") for j in root.findall("joint")}, {"joint1"})

    async def test_keeps_descendants_of_control_chain(self):
        root = _urdf(
            ["world", "base", "link1", "tool"],
            [
                ("floating_base", "floating", "world", "link1"),
                ("joint1", "revolute", "base", "link1"),
                ("tool_fixed", "fixed", "link1", "tool"),
            ],
        )
        _prune_to_control_tree(root, ["joint1"])
        self.assertEqual({l.get("name") for l in root.findall("link")}, {"base", "link1", "tool"})
        self.assertEqual({j.get("name") for j in root.findall("joint")}, {"joint1", "tool_fixed"})


class TestFixRootLinkInertia(omni.kit.test.AsyncTestCase):
    async def test_inserts_dummy_for_inertial_root(self):
        root = _urdf(["base", "link1"], [("j", "fixed", "base", "link1")])
        ET.SubElement(root.find("link[@name='base']"), "inertial")
        _fix_root_link_inertia(root)
        children = list(root)
        self.assertEqual(children[0].tag, "link")
        self.assertEqual(children[0].get("name"), "base_kdl_dummy_root")
        self.assertEqual(children[1].tag, "joint")
        self.assertEqual(children[1].get("type"), "fixed")
        self.assertEqual(children[1].find("child").get("link"), "base")

    async def test_noop_when_root_has_no_inertial(self):
        root = _urdf(["base", "link1"], [("j", "fixed", "base", "link1")])
        before = ET.tostring(root)
        _fix_root_link_inertia(root)
        self.assertEqual(ET.tostring(root), before)

    async def test_noop_when_no_links(self):
        root = ET.Element("robot")
        _fix_root_link_inertia(root)
        self.assertEqual(len(list(root)), 0)


class TestSpliceSensorOverlay(omni.kit.test.AsyncTestCase):
    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix=".urdf")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        return path

    def _dst(self, name="arm"):
        root = ET.Element("robot")
        rc = ET.SubElement(root, "ros2_control", {"name": name, "type": "system"})
        ET.SubElement(rc, "hardware")
        return root

    async def test_empty_path_noop(self):
        root = self._dst()
        _splice_sensor_overlay(root, "")
        self.assertEqual(root.find("ros2_control").findall("sensor"), [])

    async def test_missing_file_warns_and_noops(self):
        root = self._dst()
        with mock.patch("isaacsim.ros2.control.urdf_synth.carb.log_warn") as warn:
            _splice_sensor_overlay(root, "/no/such/file.urdf")
        warn.assert_called_once()
        self.assertEqual(root.find("ros2_control").findall("sensor"), [])

    async def test_sensor_merged_into_named_block(self):
        path = self._write("<robot><ros2_control name='arm'><sensor name='ft'/></ros2_control></robot>")
        root = self._dst("arm")
        _splice_sensor_overlay(root, path)
        sensors = root.find("ros2_control[@name='arm']").findall("sensor")
        self.assertEqual([s.get("name") for s in sensors], ["ft"])

    async def test_name_mismatch_falls_back_to_first_block(self):
        # Overlay names 'other', destination has 'first' and 'second' (neither matches):
        # the sensor must land in the first block only, not the second.
        path = self._write("<robot><ros2_control name='other'><sensor name='imu'/></ros2_control></robot>")
        root = ET.Element("robot")
        first = ET.SubElement(root, "ros2_control", {"name": "first"})
        ET.SubElement(first, "hardware")
        second = ET.SubElement(root, "ros2_control", {"name": "second"})
        ET.SubElement(second, "hardware")
        _splice_sensor_overlay(root, path)
        self.assertEqual([s.get("name") for s in first.findall("sensor")], ["imu"])
        self.assertEqual(second.findall("sensor"), [])

    async def test_overlay_prim_path_ignored_but_ours_preserved(self):
        path = self._write(
            "<robot><ros2_control name='arm'><hardware>"
            "<param name='prim_path'>/evil</param><param name='foo'>bar</param>"
            "</hardware></ros2_control></robot>"
        )
        root = self._dst("arm")
        ours = ET.SubElement(root.find("ros2_control/hardware"), "param", {"name": "prim_path"})
        ours.text = "/World/R"  # synthesized: we own prim_path
        _splice_sensor_overlay(root, path)
        params = root.find("ros2_control/hardware").findall("param")
        prim_paths = [p.text for p in params if p.get("name") == "prim_path"]
        self.assertEqual(prim_paths, ["/World/R"])  # overlay's /evil dropped, ours intact
        self.assertEqual([p.text for p in params if p.get("name") == "foo"], ["bar"])

    async def test_malformed_overlay_raises(self):
        path = self._write("<robot><ros2_control></robot>")  # unclosed tag
        with self.assertRaises(ET.ParseError):
            _splice_sensor_overlay(self._dst(), path)


class TestStableExportDir(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        # Redirect the export root into a private temp dir so concurrent runs
        # (and the real /tmp) cannot collide on the deterministic dir name.
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        patcher = mock.patch("isaacsim.ros2.control.urdf_synth.tempfile.gettempdir", return_value=self._tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_deterministic_and_distinct(self):
        a1 = _stable_export_dir("/World/RobotA")
        a2 = _stable_export_dir("/World/RobotA")
        b = _stable_export_dir("/World/RobotB")
        self.assertEqual(a1, a2)
        self.assertNotEqual(a1, b)
        self.assertTrue(os.path.isdir(a1))

    async def test_cleared_on_recreate(self):
        d = _stable_export_dir("/World/RobotC")
        stale = os.path.join(d, "stale.txt")
        with open(stale, "w", encoding="utf-8") as f:
            f.write("x")
        _stable_export_dir("/World/RobotC")
        self.assertFalse(os.path.exists(stale))


class TestBuildFullUrdfArticulationGuard(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(self.stage, "/World")

    def _write_overlay(self, text):
        fd, path = tempfile.mkstemp(suffix=".urdf")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        return path

    def _add_root(self, name):
        prim = UsdGeom.Xform.Define(self.stage, f"/World/{name}").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(prim)

    async def test_multiple_articulations_rejected(self):
        self._add_root("A")
        self._add_root("B")
        with self.assertRaises(ValueError):
            build_full_urdf(self.stage, "/World")

    async def test_single_articulation_emits_ros2_control(self):
        self._add_root("A")
        _drive_joint(self.stage, "/World/A/j", stiffness=10.0)
        with mock.patch(
            "isaacsim.ros2.control.urdf_synth._export_kinematic_urdf",
            side_effect=_postprocessed_export("<robot name='r'><link name='base'/></robot>"),
        ):
            urdf = build_full_urdf(self.stage, "/World")
        root = ET.fromstring(urdf)
        rc = root.find("ros2_control")
        self.assertIsNotNone(rc)
        self.assertEqual(rc.find("hardware/param[@name='prim_path']").text, "/World/A")
        self.assertEqual([j.get("name") for j in rc.findall("joint")], ["j"])

    async def test_parent_target_keeps_sibling_joint_for_nested_articulation_root(self):
        root = UsdGeom.Xform.Define(self.stage, "/World/Robot/base").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(root)
        _drive_joint(self.stage, "/World/Robot/shoulder", stiffness=10.0)
        kinematic_urdf = (
            "<robot name='r'>"
            "<link name='base'/><link name='link1'/>"
            "<joint name='shoulder' type='revolute'>"
            "<parent link='base'/><child link='link1'/>"
            "</joint></robot>"
        )
        with mock.patch(
            "isaacsim.ros2.control.urdf_synth._export_kinematic_urdf",
            side_effect=_postprocessed_export(kinematic_urdf),
        ):
            urdf = build_full_urdf(self.stage, "/World/Robot")

        rc = ET.fromstring(urdf).find("ros2_control")
        self.assertIsNotNone(rc)
        self.assertEqual(rc.find("hardware/param[@name='prim_path']").text, "/World/Robot/base")
        self.assertEqual([j.get("name") for j in rc.findall("joint")], ["shoulder"])

    async def test_sensor_overlay_is_spliced_into_full_backend_urdf(self):
        self._add_root("Arm")
        _drive_joint(self.stage, "/World/Arm/shoulder", stiffness=10.0)
        overlay_path = self._write_overlay(
            "<robot><ros2_control name='Arm'><hardware>"
            "<param name='prim_path'>/overlay/should/not/win</param>"
            "<param name='sensor_topic'>imu/data</param>"
            "</hardware><sensor name='imu'><state_interface name='orientation.x'/>"
            "</sensor></ros2_control></robot>"
        )
        kinematic_urdf = (
            "<robot name='arm'>"
            "<link name='base'/><link name='upper'/>"
            "<joint name='shoulder' type='revolute'>"
            "<parent link='base'/><child link='upper'/>"
            "</joint></robot>"
        )

        with mock.patch(
            "isaacsim.ros2.control.urdf_synth._export_kinematic_urdf",
            side_effect=_postprocessed_export(kinematic_urdf),
        ):
            urdf = build_full_urdf(self.stage, "/World/Arm", sensor_overlay_urdf_path=overlay_path)

        root = ET.fromstring(urdf)
        rc = root.find("ros2_control[@name='Arm']")
        self.assertIsNotNone(rc)
        self.assertEqual([j.get("name") for j in rc.findall("joint")], ["shoulder"])
        self.assertIsNotNone(rc.find("sensor[@name='imu']/state_interface[@name='orientation.x']"))
        self.assertEqual(rc.find("hardware/param[@name='sensor_topic']").text, "imu/data")
        prim_paths = [p.text for p in rc.findall("hardware/param") if p.get("name") == "prim_path"]
        self.assertEqual(prim_paths, ["/World/Arm"])

    async def test_mimic_joint_survives_full_urdf_and_is_state_only(self):
        self._add_root("Hand")
        source = _drive_joint(self.stage, "/World/Hand/source_joint", stiffness=10.0)
        mimic = _drive_joint(self.stage, "/World/Hand/mimic_joint", drive=False)
        mimic_api = PhysxSchema.PhysxMimicJointAPI.Apply(mimic, "rotZ")
        mimic_api.CreateReferenceJointRel().SetTargets([source.GetPath()])
        mimic_api.CreateGearingAttr().Set(-1.0)
        mimic_api.CreateOffsetAttr().Set(0.1)
        kinematic_urdf = (
            "<robot name='hand'>"
            "<link name='base'/><link name='finger_a'/><link name='finger_b'/>"
            "<joint name='source_joint' type='revolute'>"
            "<parent link='base'/><child link='finger_a'/>"
            "</joint>"
            "<joint name='mimic_joint' type='revolute'>"
            "<parent link='finger_a'/><child link='finger_b'/>"
            "<mimic joint='source_joint' multiplier='-1.0' offset='0.1'/>"
            "</joint></robot>"
        )

        with mock.patch(
            "isaacsim.ros2.control.urdf_synth._export_kinematic_urdf",
            side_effect=_postprocessed_export(kinematic_urdf),
        ):
            urdf = build_full_urdf(self.stage, "/World/Hand")

        root = ET.fromstring(urdf)
        mimic_tag = root.find("joint[@name='mimic_joint']/mimic")
        self.assertIsNotNone(mimic_tag)
        self.assertEqual(mimic_tag.attrib, {"joint": "source_joint", "multiplier": "-1.0", "offset": "0.1"})

        rc = root.find("ros2_control[@name='Hand']")
        self.assertIsNotNone(rc)
        source_rc = rc.find("joint[@name='source_joint']")
        mimic_rc = rc.find("joint[@name='mimic_joint']")
        self.assertIsNotNone(source_rc.find("command_interface[@name='position']"))
        self.assertEqual(mimic_rc.findall("command_interface"), [])
        self.assertEqual(
            {s.get("name") for s in mimic_rc.findall("state_interface")},
            {"position", "velocity", "effort"},
        )


def _make_link(stage, path, mass=1.0):
    """A rigid-body link with explicit mass/inertia (no geometry needed)."""
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.1, 0.1))
    return prim


def _make_drive_joint(stage, path, parent, child, *, prismatic=False, stiffness=None, damping=None):
    if prismatic:
        joint = UsdPhysics.PrismaticJoint.Define(stage, path)
        token = "linear"
    else:
        joint = UsdPhysics.RevoluteJoint.Define(stage, path)
        token = "angular"
    joint.CreateBody0Rel().SetTargets([parent.GetPath()])
    joint.CreateBody1Rel().SetTargets([child.GetPath()])
    joint.CreateAxisAttr("Z")
    if stiffness is not None or damping is not None:
        drv = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), token)
        if stiffness is not None:
            drv.CreateStiffnessAttr().Set(stiffness)
        if damping is not None:
            drv.CreateDampingAttr().Set(damping)
    return joint.GetPrim()


class TestBuildFullUrdfRealConverter(omni.kit.test.AsyncTestCase):
    """Characterization tests: drive build_full_urdf through the REAL converter
    (no mock) to pin current end-to-end output. build_full_urdf never loads the
    ros2_control backend, so no physics/Play/ROS is needed."""

    def _build_arm(self, *, prismatic=False, stiffness=2000.0, damping=200.0, root="/World/Arm", with_mesh=False):
        """Fixed-base 1-DOF arm: base (fixed to world) -> driven joint -> link1."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        arm = UsdGeom.Xform.Define(stage, root).GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(arm)
        base = _make_link(stage, f"{root}/base")
        UsdPhysics.FixedJoint.Define(stage, f"{root}/base_fixed").CreateBody1Rel().SetTargets([base.GetPath()])
        link1 = _make_link(stage, f"{root}/link1")
        _make_drive_joint(
            stage, f"{root}/joint1", base, link1, prismatic=prismatic, stiffness=stiffness, damping=damping
        )
        if with_mesh:
            mesh = UsdGeom.Mesh.Define(stage, f"{root}/base/visual")
            mesh.CreatePointsAttr().Set([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 0, 1)])
            mesh.CreateFaceVertexCountsAttr().Set([3, 3, 3, 3])
            mesh.CreateFaceVertexIndicesAttr().Set([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3])
        return stage

    def _assert_single_root_tree(self, root: ET.Element) -> None:
        """The URDF must be one connected, acyclic tree (controller_manager / KDL)."""
        links = [link.get("name") for link in root.findall("link")]
        joints = root.findall("joint")
        children = [j.find("child").get("link") for j in joints if j.find("child") is not None]
        self.assertEqual(len(children), len(set(children)), "a link is the child of >1 joint (not a tree)")
        roots = [link for link in links if link not in children]
        self.assertEqual(len(roots), 1, f"expected exactly one root link, got {roots}")

        edges: dict[str, list[str]] = {}
        for j in joints:
            parent, child = j.find("parent"), j.find("child")
            if parent is not None and child is not None:
                edges.setdefault(parent.get("link"), []).append(child.get("link"))
        seen, queue = set(roots), deque(roots)
        while queue:
            for child in edges.get(queue.popleft(), []):
                self.assertNotIn(child, seen, "cycle detected in URDF tree")
                seen.add(child)
                queue.append(child)
        self.assertEqual(seen, set(links), "URDF links not all reachable from the root (disconnected)")

    def _assert_movable_joints_have_limits(self, root: ET.Element) -> None:
        """Every revolute/prismatic joint carries finite effort+velocity (URDF parser requirement)."""
        for joint in root.findall("joint"):
            if joint.get("type") not in ("revolute", "prismatic"):
                continue
            limit = joint.find("limit")
            self.assertIsNotNone(limit, f"joint {joint.get('name')!r} missing <limit>")
            self.assertTrue(_is_finite_float(limit.get("effort")), f"joint {joint.get('name')!r} effort not finite")
            self.assertTrue(_is_finite_float(limit.get("velocity")), f"joint {joint.get('name')!r} velocity not finite")
            self.assertTrue(_is_finite_float(limit.get("lower")))
            self.assertTrue(_is_finite_float(limit.get("upper")))

    async def test_position_arm_emits_block_limits_and_tree(self):
        root = ET.fromstring(build_full_urdf(self._build_arm(stiffness=2000.0, damping=200.0), "/World/Arm"))

        rc = root.find("ros2_control")
        self.assertIsNotNone(rc, "no <ros2_control> block emitted")
        self.assertEqual(rc.findtext("hardware/plugin"), HARDWARE_PLUGIN_NAME)
        self.assertEqual(rc.find("hardware/param[@name='prim_path']").text, "/World/Arm")
        joint_rc = rc.find("joint[@name='joint1']")
        self.assertIsNotNone(joint_rc, "joint1 not in <ros2_control> block")
        self.assertIsNotNone(joint_rc.find("command_interface[@name='position']"))
        self.assertEqual(
            {s.get("name") for s in joint_rc.findall("state_interface")}, {"position", "velocity", "effort"}
        )

        self._assert_movable_joints_have_limits(root)
        self._assert_single_root_tree(root)

    async def test_velocity_drive_classified_as_velocity(self):
        root = ET.fromstring(build_full_urdf(self._build_arm(stiffness=0.0, damping=5.0), "/World/Arm"))
        joint_rc = root.find("ros2_control/joint[@name='joint1']")
        self.assertIsNotNone(joint_rc.find("command_interface[@name='velocity']"))
        self.assertEqual(len(joint_rc.findall("command_interface")), 3)

    async def test_effort_drive_classified_as_effort(self):
        root = ET.fromstring(build_full_urdf(self._build_arm(stiffness=0.0, damping=0.0), "/World/Arm"))
        joint_rc = root.find("ros2_control/joint[@name='joint1']")
        self.assertIsNotNone(joint_rc.find("command_interface[@name='effort']"))

    async def test_prismatic_arm_block_and_tree(self):
        root = ET.fromstring(build_full_urdf(self._build_arm(prismatic=True, stiffness=500.0), "/World/Arm"))
        joint_rc = root.find("ros2_control/joint[@name='joint1']")
        self.assertIsNotNone(joint_rc.find("command_interface[@name='position']"))
        self._assert_movable_joints_have_limits(root)
        self._assert_single_root_tree(root)

    async def test_mesh_reference_resolves_to_exported_file(self):
        root = ET.fromstring(build_full_urdf(self._build_arm(with_mesh=True), "/World/Arm"))
        mesh = root.find(".//mesh")
        self.assertIsNotNone(mesh)
        uri = mesh.get("filename")
        self.assertIsNotNone(uri)
        self.assertTrue(uri.startswith("file://"), uri)
        self.assertTrue(os.path.isfile(unquote(uri.removeprefix("file://"))), uri)

    async def test_malformed_overlay_reports_postprocessing_failure(self):
        fd, overlay_path = tempfile.mkstemp(suffix=".urdf")
        os.close(fd)
        self.addCleanup(os.remove, overlay_path)
        with open(overlay_path, "w", encoding="utf-8") as overlay:
            overlay.write("<robot><ros2_control></robot>")

        with self.assertRaisesRegex(RuntimeError, "URDF export/postprocessing failed"):
            build_full_urdf(self._build_arm(), "/World/Arm", sensor_overlay_urdf_path=overlay_path)

    def _build_loop_arm(self, root: str = "/World/Loop"):
        """Fixed-base chain base->a->b->c with a loop-closing joint c->a marked
        excludeFromArticulation. The converter must emit the closure as a
        <loop_joint>, leaving the <joint> graph an acyclic single-root tree."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        arm = UsdGeom.Xform.Define(stage, root).GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(arm)
        base = _make_link(stage, f"{root}/base")
        UsdPhysics.FixedJoint.Define(stage, f"{root}/base_fixed").CreateBody1Rel().SetTargets([base.GetPath()])
        link_a = _make_link(stage, f"{root}/link_a")
        link_b = _make_link(stage, f"{root}/link_b")
        link_c = _make_link(stage, f"{root}/link_c")
        _make_drive_joint(stage, f"{root}/joint_a", base, link_a, stiffness=2000.0)
        _make_drive_joint(stage, f"{root}/joint_b", link_a, link_b, stiffness=2000.0)
        _make_drive_joint(stage, f"{root}/joint_c", link_b, link_c, stiffness=2000.0)
        closing = UsdPhysics.FixedJoint.Define(stage, f"{root}/loop_close")
        closing.CreateBody0Rel().SetTargets([link_c.GetPath()])
        closing.CreateBody1Rel().SetTargets([link_a.GetPath()])
        UsdPhysics.Joint(closing.GetPrim()).CreateExcludeFromArticulationAttr().Set(True)
        return stage

    async def test_closed_chain_yields_single_root_tree(self):
        root = ET.fromstring(build_full_urdf(self._build_loop_arm(), "/World/Loop"))
        joint_names = {j.get("name") for j in root.findall("joint")}
        self.assertTrue(
            {"joint_a", "joint_b", "joint_c"}.issubset(joint_names),
            f"chain joints missing from URDF <joint> tree: {joint_names}",
        )
        self.assertNotIn("loop_close", joint_names, "loop closure leaked into the URDF <joint> tree")
        self._assert_single_root_tree(root)


class TestMimicJoints(omni.kit.test.AsyncTestCase):
    """Mimic joints are read from the exporter's <mimic> tags and emitted state-only with
    mimic <param>s so IsaacSimSystem can enforce them on any physics backend."""

    async def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()

    async def test_extract_mimic_map_reads_mimic_tags(self):
        robot = _urdf(["base", "l1", "l2"], [("lead", "revolute", "base", "l1"), ("follow", "revolute", "l1", "l2")])
        follow = next(j for j in robot.findall("joint") if j.get("name") == "follow")
        ET.SubElement(follow, "mimic", {"joint": "lead", "multiplier": "-1.5", "offset": "0.25"})
        mimic_map = _extract_mimic_map(robot)
        self.assertEqual(mimic_map, {"follow": ("lead", -1.5, 0.25)})

    async def test_extract_mimic_map_defaults_multiplier_and_offset(self):
        robot = _urdf(["base", "l1"], [("follow", "revolute", "base", "l1")])
        follow = robot.find("joint")
        ET.SubElement(follow, "mimic", {"joint": "lead"})  # no multiplier/offset attrs
        self.assertEqual(_extract_mimic_map(robot), {"follow": ("lead", 1.0, 0.0)})

    async def test_extract_mimic_map_empty_without_mimic(self):
        robot = _urdf(["base", "l1"], [("j", "revolute", "base", "l1")])
        self.assertEqual(_extract_mimic_map(robot), {})

    async def test_synthesize_marks_mimic_state_only(self):
        UsdGeom.Xform.Define(self.stage, "/World")
        arm = UsdGeom.Xform.Define(self.stage, "/World/Arm").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(arm)
        _drive_joint(self.stage, "/World/Arm/lead", stiffness=2000.0)
        _drive_joint(self.stage, "/World/Arm/follow", stiffness=2000.0)
        block = synthesize_hardware_block(self.stage, arm, arm, mimic_map={"follow": ("lead", -1.0, 0.0)})
        by_name = {j.name: j for j in block.joints}
        self.assertEqual(by_name["lead"].command_kind, "position")
        self.assertIsNone(by_name["lead"].mimic)
        # The mimic follower is state-only with the relationship captured.
        self.assertIsNone(by_name["follow"].command_kind)
        self.assertEqual(by_name["follow"].mimic, "lead")
        self.assertEqual(by_name["follow"].mimic_multiplier, -1.0)

    async def test_hardware_block_xml_emits_mimic_params_and_no_command_interface(self):
        block = HardwareBlock(name="arm", articulation_prim_path="/World/Arm")
        block.joints.append(JointEntry(name="lead", command_kind="position"))
        block.joints.append(
            JointEntry(name="follow", command_kind=None, mimic="lead", mimic_multiplier=-1.0, mimic_offset=0.1)
        )
        el = hardware_block_to_xml(block)
        follow = next(j for j in el.findall("joint") if j.get("name") == "follow")
        # No command interface for a mimic follower.
        self.assertEqual(follow.findall("command_interface"), [])
        params = {p.get("name"): p.text for p in follow.findall("param")}
        self.assertEqual(params.get("mimic"), "lead")
        self.assertEqual(float(params["multiplier"]), -1.0)
        self.assertEqual(float(params["offset"]), 0.1)
        # State interfaces are still present so the follower reports in /joint_states.
        self.assertEqual({s.get("name") for s in follow.findall("state_interface")}, {"position", "velocity", "effort"})
        # Leader keeps its command interfaces.
        lead = next(j for j in el.findall("joint") if j.get("name") == "lead")
        self.assertEqual({c.get("name") for c in lead.findall("command_interface")}, {"position", "velocity", "effort"})
