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

"""USD → ros2_control URDF synthesis.

Walks USD under a target prim, requires exactly one ArticulationRoot, inspects its
driven joints, emits one `<ros2_control>` block, and splices it into a kinematic
URDF from isaacsim.asset.exporter.urdf. The result is suitable as
`robot_description` for ros2_control's controller_manager.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

import carb
from pxr import Usd, UsdPhysics


@dataclass
class JointEntry:
    """A driven joint's command kind and optional mimic relationship."""

    name: str
    command_kind: str | None  # "position" | "velocity" | "effort" | None (state-only)
    mimic: str | None = None  # leader joint name if this joint mimics another, else None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


@dataclass
class HardwareBlock:
    """One ``<ros2_control>`` block: an articulation and the joints it exposes."""

    name: str
    articulation_prim_path: str
    joints: list[JointEntry] = field(default_factory=list)


def _detect_command_kind(joint_prim: Usd.Prim) -> str | None:
    """Return the joint's DriveAPI command kind ("position"/"velocity"/"effort"), or None if state-only."""
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        token = "angular"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        token = "linear"
    else:
        return None
    if not joint_prim.HasAPI(UsdPhysics.DriveAPI, token):
        return None
    drv = UsdPhysics.DriveAPI(joint_prim, token)
    stiffness = drv.GetStiffnessAttr().Get()
    damping = drv.GetDampingAttr().Get()
    if stiffness is None:
        stiffness = 0.0
    if damping is None:
        damping = 0.0
    if stiffness > 0:
        return "position"
    if damping > 0:
        return "velocity"
    return "effort"


def discover_articulations(stage: Usd.Stage, target_prim_path: str) -> list[Usd.Prim]:
    """Return every prim with ArticulationRootAPI under ``target_prim_path`` (inclusive)."""
    target = stage.GetPrimAtPath(target_prim_path)
    if not target.IsValid():
        raise ValueError(f"targetPrim {target_prim_path!r} not found in stage")
    out = [p for p in Usd.PrimRange(target) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not out:
        raise ValueError(f"no ArticulationRootAPI prim under {target_prim_path!r}")
    return out


def synthesize_hardware_block(
    stage: Usd.Stage,
    articulation_root: Usd.Prim,
    joint_scope: Usd.Prim,
    mimic_map: dict[str, tuple[str, float, float]] | None = None,
) -> HardwareBlock:
    """Build the ``<ros2_control>`` HardwareBlock for one articulation.

    Joints are walked from ``joint_scope`` (the user's targetPrim); in USD they
    often live as siblings rather than descendants of the articulation root. Only
    revolute/prismatic joints get an entry. ``mimic_map`` maps a follower joint name
    to (leader, multiplier, offset); mimic joints are exported state-only (driven by
    their leader inside IsaacSimSystem, not by a controller).
    """
    mimic_map = mimic_map or {}
    block = HardwareBlock(
        name=articulation_root.GetName() or "isaac_sim_hw",
        articulation_prim_path=str(articulation_root.GetPath()),
    )
    for prim in Usd.PrimRange(joint_scope):
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            name = prim.GetName()
            mimic = mimic_map.get(name)
            if mimic is not None:
                leader, multiplier, offset = mimic
                block.joints.append(
                    JointEntry(
                        name=name,
                        command_kind=None,
                        mimic=leader,
                        mimic_multiplier=multiplier,
                        mimic_offset=offset,
                    )
                )
            else:
                block.joints.append(JointEntry(name=name, command_kind=_detect_command_kind(prim)))
    return block


HARDWARE_PLUGIN_NAME = "isaacsim_ros2_control/IsaacSimSystem"


def hardware_block_to_xml(block: HardwareBlock) -> ET.Element:
    """Build the ``<ros2_control>`` XML element for a hardware block."""
    el = ET.Element("ros2_control", {"name": block.name, "type": "system"})
    hw = ET.SubElement(el, "hardware")
    ET.SubElement(hw, "plugin").text = HARDWARE_PLUGIN_NAME
    ET.SubElement(hw, "param", {"name": "prim_path"}).text = block.articulation_prim_path
    for j in block.joints:
        jel = ET.SubElement(el, "joint", {"name": j.name})
        # Driven joints export all three command interfaces; the active one is
        # selected at runtime via perform_command_mode_switch. Mimic joints export
        # none: IsaacSimSystem drives them from their leader.
        if j.command_kind is not None:
            for kind in ("position", "velocity", "effort"):
                ET.SubElement(jel, "command_interface", {"name": kind})
        # Mimic relationship forwarded to IsaacSimSystem, which applies it
        # engine-agnostically (ros2_control's own mimic handling exists on jazzy
        # but not humble). Param names match ros2_control's mock convention.
        if j.mimic is not None:
            ET.SubElement(jel, "param", {"name": "mimic"}).text = j.mimic
            ET.SubElement(jel, "param", {"name": "multiplier"}).text = repr(j.mimic_multiplier)
            ET.SubElement(jel, "param", {"name": "offset"}).text = repr(j.mimic_offset)
        ET.SubElement(jel, "state_interface", {"name": "position"})
        ET.SubElement(jel, "state_interface", {"name": "velocity"})
        ET.SubElement(jel, "state_interface", {"name": "effort"})
    return el


def _stable_export_dir(target_prim_path: str) -> str:
    """Return a stable per-prim export directory under the system temp root.

    The directory is cleared and recreated on each call so mesh files from a
    prior Play do not accumulate. One dir per (process, prim path); the PID keeps
    concurrent Isaac Sim instances from clobbering each other's exports.
    Lifetime: until the next export for that prim or external/system temporary-file cleanup.
    """
    tag = hashlib.sha1(target_prim_path.encode()).hexdigest()[:12]
    d = os.path.join(tempfile.gettempdir(), f"ros2_control_urdf_{os.getpid()}_{tag}")
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    return d


def _export_kinematic_urdf(
    stage: Usd.Stage,
    target_prim_path: str,
    postprocess: Callable[[ET.Element], None],
) -> str:
    from isaacsim.asset.exporter.urdf.converter.usd_to_urdf import UsdToUrdfConverter

    export_dir = _stable_export_dir(target_prim_path)
    output_path = os.path.join(export_dir, "robot.urdf")
    converter = UsdToUrdfConverter(stage, target_prim_path, mesh_path_prefix="file://")
    try:
        converter.convert(output_path, postprocess=postprocess)
        with open(output_path, encoding="utf-8") as urdf_file:
            return urdf_file.read()
    except Exception as e:
        raise RuntimeError(f"URDF export/postprocessing failed for {target_prim_path!r}: {e}") from e


def _splice_sensor_overlay(root: ET.Element, overlay_urdf_path: str) -> None:
    """Splice ``<sensor>``/``<param>`` overlay elements into the matching ``<ros2_control>`` block."""
    if not overlay_urdf_path:
        return
    if not os.path.isfile(overlay_urdf_path):
        carb.log_warn(f"urdfPath {overlay_urdf_path!r} does not exist; skipping sensor overlay")
        return
    overlay_root = ET.parse(overlay_urdf_path).getroot()
    for src_rc in overlay_root.findall("ros2_control"):
        dst_rc = root.find(f"ros2_control[@name='{src_rc.get('name')}']")
        if dst_rc is None:
            dst_rc = root.find("ros2_control")
        if dst_rc is None:
            continue
        for sensor in src_rc.findall("sensor"):
            dst_rc.append(sensor)
        src_hw = src_rc.find("hardware")
        dst_hw = dst_rc.find("hardware")
        if src_hw is not None and dst_hw is not None:
            for param in src_hw.findall("param"):
                if param.get("name") == "prim_path":
                    continue  # we own prim_path
                dst_hw.append(param)


# URDF parsing requires effort/velocity values. These defaults do not alter the
# USD DriveAPI configuration, but downstream URDF consumers may interpret them.
_DEFAULT_EFFORT = "1000.0"
_DEFAULT_VELOCITY = "10.0"
_DEFAULT_REVOLUTE_LOWER = "-3.141592653589793"
_DEFAULT_REVOLUTE_UPPER = "3.141592653589793"
_DEFAULT_PRISMATIC_LOWER = "-1.0"
_DEFAULT_PRISMATIC_UPPER = "1.0"


def _is_finite_float(value: str | None) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def _fix_joint_limits(root: ET.Element) -> None:
    for joint in root.findall("joint"):
        joint_type = joint.get("type")
        if joint_type not in ("continuous", "revolute", "prismatic"):
            continue
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        if joint_type == "revolute":
            default_lower = _DEFAULT_REVOLUTE_LOWER
            default_upper = _DEFAULT_REVOLUTE_UPPER
        elif joint_type == "prismatic":
            default_lower = _DEFAULT_PRISMATIC_LOWER
            default_upper = _DEFAULT_PRISMATIC_UPPER
        else:
            default_lower = None
            default_upper = None
        if default_lower is not None and not _is_finite_float(limit.get("lower")):
            limit.set("lower", default_lower)
        if default_upper is not None and not _is_finite_float(limit.get("upper")):
            limit.set("upper", default_upper)
        # `urdfdom` rejects non-finite limits, so replace values such as `inf` from unlimited USD drives.
        if not _is_finite_float(limit.get("effort")):
            limit.set("effort", _DEFAULT_EFFORT)
        if not _is_finite_float(limit.get("velocity")):
            limit.set("velocity", _DEFAULT_VELOCITY)


def _prune_to_control_tree(root: ET.Element, control_joint_names: list[str]) -> None:
    """Keep the URDF tree rooted at the ros2_control joint chain.

    The USD exporter may emit an extra world/floating-base component for
    fixed-base articulations. controller_manager requires a single URDF root, so
    prefer the tree that contains the joints exported into <ros2_control>.
    """
    if not control_joint_names:
        return

    joints = list(root.findall("joint"))
    joints_by_name = {j.get("name"): j for j in joints if j.get("name")}
    control_joints = [joints_by_name[name] for name in control_joint_names if name in joints_by_name]
    if not control_joints:
        return

    def parent_child(joint: ET.Element) -> tuple[str | None, str | None]:
        parent = joint.find("parent")
        child = joint.find("child")
        return (
            parent.get("link") if parent is not None else None,
            child.get("link") if child is not None else None,
        )

    control_edges = [(p, c) for p, c in (parent_child(j) for j in control_joints) if p and c]
    if not control_edges:
        return

    control_children = {c for _, c in control_edges}
    roots: list[str] = []
    for p, _ in control_edges:
        if p not in control_children and p not in roots:
            roots.append(p)
    if not roots:
        roots = [control_edges[0][0]]

    out_edges: dict[str, list[ET.Element]] = {}
    for joint in joints:
        parent, child = parent_child(joint)
        if parent and child:
            out_edges.setdefault(parent, []).append(joint)

    reachable_links = set(roots)
    reachable_joint_names: set[str] = set()
    queue = deque(roots)
    while queue:
        link = queue.popleft()
        for joint in out_edges.get(link, []):
            _, child = parent_child(joint)
            if child is None or child in reachable_links:
                continue
            reachable_links.add(child)
            name = joint.get("name")
            if name:
                reachable_joint_names.add(name)
            queue.append(child)

    if any(j.get("name") not in reachable_joint_names for j in control_joints):
        return

    for joint in joints:
        if joint.get("name") not in reachable_joint_names:
            root.remove(joint)
    for link in list(root.findall("link")):
        if link.get("name") not in reachable_links:
            root.remove(link)


# KDL (used by robot_state_publisher and MoveIt's kdl_kinematics_plugin) does
# not support inertia on the root link. Prepend a zero-mass dummy link connected
# by a fixed joint, as recommended by KDL's warning.
def _fix_root_link_inertia(root: ET.Element) -> None:
    links = root.findall("link")
    if not links:
        return
    child_links = {j.find("child").get("link") for j in root.findall("joint") if j.find("child") is not None}
    root_link = next((el for el in links if el.get("name") not in child_links), None)
    if root_link is None or root_link.find("inertial") is None:
        return

    original_root_name = root_link.get("name")
    dummy_name = f"{original_root_name}_kdl_dummy_root"

    dummy = ET.Element("link", {"name": dummy_name})
    fixed_joint = ET.Element("joint", {"name": f"{dummy_name}_to_{original_root_name}", "type": "fixed"})
    ET.SubElement(fixed_joint, "parent", {"link": dummy_name})
    ET.SubElement(fixed_joint, "child", {"link": original_root_name})
    root.insert(0, dummy)
    root.insert(1, fixed_joint)


def _extract_mimic_map(root: ET.Element) -> dict[str, tuple[str, float, float]]:
    """Map follower joint name to (leader, multiplier, offset) from URDF `<mimic>` tags.

    The kinematic URDF emitted by isaacsim.asset.exporter.urdf carries
    `<joint><mimic joint=".." multiplier=".." offset=".."/>` for joints with a USD
    PhysX mimic relationship. The relationship is read here and forwarded into the
    `<ros2_control>` block so IsaacSimSystem can enforce it on any physics backend.
    """
    out: dict[str, tuple[str, float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        mimic = joint.find("mimic")
        if not name or mimic is None:
            continue
        leader = mimic.get("joint")
        if not leader:
            continue
        try:
            multiplier = float(mimic.get("multiplier", "1.0"))
            offset = float(mimic.get("offset", "0.0"))
        except ValueError:
            carb.log_warn(f"Joint {name!r} has a non-numeric <mimic> multiplier/offset; using 1.0/0.0")
            multiplier, offset = 1.0, 0.0
        out[name] = (leader, multiplier, offset)
    return out


def build_full_urdf(
    stage: Usd.Stage,
    target_prim_path: str,
    sensor_overlay_urdf_path: str | None = None,
) -> str:
    """Live-export the kinematic URDF, patch USD quirks, append <ros2_control> blocks.

    Meshes are exported to a stable per-prim directory and referenced with
    absolute file:// paths so /robot_description consumers (RViz, RSP) resolve
    them without a ROS package.
    """
    target = stage.GetPrimAtPath(target_prim_path)
    roots = discover_articulations(stage, target_prim_path)
    # One articulation per node: emitting a block per root would duplicate every
    # joint into each block. Use a separate ROS2ControlManager node per robot.
    if len(roots) > 1:
        raise ValueError(
            f"{target_prim_path!r} contains {len(roots)} ArticulationRootAPI prims; "
            "multiple articulations under one targetPrim are not supported. "
            "Use one ROS2ControlManager node per articulation root."
        )

    def postprocess(root: ET.Element) -> None:
        mimic_map = _extract_mimic_map(root)
        block = synthesize_hardware_block(stage, roots[0], joint_scope=target, mimic_map=mimic_map)
        _prune_to_control_tree(root, [j.name for j in block.joints])
        _fix_joint_limits(root)
        _fix_root_link_inertia(root)
        root.append(hardware_block_to_xml(block))
        if sensor_overlay_urdf_path:
            _splice_sensor_overlay(root, sensor_overlay_urdf_path)

    return _export_kinematic_urdf(stage, target_prim_path, postprocess)
