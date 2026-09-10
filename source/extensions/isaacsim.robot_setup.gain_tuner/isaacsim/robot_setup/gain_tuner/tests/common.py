# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared USD articulation fixtures and math helpers for gain-tuner tests."""

from __future__ import annotations

import asyncio
import math
import os
from enum import Enum

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
import omni.timeline
import omni.usd
import usd.schema.isaac.robot_schema as robot_schema
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_setup import gain_tuner
from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdPhysics


class JointModality(Enum):
    """Joint type for the test articulation."""

    PRISMATIC = "prismatic"
    REVOLUTE = "revolute"


class DriveSubmodality(Enum):
    """Drive type for the joint."""

    FORCE = "force"
    ACCELERATION = "acceleration"


# PhysX stores revolute drive stiffness/damping in N·m/deg and N·m·s/deg (see asset importer utils).
_REVOLUTE_DRIVE_GAIN_USD_SCALE = math.pi / 180.0

# Root prim and joint used by the importer-layout fixture below.
IMPORTER_ROBOT_PATH = "/Robot"
IMPORTER_JOINT_PATH = "/Robot/Physics/joint0"


def write_importer_asset_layout(tmp_dir: str, *, variant_selection: str | None = None) -> str:
    """Write the URDF / MJCF importer asset layout on disk and return the root layer path.

    Mirrors what the importer produces: an interface layer whose robot prim
    *references* ``payloads/base.usda`` and carries a ``Physics`` variant set whose
    variants *payload* the matching ``payloads/Physics/<backend>.usda`` layer.  As
    the importer does, only ``maxForce`` is authored on the joint drive, leaving
    stiffness and damping at their schema fallbacks.

    Args:
        tmp_dir: Directory the asset is written into.
        variant_selection: ``Physics`` variant to select, or None to leave the set
            unselected the way the importer does.

    Returns:
        Path to the interface (root) layer.
    """
    physics_dir = os.path.join(tmp_dir, "payloads", "Physics")
    os.makedirs(physics_dir, exist_ok=True)
    base_path = os.path.join(tmp_dir, "payloads", "base.usda")
    physics_path = os.path.join(physics_dir, "physics.usda")
    root_path = os.path.join(tmp_dir, "robot.usda")

    base_stage = Usd.Stage.CreateNew(base_path)
    base_stage.SetDefaultPrim(UsdGeom.Xform.Define(base_stage, IMPORTER_ROBOT_PATH).GetPrim())
    base_stage.Save()

    physics_stage = Usd.Stage.CreateNew(physics_path)
    physics_stage.SetDefaultPrim(UsdGeom.Xform.Define(physics_stage, IMPORTER_ROBOT_PATH).GetPrim())
    UsdPhysics.RevoluteJoint.Define(physics_stage, IMPORTER_JOINT_PATH)
    drive = UsdPhysics.DriveAPI.Apply(physics_stage.GetPrimAtPath(IMPORTER_JOINT_PATH), "angular")
    drive.CreateMaxForceAttr(10.0)
    physics_stage.Save()

    # Backend overlays sublayer the neutral base physics layer.
    for backend in ("physx", "mujoco"):
        overlay = Sdf.Layer.CreateNew(os.path.join(physics_dir, f"{backend}.usda"))
        overlay.subLayerPaths.append("./physics.usda")
        overlay.Save()

    root_stage = Usd.Stage.CreateNew(root_path)
    robot = UsdGeom.Xform.Define(root_stage, IMPORTER_ROBOT_PATH).GetPrim()
    root_stage.SetDefaultPrim(robot)
    robot.GetReferences().AddReference("./payloads/base.usda")
    variant_set = robot.GetVariantSets().AddVariantSet("Physics")
    variant_set.AddVariant("none")
    for backend in ("physics", "physx", "mujoco"):
        variant_set.AddVariant(backend)
        variant_set.SetVariantSelection(backend)
        with variant_set.GetVariantEditContext():
            robot.GetPayloads().AddPayload(f"./payloads/Physics/{backend}.usda")
    variant_set.ClearVariantSelection()
    if variant_selection is not None:
        variant_set.SetVariantSelection(variant_selection)
    root_stage.Save()
    return root_path


def write_scene_referencing(scene_path: str, assets: dict[str, str]) -> str:
    """Write a scene layer referencing each asset under ``/World/<name>``.

    This is the shape a user assembles in Kit: the robot's interface layer is not
    the stage root, so its ``payloads/Physics/`` folder is reachable only through
    the composition arcs.

    Args:
        scene_path: Path of the scene layer to create.
        assets: Mapping of prim name under ``/World`` to asset layer path.

    Returns:
        ``scene_path``, for convenience.
    """
    scene_stage = Usd.Stage.CreateNew(scene_path)
    scene_stage.SetDefaultPrim(UsdGeom.Xform.Define(scene_stage, "/World").GetPrim())
    for name, asset_path in assets.items():
        prim = UsdGeom.Xform.Define(scene_stage, f"/World/{name}").GetPrim()
        prim.GetReferences().AddReference(asset_path)
    scene_stage.Save()
    return scene_path


def _set_world_translation(prim: Usd.Prim, translation: Gf.Vec3f) -> None:
    """Set world translation on a prim, reusing an existing translate op if present.

    Args:
        prim: Prim to translate.
        translation: World translation value to author.
    """
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(translation)
            return
    xformable.AddTranslateOp().Set(translation)


def _revolute_drive_stiffness_damping_si_to_usd(k_si: float, d_si: float) -> tuple[float, float]:
    """Convert SI revolute gains (N*m/rad, N*m*s/rad) to USD drive attribute units.

    Args:
        k_si: Stiffness in SI units.
        d_si: Damping in SI units.

    Returns:
        ``(stiffness, damping)`` in USD drive attribute units.
    """
    s = _REVOLUTE_DRIVE_GAIN_USD_SCALE
    return k_si * s, d_si * s


def _revolute_drive_stiffness_damping_usd_to_si(k_usd: float, d_usd: float) -> tuple[float, float]:
    """Convert USD revolute drive gains back to SI.

    Args:
        k_usd: Stiffness in USD drive attribute units.
        d_usd: Damping in USD drive attribute units.

    Returns:
        ``(stiffness, damping)`` in SI units.
    """
    s = _REVOLUTE_DRIVE_GAIN_USD_SCALE
    return k_usd / s, d_usd / s


def _compute_stiffness_damping_prismatic(
    mass: float, natural_freq_hz: float, damping_ratio: float
) -> tuple[float, float]:
    """Compute stiffness and damping for prismatic joint from natural freq and damping ratio.

    For m*x'' + D*x' + K*x = 0:
    w_n = sqrt(K/m) => K = m * w_n^2
    zeta = D / (2*sqrt(m*K)) => D = 2*zeta*sqrt(m*K) = 2*zeta*m*w_n

    Args:
        mass: Mass of the prismatic joint link.
        natural_freq_hz: Natural frequency in Hz.
        damping_ratio: Damping ratio.

    Returns:
        Tuple of stiffness and damping values.
    """
    w_n = 2.0 * math.pi * natural_freq_hz
    stiffness = mass * (w_n**2)
    damping = 2.0 * damping_ratio * mass * w_n
    return stiffness, damping


def _compute_stiffness_damping_revolute(
    inertia: float, natural_freq_hz: float, damping_ratio: float
) -> tuple[float, float]:
    """Compute stiffness and damping for revolute joint from natural freq and damping ratio.

    For I*theta'' + D*theta' + K*theta = 0:
    w_n = sqrt(K/I) => K = I * w_n^2
    zeta = D / (2*sqrt(I*K)) => D = 2*zeta*I*w_n

    Args:
        inertia: Moment of inertia for the revolute joint.
        natural_freq_hz: Natural frequency in Hz.
        damping_ratio: Damping ratio.

    Returns:
        Tuple of stiffness and damping values.
    """
    w_n = 2.0 * math.pi * natural_freq_hz
    stiffness = inertia * (w_n**2)
    damping = 2.0 * damping_ratio * inertia * w_n
    return stiffness, damping


def _compute_natural_freq_damping_revolute(stiffness: float, damping: float, inertia: float) -> tuple[float, float]:
    """Compute natural frequency (Hz) and damping ratio from revolute drive gains.

    Inverse of _compute_stiffness_damping_revolute: w_n = sqrt(K/I), zeta = D/(2*sqrt(I*K)).

    Args:
        stiffness: Stiffness value of the revolute joint drive.
        damping: Damping value of the revolute joint drive.
        inertia: Moment of inertia.

    Returns:
        Tuple of natural frequency in Hz and damping ratio.
    """
    if inertia <= 0 or stiffness <= 0:
        return 0.0, 0.0
    w_n = math.sqrt(stiffness / inertia)
    natural_freq_hz = w_n / (2.0 * math.pi)
    zeta = damping / (2.0 * math.sqrt(inertia * stiffness))
    return natural_freq_hz, zeta


def _compute_natural_freq_damping_prismatic(stiffness: float, damping: float, mass: float) -> tuple[float, float]:
    """Compute natural frequency (Hz) and damping ratio from prismatic drive gains.

    Inverse of _compute_stiffness_damping_prismatic: w_n = sqrt(K/m), zeta = D/(2*sqrt(m*K)).

    Args:
        stiffness: Stiffness value of the prismatic joint drive.
        damping: Damping value of the prismatic joint drive.
        mass: Mass of the link.

    Returns:
        Tuple of natural frequency in Hz and damping ratio.
    """
    if mass <= 0 or stiffness <= 0:
        return 0.0, 0.0
    w_n = math.sqrt(stiffness / mass)
    natural_freq_hz = w_n / (2.0 * math.pi)
    zeta = damping / (2.0 * math.sqrt(mass * stiffness))
    return natural_freq_hz, zeta


class TestGainTunerHarness(omni.kit.test.AsyncTestCase):
    """Shared USD articulation fixtures and helpers for gain-tuner extension tests."""

    __test__ = False  # Not a test suite; concrete subclasses collect ``test_*`` methods.

    async def setUp(self) -> None:
        """Set up test environment with physics timeline and gain tuner."""
        self._physics_fps = 60
        self._physics_dt = 1.0 / self._physics_fps
        self._timeline = omni.timeline.get_timeline_interface()
        self._gain_tuner = gain_tuner.GainTuner()
        await stage_utils.create_new_stage_async()
        self._stage = omni.usd.get_context().get_stage()
        self._stage.DefinePrim(Sdf.Path("/World"), "Xform")
        PhysicsSchemaTools.addGroundPlane(
            self._stage, "/World/groundPlane", "Z", 100, Gf.Vec3f(0, 0, -0.5), Gf.Vec3f(1.0)
        )
        SimulationManager.set_physics_dt(self._physics_dt)
        await app_utils.update_app_async()

    async def tearDown(self) -> None:
        """Tear down test environment and reset gain tuner."""
        self._timeline.stop()
        self._gain_tuner.reset()
        await app_utils.update_app_async()

        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)

        stage_utils.close_stage()

    def _create_articulation(
        self,
        joint_modalities: list[JointModality],
        drive_submodality: DriveSubmodality,
        distance: float,
        mass: float,
        inertia_diag: float,
        natural_freq_hz: float,
        damping_ratio: float,
        *,
        chain: bool = False,
        fixed_base: bool = True,
        joint_axes: list[str] | None = None,
        link_positions: list[tuple[float, float, float]] | None = None,
        base_mass: float = 1000.0,
        base_inertia: float = 1.0,
        collision_enabled: bool = True,
        joint_limit_revolute: tuple[float, float] | None = None,
        joint_limit_prismatic: tuple[float, float] | None = None,
    ) -> str:
        """Create an articulation: fixed or floating base + one or more links connected by joints.

        Args:
            joint_modalities: Type of each joint (revolute or prismatic).
            drive_submodality: Force or acceleration drive.
            distance: Spacing used for default link positions (link i at (i+1)*distance, 0, 0.5).
            mass: Mass of each link.
            inertia_diag: Diagonal inertia of each link (and of base when fixed_base=False).
            natural_freq_hz: Target natural frequency for drive gains.
            damping_ratio: Target damping ratio for drive gains.
            chain: If True, joint i connects (base if i==0 else link_{i-1}) to link_i. If False, every joint
                connects base to link_i.
            fixed_base: If True, add a fixed joint to world and ArticulationRootAPI on it. If False, no fixed
                joint and ArticulationRootAPI on base.
            joint_axes: Per-joint axis (e.g. ["Z", "Y"] for revolute). Default: revolute "Z", prismatic "X".
            link_positions: Optional (x, y, z) for each link; default (i+1)*distance along X at z=0.5.
            base_mass: Mass of base link (used when fixed_base=False for floating base).
            base_inertia: Diagonal inertia of base (used when fixed_base=False for floating base).
            collision_enabled: When False, skip collision API on base and links (free-flight oscillation tests).
            joint_limit_revolute: Optional ``(lower, upper)`` limits in degrees for revolute joints.
            joint_limit_prismatic: Optional ``(lower, upper)`` limits in meters for prismatic joints.

        Returns:
            Robot prim path.
        """
        robot_path = "/World/robot"
        if self._stage.GetPrimAtPath(robot_path).IsValid():
            stage_utils.delete_prim(robot_path)

        base_path = f"{robot_path}/base"
        fixed_joint_path = f"{robot_path}/root_joint"
        n_joints = len(joint_modalities)
        axes = (
            joint_axes
            if joint_axes is not None
            else (["Z" if m == JointModality.REVOLUTE else "X" for m in joint_modalities])
        )

        # Base
        base_geom = UsdGeom.Cube.Define(self._stage, base_path)
        base_geom.CreateSizeAttr(0.1)
        base_prim = self._stage.GetPrimAtPath(base_path)
        _set_world_translation(base_prim, Gf.Vec3f(0, 0, 0.5))
        if collision_enabled:
            UsdPhysics.CollisionAPI.Apply(base_prim)
        UsdPhysics.RigidBodyAPI.Apply(base_prim)
        UsdPhysics.MassAPI.Apply(base_prim)
        UsdPhysics.MassAPI(base_prim).CreateMassAttr(base_mass)
        UsdPhysics.MassAPI(base_prim).CreateDiagonalInertiaAttr(Gf.Vec3f(base_inertia, base_inertia, base_inertia))

        if fixed_base:
            fixed_joint = UsdPhysics.FixedJoint.Define(self._stage, fixed_joint_path)
            fixed_joint.CreateBody1Rel().SetTargets([base_path])
            UsdPhysics.ArticulationRootAPI.Apply(fixed_joint.GetPrim())
        else:
            fixed_joint = None
            UsdPhysics.ArticulationRootAPI.Apply(base_prim)

        link_prims = []
        joint_prims = []

        base_pos = Gf.Vec3f(0, 0, 0.5)
        body0_positions = {}
        body0_positions[base_path] = base_pos

        for joint_index, joint_modality in enumerate(joint_modalities):
            link_path = f"{robot_path}/link_{joint_index}"
            if link_positions is not None and joint_index < len(link_positions):
                pos = Gf.Vec3f(*link_positions[joint_index])
            else:
                pos = Gf.Vec3f((joint_index + 1) * distance, 0, 0.5)
            link_geom = UsdGeom.Cube.Define(self._stage, link_path)
            link_geom.CreateSizeAttr(0.2)
            link_prim = self._stage.GetPrimAtPath(link_path)
            _set_world_translation(link_prim, pos)
            if collision_enabled:
                UsdPhysics.CollisionAPI.Apply(link_prim)
            UsdPhysics.RigidBodyAPI.Apply(link_prim)
            UsdPhysics.MassAPI.Apply(link_prim)
            UsdPhysics.MassAPI(link_prim).CreateMassAttr(mass)
            UsdPhysics.MassAPI(link_prim).CreateDiagonalInertiaAttr(Gf.Vec3f(inertia_diag, inertia_diag, inertia_diag))
            link_prims.append(link_prim)
            body0_positions[link_path] = pos

            body0 = base_path if (not chain or joint_index == 0) else f"{robot_path}/link_{joint_index - 1}"
            body0_pos = body0_positions[body0]
            axis = (
                axes[joint_index]
                if joint_index < len(axes)
                else ("Z" if joint_modality == JointModality.REVOLUTE else "X")
            )

            if joint_modality == JointModality.PRISMATIC:
                joint = UsdPhysics.PrismaticJoint.Define(self._stage, f"{robot_path}/joint_{joint_index}")
                joint.CreateAxisAttr(axis)
                joint.CreateBody0Rel().SetTargets([body0])
                joint.CreateBody1Rel().SetTargets([link_path])
                drive_type = "linear"
                stiffness, damping = _compute_stiffness_damping_prismatic(mass, natural_freq_hz, damping_ratio)
            else:
                joint = UsdPhysics.RevoluteJoint.Define(self._stage, f"{robot_path}/joint_{joint_index}")
                joint.CreateAxisAttr(axis)
                joint.CreateBody0Rel().SetTargets([body0])
                joint.CreateBody1Rel().SetTargets([link_path])
                drive_type = "angular"
                stiffness, damping = _compute_stiffness_damping_revolute(inertia_diag, natural_freq_hz, damping_ratio)
                stiffness, damping = _revolute_drive_stiffness_damping_si_to_usd(stiffness, damping)

            local_pos1 = Gf.Vec3f(body0_pos[0] - pos[0], body0_pos[1] - pos[1], body0_pos[2] - pos[2])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
            joint.CreateLocalPos1Attr().Set(local_pos1)

            drive_api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive_type)
            drive_api.CreateTypeAttr(drive_submodality.value)
            drive_api.CreateStiffnessAttr(stiffness)
            drive_api.CreateDampingAttr(damping)
            # Free-flight harness: default drive effort limits can saturate PD and yield a much
            # lower apparent oscillation frequency than the authored K/D would predict.
            if not collision_enabled:
                if not drive_api.GetMaxForceAttr():
                    drive_api.CreateMaxForceAttr(1.0e7)
                else:
                    drive_api.GetMaxForceAttr().Set(1.0e7)
            if joint_modality == JointModality.REVOLUTE and joint_limit_revolute is not None:
                joint.CreateLowerLimitAttr(joint_limit_revolute[0])
                joint.CreateUpperLimitAttr(joint_limit_revolute[1])
            if joint_modality == JointModality.PRISMATIC and joint_limit_prismatic is not None:
                joint.CreateLowerLimitAttr(joint_limit_prismatic[0])
                joint.CreateUpperLimitAttr(joint_limit_prismatic[1])
            joint_prims.append(joint.GetPrim())

        # Robot schema: all links and all joints (fixed first when present)
        robot_prim = self._stage.GetPrimAtPath(robot_path)
        robot_schema.ApplyRobotAPI(robot_prim)
        all_links = [base_prim.GetPath()] + [p.GetPath() for p in link_prims]
        all_joints = ([fixed_joint.GetPrim().GetPath()] if fixed_joint is not None else []) + [
            p.GetPath() for p in joint_prims
        ]
        robot_prim.GetRelationship(robot_schema.Relations.ROBOT_LINKS.name).SetTargets(all_links)
        robot_prim.GetRelationship(robot_schema.Relations.ROBOT_JOINTS.name).SetTargets(all_joints)
        for p in [base_prim] + link_prims:
            robot_schema.ApplyLinkAPI(p)
        for j in ([] if fixed_joint is None else [fixed_joint.GetPrim()]) + joint_prims:
            robot_schema.ApplyJointAPI(j)

        return robot_path

    async def _run_setup_and_compute_inertia(self, robot_path: str, num_physics_steps: int = 60) -> None:
        """Setup gain tuner, run physics so mass query completes, then compute joint inertias.

        Args:
            robot_path: USD path to the robot prim.
            num_physics_steps: Number of physics steps to run before computing inertia.
        """
        self._gain_tuner.setup(robot_path)
        for _ in range(2):
            await app_utils.update_app_async()
        self._timeline.play()
        for _ in range(num_physics_steps):
            await app_utils.update_app_async()
        self._gain_tuner.compute_joints_accumulated_inertia()
        self._timeline.stop()

    def _create_fixed_base_two_revolute_chain(
        self,
        distance: float,
        mass: float,
        inertia_diag: float,
        natural_freq_hz: float,
        damping_ratio: float,
        second_axis_z: bool = True,
    ) -> tuple[str, list[float]]:
        """Create fixed base -> revolute0 -> link0 -> revolute1 -> link1. Same plane if second_axis_z True.

        Args:
            distance: Distance between links.
            mass: Mass of each link.
            inertia_diag: Diagonal inertia component for each link.
            natural_freq_hz: Natural frequency in Hz for the drive.
            damping_ratio: Damping ratio for the drive.
            second_axis_z: Whether the second joint axis is Z (True) or Y (False).

        Returns:
            Tuple of robot path and list of expected equivalent inertias.
        """
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE, JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=distance,
            mass=mass,
            inertia_diag=inertia_diag,
            natural_freq_hz=natural_freq_hz,
            damping_ratio=damping_ratio,
            chain=True,
            joint_axes=["Z", "Z" if second_axis_z else "Y"],
        )
        # j0: pose at base; forward = link0+link1 about base -> I_fwd_j0 = (I_d+m*d^2)+(I_d+m*(2d)^2) = 3.25.
        I_fwd_j0 = (inertia_diag + mass * distance**2) + (inertia_diag + mass * (2 * distance) ** 2)
        # j1: backward_links = [link0, base]; base is fixed so _accumulate_link_inertia returns (0,0,True).
        # I_eq = forward only. Joint pose at body0 (link0); forward = link1 about j1 = I_d + m*d^2 = 1.25.
        I_eq_j1 = inertia_diag + mass * distance**2
        return robot_path, [I_fwd_j0, I_eq_j1]

    def _create_fixed_base_two_prismatic_chain(
        self,
        distance: float,
        mass: float,
        natural_freq_hz: float,
        damping_ratio: float,
        same_axis: bool = True,
    ) -> tuple[str, list[float]]:
        """Create fixed base -> prism0 -> link0 -> prism1 -> link1. Same axis X if same_axis else second Y.

        Args:
            distance: Distance between links.
            mass: Mass of each link.
            natural_freq_hz: Natural frequency in Hz for the drive.
            damping_ratio: Damping ratio for the drive.
            same_axis: Whether both joints share the same X axis.

        Returns:
            Tuple of robot path and list of expected equivalent inertias.
        """
        link_positions = None if same_axis else [(distance, 0, 0.5), (distance, distance, 0.5)]
        robot_path = self._create_articulation(
            [JointModality.PRISMATIC, JointModality.PRISMATIC],
            DriveSubmodality.FORCE,
            distance=distance,
            mass=mass,
            inertia_diag=1.0,
            natural_freq_hz=natural_freq_hz,
            damping_ratio=damping_ratio,
            chain=True,
            joint_axes=["X", "X" if same_axis else "Y"],
            link_positions=link_positions,
        )
        # j0: backward fixed, forward = m0 + m1 = 2*mass. j1: backward chain includes fixed base
        # -> I_eq = forward mass only = mass.
        return robot_path, [2.0 * mass, mass]

    def _create_moving_base_single_joint(
        self,
        joint_revolute: bool,
        distance: float,
        base_mass: float,
        base_inertia: float,
        link_mass: float,
        link_inertia_diag: float,
        natural_freq_hz: float,
        damping_ratio: float,
    ) -> tuple[str, list[float]]:
        """Create floating base -> single joint -> link. No fixed joint. ArticulationRootAPI on base.

        Args:
            joint_revolute: Whether the joint is revolute (True) or prismatic (False).
            distance: Distance between links.
            base_mass: Mass of the base link.
            base_inertia: Inertia of the base link.
            link_mass: Mass of the child link.
            link_inertia_diag: Diagonal inertia component for the child link.
            natural_freq_hz: Natural frequency in Hz for the drive.
            damping_ratio: Damping ratio for the drive.

        Returns:
            Tuple of robot path and list of expected equivalent inertias.
        """
        modality = JointModality.REVOLUTE if joint_revolute else JointModality.PRISMATIC
        robot_path = self._create_articulation(
            [modality],
            DriveSubmodality.FORCE,
            distance=distance,
            mass=link_mass,
            inertia_diag=link_inertia_diag,
            natural_freq_hz=natural_freq_hz,
            damping_ratio=damping_ratio,
            fixed_base=False,
            base_mass=base_mass,
            base_inertia=base_inertia,
        )
        if joint_revolute:
            I_eq = (base_inertia * link_inertia_diag) / (base_inertia + link_inertia_diag)
        else:
            I_eq = (base_mass * link_mass) / (base_mass + link_mass)
        return robot_path, [I_eq]

    def _create_fixed_base_revolute_prismatic_chain(
        self,
        distance: float,
        mass: float,
        inertia_diag: float,
        natural_freq_hz: float,
        damping_ratio: float,
    ) -> tuple[str, list[float]]:
        """Create fixed base -> revolute -> link0 -> prismatic -> link1.

        Args:
            distance: Distance between links.
            mass: Mass of each link.
            inertia_diag: Diagonal inertia component for each link.
            natural_freq_hz: Natural frequency in Hz for the drive.
            damping_ratio: Damping ratio for the drive.

        Returns:
            Tuple of robot path and list of expected equivalent inertias.
        """
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE, JointModality.PRISMATIC],
            DriveSubmodality.FORCE,
            distance=distance,
            mass=mass,
            inertia_diag=inertia_diag,
            natural_freq_hz=natural_freq_hz,
            damping_ratio=damping_ratio,
            chain=True,
            joint_axes=["Z", "X"],
        )
        # j0 at base: I_fwd = (I_d + m*d^2) + (I_d + m*(2d)^2) = 3.25. j1 prismatic: backward includes
        # fixed base -> I_eq = forward mass = mass.
        I_eq_j0 = (inertia_diag + mass * distance**2) + (inertia_diag + mass * (2 * distance) ** 2)
        I_eq_j1 = mass
        return robot_path, [I_eq_j0, I_eq_j1]
