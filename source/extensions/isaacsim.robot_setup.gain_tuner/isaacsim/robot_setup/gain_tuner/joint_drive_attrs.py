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

"""USD joint drive attribute helpers for gain tuning (core, UI-independent)."""

from __future__ import annotations

from enum import IntEnum

import pxr
from pxr import UsdPhysics


class JointDriveMode(IntEnum):
    """Drive mode for a joint in the gain tuner."""

    NONE = 0
    """No position or velocity drive is active."""

    POSITION = 1
    """Position drive (non-zero stiffness)."""

    VELOCITY = 2
    """Velocity-only drive (zero stiffness, non-zero damping)."""

    MIMIC = 3
    """Mimic joint (PhysX PhysxMimicJointAPI or Newton NewtonMimicAPI)."""


# Applied-schema tokens for mimic joints. PhysX uses a multi-apply schema whose
# applied name ends in the drive axis (e.g. ``PhysxMimicJointAPI:rotZ``). Newton
# uses a single-apply schema (``NewtonMimicAPI``) with no axis suffix and no
# natural-frequency / damping-ratio / damping tuning attributes.
_PHYSX_MIMIC_SCHEMA_TOKEN = "MimicJointAPI"
_NEWTON_MIMIC_SCHEMA_TOKEN = "NewtonMimicAPI"


def _get_physx_mimic_axis(joint: object) -> str | None:
    """Return the drive axis of an applied PhysX ``MimicJointAPI``, or None.

    Args:
        joint: The joint to inspect.

    Returns:
        The axis instance token (e.g. ``rotZ``) if a PhysX mimic schema is
        applied, otherwise None (including for Newton-only mimic joints).
    """
    physx_schemas = [a for a in joint.GetAppliedSchemas() if _PHYSX_MIMIC_SCHEMA_TOKEN in a]
    if not physx_schemas:
        return None
    return physx_schemas[-1].split(":")[-1]


def is_joint_mimic(joint: object) -> bool:
    """Check if a joint has a mimic joint API applied.

    Detects both the legacy PhysX ``PhysxMimicJointAPI`` (multi-apply, per-axis)
    and the current Newton ``NewtonMimicAPI`` (single-apply) schemas so mimic
    joints authored by either path are recognized.

    Args:
        joint: The joint to check.

    Returns:
        True if joint has a PhysX or Newton mimic API applied, False otherwise.
    """
    return any(
        _PHYSX_MIMIC_SCHEMA_TOKEN in a or a.startswith(_NEWTON_MIMIC_SCHEMA_TOKEN) for a in joint.GetAppliedSchemas()
    )


def get_mimic_natural_frequency_attr(joint: object) -> pxr.Usd.Attribute | None:
    """Get the natural frequency attribute for a mimic joint.

    Only PhysX mimic joints expose a natural-frequency attribute; Newton mimic
    joints have no tunable gains, so None is returned for them.

    Args:
        joint: The joint to get the attribute from.

    Returns:
        The natural frequency attribute for a PhysX mimic joint, None otherwise.
    """
    mimic_axis = _get_physx_mimic_axis(joint)
    if mimic_axis is not None:
        return joint.GetAttribute(f"physxMimicJoint:{mimic_axis}:naturalFrequency")
    return None


def get_mimic_damping_ratio_attr(joint: object) -> pxr.Usd.Attribute | None:
    """Get the damping ratio attribute for a mimic joint.

    Only PhysX mimic joints expose a damping-ratio attribute; Newton mimic
    joints have no tunable gains, so None is returned for them.

    Args:
        joint: The joint to get the attribute from.

    Returns:
        The damping ratio attribute for a PhysX mimic joint, None otherwise.
    """
    mimic_axis = _get_physx_mimic_axis(joint)
    if mimic_axis is not None:
        return joint.GetAttribute(f"physxMimicJoint:{mimic_axis}:dampingRatio")
    return None


def get_stiffness_attr(joint: object, drive_axis: object = None) -> pxr.Usd.Attribute | None:
    """Get the stiffness attribute for a joint.

    Args:
        joint: The joint to get the attribute from.
        drive_axis: Optional D6 drive axis token.

    Returns:
        The stiffness attribute if valid joint type, None otherwise.
    """
    if drive_axis:
        driveAPI = UsdPhysics.DriveAPI(joint, drive_axis)
        if driveAPI:
            return driveAPI.GetStiffnessAttr()
        return None
    if joint.IsA(UsdPhysics.RevoluteJoint):
        driveAPI = UsdPhysics.DriveAPI(joint, "angular")
        return driveAPI.GetStiffnessAttr()
    elif joint.IsA(UsdPhysics.PrismaticJoint):
        driveAPI = UsdPhysics.DriveAPI(joint, "linear")
        return driveAPI.GetStiffnessAttr()
    return None


def get_joint_drive_type_attr(joint: object, drive_axis: object = None) -> pxr.Usd.Attribute | None:
    """Get the drive type attribute for a joint.

    Args:
        joint: The joint to get the attribute from.
        drive_axis: Optional D6 drive axis token.

    Returns:
        The drive type attribute if valid joint type, None otherwise.
    """
    if drive_axis:
        driveAPI = UsdPhysics.DriveAPI(joint, drive_axis)
        if driveAPI:
            return driveAPI.GetTypeAttr()
        return None
    driveAPI = None
    if joint.IsA(UsdPhysics.RevoluteJoint):
        driveAPI = UsdPhysics.DriveAPI(joint, "angular")
    elif joint.IsA(UsdPhysics.PrismaticJoint):
        driveAPI = UsdPhysics.DriveAPI(joint, "linear")
    if driveAPI:
        return driveAPI.GetTypeAttr()
    return None


def get_damping_attr(joint: object, drive_axis: object = None) -> pxr.Usd.Attribute | None:
    """Get the damping attribute for a joint.

    Args:
        joint: The joint to get the attribute from.
        drive_axis: Optional D6 drive axis token.

    Returns:
        The damping attribute if valid joint type, None otherwise.
    """
    if is_joint_mimic(joint):
        # Mimic joints are constraint-driven and never expose a drive damping
        # attribute. PhysX mimic joints expose a dedicated damping attr; Newton
        # mimic joints have none, so return None rather than falling through to
        # the DriveAPI damping.
        mimic_axis = _get_physx_mimic_axis(joint)
        if mimic_axis is not None:
            return joint.GetAttribute(f"physxMimicJoint:{mimic_axis}:damping")
        return None
    if drive_axis:
        driveAPI = UsdPhysics.DriveAPI(joint, drive_axis)
        if driveAPI:
            return driveAPI.GetDampingAttr()
        return None
    if joint.IsA(UsdPhysics.RevoluteJoint):
        driveAPI = UsdPhysics.DriveAPI(joint, "angular")
        return driveAPI.GetDampingAttr()
    if joint.IsA(UsdPhysics.PrismaticJoint):
        driveAPI = UsdPhysics.DriveAPI(joint, "linear")
        return driveAPI.GetDampingAttr()
    return None


def get_joint_drive_mode(joint: object) -> int:
    """Get the drive mode for a joint.

    Args:
        joint: The joint to get the mode from.

    Returns:
        The drive mode value.
    """
    if is_joint_mimic(joint):
        return JointDriveMode.MIMIC.value
    stiffness = get_stiffness_attr(joint)
    damping = get_damping_attr(joint)
    if stiffness:
        if stiffness.Get() > 0:
            return JointDriveMode.POSITION.value
        else:
            if damping:
                if damping.Get() > 0:
                    return JointDriveMode.VELOCITY.value
        return JointDriveMode.NONE.value
    return JointDriveMode.NONE.value
