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

"""Pure math for position-drive natural frequency and damping ratio.

``UsdPhysics`` stores a drive's stiffness and damping in the drive's own angular
or linear convention, while the spring-mass relations ``omega_n = sqrt(K/m)`` and
``zeta = D / (2 sqrt(m K))`` only hold in SI (radians or metres). Every function
here therefore converts through :func:`stored_gain_scale`, which reports the one
scale that separates the two.

Both angular gains are stored *per degree* --- ``UsdPhysics`` declares stiffness
as ``mass*DIST*DIST/degrees/second/second`` and damping as
``mass*DIST*DIST/second/degrees`` --- so the same scale applies to each. Applying
it to only one of them leaves the pair inconsistent by that scale, which is the
kind of error the ratio ``D/K == 2 zeta / omega_n`` exposes regardless of inertia.
Linear drives carry no angle unit and take no scale at all.
"""

from __future__ import annotations

import math

DEG_TO_RAD = math.pi / 180.0

# Linear drive gains are already SI (per stage linear unit), so they need no scaling.
_LINEAR_GAIN_SCALE = 1.0


def stored_gain_scale(*, is_angular: bool) -> float:
    """Scale relating a stored position-drive gain to its SI value.

    A stored gain divided by this scale is the SI gain; an SI gain multiplied by
    it is the stored gain. Applies identically to stiffness and damping.

    Args:
        is_angular: True for revolute / D6-rotational DOFs whose gains are stored
            per degree, False for prismatic / linear DOFs.

    Returns:
        ``pi/180`` for angular DOFs, ``1.0`` for linear DOFs.
    """
    return DEG_TO_RAD if is_angular else _LINEAR_GAIN_SCALE


def meq_for_drive_frequency(*, use_force_drive: bool, m_eq: float) -> float:
    """Equivalent inertia (or mass) scalar used in natural-frequency formulas.

    Acceleration drive uses ``1.0``; force drive uses ``m_eq`` with a fallback
    when zero.

    Args:
        use_force_drive: True when the joint drive uses force mode.
        m_eq: Equivalent mass or inertia for the joint.

    Returns:
        Equivalent scalar used in the drive frequency calculation.
    """
    if not use_force_drive:
        return 1.0
    return 1.0 if m_eq == 0 else m_eq


def natural_frequency_hz_from_stiffness_position_drive(
    stiffness_stored: float, *, is_angular: bool, use_force_drive: bool, m_eq: float
) -> float:
    """Natural frequency (Hz) from a stored position-drive stiffness (non-mimic).

    Args:
        stiffness_stored: Stiffness as stored on the joint / in the UI model.
        is_angular: True for revolute DOFs (gains stored per degree), False for prismatic.
        use_force_drive: True if drive type is force (uses ``m_eq``).
        m_eq: Equivalent inertia or mass from the gain tuner pipeline.

    Returns:
        Natural frequency in Hz.
    """
    m = meq_for_drive_frequency(use_force_drive=use_force_drive, m_eq=m_eq)
    stiffness_si = stiffness_stored / stored_gain_scale(is_angular=is_angular)
    return math.sqrt(stiffness_si / m) / (2.0 * math.pi)


def damping_ratio_from_stiffness_damping_position_drive(
    damping_stored: float, stiffness_stored: float, *, is_angular: bool, use_force_drive: bool, m_eq: float
) -> float:
    """Damping ratio from a stored stiffness / damping pair (non-mimic).

    Both gains are converted out of the stored convention before applying
    ``zeta = D / (2 sqrt(m K))``, so the ratio is dimensionless as intended.

    Args:
        damping_stored: Damping as stored on the joint / in the UI model.
        stiffness_stored: Stiffness as stored on the joint / in the UI model.
        is_angular: True for revolute DOFs (gains stored per degree), False for prismatic.
        use_force_drive: True when the joint drive uses force mode.
        m_eq: Equivalent inertia or mass from the gain tuner pipeline.

    Returns:
        Damping ratio for the drive, or ``0.0`` when there is no stiffness.
    """
    if stiffness_stored <= 0:
        return 0.0
    m = meq_for_drive_frequency(use_force_drive=use_force_drive, m_eq=m_eq)
    scale = stored_gain_scale(is_angular=is_angular)
    stiffness_si = stiffness_stored / scale
    damping_si = damping_stored / scale
    return damping_si / (2.0 * math.sqrt(m * stiffness_si))


def stiffness_and_damping_from_natural_frequency_position_drive(
    natural_freq_hz: float, damping_ratio: float, *, is_angular: bool, use_force_drive: bool, m_eq: float
) -> tuple[float, float]:
    """Stored stiffness and damping for a target ``f_n`` and ``zeta``.

    Args:
        natural_freq_hz: Target natural frequency in Hz.
        damping_ratio: Target damping ratio.
        is_angular: True for revolute DOFs (gains stored per degree), False for prismatic.
        use_force_drive: True when the joint drive uses force mode.
        m_eq: Equivalent inertia or mass from the gain tuner pipeline.

    Returns:
        ``(stiffness_stored, damping_stored)`` in the drive's stored convention.
    """
    m = meq_for_drive_frequency(use_force_drive=use_force_drive, m_eq=m_eq)
    scale = stored_gain_scale(is_angular=is_angular)
    stiffness_si = m * ((2.0 * math.pi * natural_freq_hz) ** 2)
    damping_si = damping_ratio * (2.0 * math.sqrt(m * stiffness_si))
    return stiffness_si * scale, damping_si * scale


def damping_from_damping_ratio_position_drive(
    damping_ratio: float, stiffness_stored: float, *, is_angular: bool, use_force_drive: bool, m_eq: float
) -> float:
    """Stored damping for a target damping ratio at the current stored stiffness.

    Args:
        damping_ratio: Target damping ratio.
        stiffness_stored: Stiffness as stored on the joint / in the UI model.
        is_angular: True for revolute DOFs (gains stored per degree), False for prismatic.
        use_force_drive: True when the joint drive uses force mode.
        m_eq: Equivalent inertia or mass from the gain tuner pipeline.

    Returns:
        Damping in the drive's stored convention.
    """
    m = meq_for_drive_frequency(use_force_drive=use_force_drive, m_eq=m_eq)
    scale = stored_gain_scale(is_angular=is_angular)
    stiffness_si = stiffness_stored / scale
    damping_si = damping_ratio * (2.0 * math.sqrt(m * stiffness_si))
    return damping_si * scale
