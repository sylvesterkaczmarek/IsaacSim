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

"""Apply optimizer theta rows to simulation parameter mutations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .inertia_param import LOG_CHOLESKY_DIM, log_cholesky_to_inertia_flat
from .parameter_types import (
    GLOBAL_DOF_INDEX,
    GLOBAL_LINK_INDEX,
    SysIdParameterEntry,
    SysIdParameterType,
)


@dataclass
class ParameterApplyState:
    """Decoded simulation mutations for one environment clone.

    Optional per-DOF vectors use two levels of absence:

    - ``None`` means the candidate contains no parameter of that type.
    - ``NaN`` at one DOF means that DOF was not selected and its authored or
      runtime baseline must be preserved by the consumer.

    Consumers must apply optional vectors with an ``isfinite`` mask; they must
    never forward their unset ``NaN`` elements to a simulation backend.
    """

    friction_scale: np.ndarray
    stiffness_scale: np.ndarray
    damping_scale: np.ndarray
    joint_integral_gain: np.ndarray | None
    joint_armature: np.ndarray | None
    actuator_command_delay_seconds: np.ndarray | None
    link_mass_scale: np.ndarray
    link_com_delta: np.ndarray
    link_inertia_flat: dict[int, np.ndarray] = field(default_factory=dict)
    joint_limit_lower_scale: np.ndarray | None = None
    joint_limit_upper_scale: np.ndarray | None = None


def per_dof_scale_vectors(
    num_dof: int,
    param_entries: list[SysIdParameterEntry],
    theta_row: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return joint drive scales and a scalar link-mass scale.

    Args:
        num_dof: Number of articulation degrees of freedom.
        param_entries: Metadata describing each theta column.
        theta_row: One candidate parameter vector.

    Returns:
        Multiplicative friction, stiffness, damping, and mass scales.
    """
    state = decode_theta_to_apply_state(
        num_dof=num_dof,
        num_links=1,
        param_entries=param_entries,
        theta_row=theta_row,
    )
    link_mass = float(state.link_mass_scale[0]) if state.link_mass_scale.size else 1.0
    return state.friction_scale, state.stiffness_scale, state.damping_scale, link_mass


def decode_theta_to_apply_state(
    num_dof: int,
    num_links: int,
    param_entries: list[SysIdParameterEntry],
    theta_row: torch.Tensor | np.ndarray,
    *,
    baseline_link_inertia_lc: dict[int, np.ndarray] | None = None,
) -> ParameterApplyState:
    """Decode one optimizer theta row into per-clone simulation mutations.

    Args:
        num_dof: Number of articulation degrees of freedom.
        num_links: Number of optimized rigid links.
        param_entries: Metadata describing each theta column.
        theta_row: One candidate parameter vector.
        baseline_link_inertia_lc: Optional per-link Log-Cholesky baselines.

    Returns:
        Fully decoded candidate values ready for a simulation bridge. Optional
        per-DOF vectors are ``None`` when absent; within a present vector,
        ``NaN`` marks an unselected DOF whose baseline must be preserved.
    """
    friction = np.ones(num_dof, dtype=np.float32)
    stiffness = np.ones(num_dof, dtype=np.float32)
    damping = np.ones(num_dof, dtype=np.float32)
    integral_gain = np.full(num_dof, np.nan, dtype=np.float32)
    joint_armature = np.full(num_dof, np.nan, dtype=np.float32)
    command_delay = np.full(num_dof, np.nan, dtype=np.float32)
    link_mass_scale = np.ones(max(num_links, 1), dtype=np.float32)
    link_com_delta = np.zeros((max(num_links, 1), 3), dtype=np.float32)
    lower_scale = np.ones(num_dof, dtype=np.float32)
    upper_scale = np.ones(num_dof, dtype=np.float32)

    inertia_lc: dict[int, np.ndarray] = {}
    if baseline_link_inertia_lc:
        inertia_lc = {idx: vec.copy() for idx, vec in baseline_link_inertia_lc.items()}

    has_limits = False
    has_integral_gain = False
    has_armature = False
    has_command_delay = False
    inertia_links: set[int] = set()

    # Callers that evaluate candidate batches can transfer the full batch to the
    # host once and pass NumPy rows here.  Retain the tensor API for standalone
    # callers, but perform one transfer per row instead of one synchronizing
    # ``Tensor.item()`` per parameter.
    if isinstance(theta_row, torch.Tensor):
        theta_values = theta_row.detach().to(device="cpu", dtype=torch.float32).reshape(-1).numpy()
    else:
        theta_values = np.asarray(theta_row, dtype=np.float32).reshape(-1)
    if theta_values.shape[0] != len(param_entries):
        raise ValueError(
            f"theta row has {theta_values.shape[0]} values but {len(param_entries)} parameter entries were given."
        )

    for p_idx, entry in enumerate(param_entries):
        value = float(theta_values[p_idx])
        ptype = entry.param_type

        if ptype == SysIdParameterType.LINK_MASS:
            if entry.link_index == GLOBAL_LINK_INDEX:
                link_mass_scale[:] = max(value, 0.0)
            elif 0 <= entry.link_index < num_links:
                link_mass_scale[entry.link_index] = max(value, 0.0)
            else:
                raise ValueError(
                    f"{ptype.value} link_index {entry.link_index} is outside [0, {num_links}) "
                    f"and is not GLOBAL_LINK_INDEX."
                )
            continue

        if ptype == SysIdParameterType.ACTUATOR_COMMAND_DELAY_SECONDS:
            has_command_delay = True
            delay = max(value, 0.0)
            if entry.dof_index == GLOBAL_DOF_INDEX:
                command_delay[~np.isfinite(command_delay)] = delay
            elif 0 <= entry.dof_index < num_dof:
                command_delay[entry.dof_index] = delay
            else:
                raise ValueError(
                    f"{ptype.value} dof_index {entry.dof_index} is outside [0, {num_dof}) "
                    f"and is not GLOBAL_DOF_INDEX."
                )
            continue

        if ptype in (
            SysIdParameterType.LINK_COM_OFFSET_X,
            SysIdParameterType.LINK_COM_OFFSET_Y,
            SysIdParameterType.LINK_COM_OFFSET_Z,
        ):
            axis = {
                SysIdParameterType.LINK_COM_OFFSET_X: 0,
                SysIdParameterType.LINK_COM_OFFSET_Y: 1,
                SysIdParameterType.LINK_COM_OFFSET_Z: 2,
            }[ptype]
            if entry.link_index == GLOBAL_LINK_INDEX:
                link_com_delta[:, axis] += value
            elif 0 <= entry.link_index < num_links:
                link_com_delta[entry.link_index, axis] += value
            else:
                raise ValueError(
                    f"{ptype.value} link_index {entry.link_index} is outside [0, {num_links}) "
                    f"and is not GLOBAL_LINK_INDEX."
                )
            continue

        if ptype == SysIdParameterType.LINK_INERTIA_LOG_CHOLESKY:
            if not 0 <= entry.component_index < LOG_CHOLESKY_DIM:
                raise ValueError(
                    f"{ptype.value} component_index {entry.component_index} is outside " f"[0, {LOG_CHOLESKY_DIM})."
                )
            if entry.link_index == GLOBAL_LINK_INDEX:
                link_indices = range(num_links)
            elif 0 <= entry.link_index < num_links:
                link_indices = (entry.link_index,)
            else:
                raise ValueError(
                    f"{ptype.value} link_index {entry.link_index} is outside [0, {num_links}) "
                    f"and is not GLOBAL_LINK_INDEX."
                )
            for link_idx in link_indices:
                if link_idx not in inertia_lc:
                    inertia_lc[link_idx] = np.zeros(LOG_CHOLESKY_DIM, dtype=np.float64)
                inertia_lc[link_idx][entry.component_index] = value
                inertia_links.add(link_idx)
            continue

        if entry.dof_index < 0 or entry.dof_index >= num_dof:
            raise ValueError(f"{ptype.value} dof_index {entry.dof_index} is outside [0, {num_dof}).")

        j = entry.dof_index
        if ptype == SysIdParameterType.JOINT_FRICTION:
            friction[j] *= max(value, 0.0)
        elif ptype == SysIdParameterType.JOINT_STIFFNESS:
            stiffness[j] *= max(value, 0.0)
        elif ptype == SysIdParameterType.JOINT_DAMPING:
            damping[j] *= max(value, 0.0)
        elif ptype == SysIdParameterType.JOINT_INTEGRAL_GAIN:
            has_integral_gain = True
            integral_gain[j] = max(value, 0.0)
        elif ptype == SysIdParameterType.JOINT_ARMATURE:
            has_armature = True
            joint_armature[j] = max(value, 0.0)
        elif ptype == SysIdParameterType.JOINT_LIMIT_LOWER_SCALE:
            has_limits = True
            lower_scale[j] *= max(value, 0.0)
        elif ptype == SysIdParameterType.JOINT_LIMIT_UPPER_SCALE:
            has_limits = True
            upper_scale[j] *= max(value, 0.0)
        else:
            raise ValueError(f"Unsupported parameter type: {ptype.value}.")

    # Only emit inertia for links that an optimized parameter actually touched.
    # The baseline seeds non-optimized components of a touched link, but links with
    # no inertia parameter must not be rewritten (doing so would overwrite the
    # authored inertia/principal axes via a needless decode round-trip).
    link_inertia_flat: dict[int, np.ndarray] = {}
    for link_idx in inertia_links:
        if 0 <= link_idx < num_links:
            link_inertia_flat[link_idx] = log_cholesky_to_inertia_flat(inertia_lc[link_idx]).astype(np.float32)

    return ParameterApplyState(
        friction_scale=friction,
        stiffness_scale=stiffness,
        damping_scale=damping,
        joint_integral_gain=integral_gain if has_integral_gain else None,
        joint_armature=joint_armature if has_armature else None,
        actuator_command_delay_seconds=command_delay if has_command_delay else None,
        link_mass_scale=link_mass_scale,
        link_com_delta=link_com_delta,
        link_inertia_flat=link_inertia_flat,
        joint_limit_lower_scale=lower_scale if has_limits else None,
        joint_limit_upper_scale=upper_scale if has_limits else None,
    )
