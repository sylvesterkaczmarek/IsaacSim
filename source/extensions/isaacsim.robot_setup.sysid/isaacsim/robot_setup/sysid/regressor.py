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

"""Body and joint regressor matrices for analytical Jacobians."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .inertia_param import LOG_CHOLESKY_DIM, log_cholesky_jacobian
from .parameter_types import GLOBAL_LINK_INDEX, SysIdParameterEntry, SysIdParameterType


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def _crm(v: np.ndarray) -> np.ndarray:
    w = v[:3]
    lin = v[3:]
    out = np.zeros((6, 6), dtype=np.float64)
    out[:3, :3] = _skew(w)
    out[3:, :3] = _skew(lin)
    out[3:, 3:] = _skew(w)
    return out


def _crf(v: np.ndarray) -> np.ndarray:
    return -_crm(v).T


def _pack_symmetric_inertia(basis: np.ndarray, inertia: np.ndarray) -> None:
    """Write a symmetric 3x3 block into params rows 4-9 (``[Ixx, Ixy, Ixz, Iyy, Iyz, Izz]``).

    Args:
        basis: Ten-row inertial-parameter basis to update.
        inertia: Symmetric inertia matrix to pack.
    """
    basis[4] = inertia[0, 0]
    basis[5] = 0.5 * (inertia[0, 1] + inertia[1, 0])
    basis[6] = 0.5 * (inertia[0, 2] + inertia[2, 0])
    basis[7] = inertia[1, 1]
    basis[8] = 0.5 * (inertia[1, 2] + inertia[2, 1])
    basis[9] = inertia[2, 2]


def _spatial_inertia_from_params(params: np.ndarray) -> np.ndarray:
    """Return spatial inertia from [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz].

    Args:
        params: Ten inertial parameters ``[m, h, I]``.

    Returns:
        Spatial inertia matrix with shape ``(6, 6)``.
    """
    p = np.asarray(params, dtype=np.float64).reshape(10)
    mass = p[0]
    first_moment = p[1:4]
    inertia = np.array(
        [[p[4], p[5], p[6]], [p[5], p[7], p[8]], [p[6], p[8], p[9]]],
        dtype=np.float64,
    )
    h_cross = _skew(first_moment)
    out = np.zeros((6, 6), dtype=np.float64)
    out[:3, :3] = inertia
    out[:3, 3:] = h_cross
    out[3:, :3] = h_cross.T
    out[3:, 3:] = np.eye(3, dtype=np.float64) * mass
    return out


@dataclass(frozen=True)
class UnmappedSubtreeLink:
    """One unmapped distal body rigidly composited into a mapped context link.

    ``rotation``/``translation`` place the unmapped link's body frame in the
    mapped context link's frame with every joint between them held at its rest
    configuration (exact for fixed attachments such as a gripper hand; an
    approximation for held movable joints such as gripper fingers).
    """

    context_link_index: int
    source_link_index: int
    rotation: np.ndarray
    translation: np.ndarray


@dataclass
class AnalyticalDynamicsContext:
    """Fixed-base serial dynamics context used to build an inverse-dynamics regressor."""

    parents: np.ndarray
    motion_subspaces: np.ndarray
    xup_fn: Callable[[int, float], np.ndarray] | None = None
    gravity: np.ndarray | None = None
    link_masses: np.ndarray | None = None
    #: Baseline body-frame CoM per context link, shape ``(num_links, 3)``. Needed by
    #: the analytic regressor: mass-scale and CoM-offset columns couple into the
    #: origin-referenced inertia rows through the baseline CoM. ``None`` means all
    #: baseline CoMs are zero.
    link_coms: np.ndarray | None = None
    source_link_indices: np.ndarray | None = None
    link_paths: tuple[str, ...] = ()
    joint_paths: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    #: Unmapped distal bodies (e.g. an attached gripper) reachable from a mapped
    #: link through held joints only; consumers that need total supported load
    #: (gravity feedforward comparison) composite these into their parent rows.
    unmapped_subtree_links: tuple[UnmappedSubtreeLink, ...] = ()

    @property
    def num_dof(self) -> int:  # noqa: D102
        return int(self.motion_subspaces.shape[0])

    @property
    def num_links(self) -> int:  # noqa: D102
        return int(self.parents.shape[0])

    def xup(self, link_index: int, q: float) -> np.ndarray:  # noqa: D102
        if self.xup_fn is None:
            del link_index, q
            return np.eye(6, dtype=np.float64)
        return np.asarray(self.xup_fn(link_index, q), dtype=np.float64).reshape(6, 6)

    def gravity_acceleration(self) -> np.ndarray:  # noqa: D102
        if self.gravity is None:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, -9.80665], dtype=np.float64)
        return np.asarray(self.gravity, dtype=np.float64).reshape(6)


def rnea_inverse_dynamics(
    context: AnalyticalDynamicsContext,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    inertial_params: np.ndarray,
) -> np.ndarray:
    """Recursive Newton-Euler inverse dynamics for fixed-base serial articulations.

    Args:
        context: Serial-articulation dynamics model.
        q: Joint positions for one dynamics sample.
        qd: Joint velocities for one dynamics sample.
        qdd: Joint accelerations for one dynamics sample.
        inertial_params: Per-link ten-parameter spatial inertia rows.

    Returns:
        Joint torque vector with shape ``(context.num_dof,)``.
    """
    n = context.num_dof
    parents = np.asarray(context.parents, dtype=np.int64).reshape(-1)
    if parents.shape != (n,):
        raise ValueError(f"RNEA context must provide {n} parent indices, got {parents.shape[0]}.")
    for link_index, parent_index in enumerate(parents):
        parent = int(parent_index)
        if parent < -1 or parent >= n:
            raise ValueError(f"RNEA parent index {parent} for link {link_index} is outside [-1, {n}).")
        if parent >= link_index:
            raise ValueError(
                "RNEA context is not topologically ordered: "
                f"link {link_index} references parent {parent}, which has not been evaluated."
            )

    v = [np.zeros(6, dtype=np.float64) for _ in range(n)]
    a = [np.zeros(6, dtype=np.float64) for _ in range(n)]
    f = [np.zeros(6, dtype=np.float64) for _ in range(n)]
    xup = [np.eye(6, dtype=np.float64) for _ in range(n)]
    gravity = context.gravity_acceleration()

    for i in range(n):
        parent = int(parents[i])
        s_i = context.motion_subspaces[i]
        xup[i] = context.xup(i, float(q[i]))
        vj = s_i * float(qd[i])
        if parent < 0:
            v[i] = vj
            a[i] = xup[i] @ (-gravity) + s_i * float(qdd[i]) + _crm(v[i]) @ vj
        else:
            v[i] = xup[i] @ v[parent] + vj
            a[i] = xup[i] @ a[parent] + s_i * float(qdd[i]) + _crm(v[i]) @ vj
        inertia = _spatial_inertia_from_params(inertial_params[i])
        f[i] = inertia @ a[i] + _crf(v[i]) @ (inertia @ v[i])

    tau = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        tau[i] = float(context.motion_subspaces[i] @ f[i])
        parent = int(parents[i])
        if parent >= 0:
            f[parent] += xup[i].T @ f[i]
    return tau


def assemble_inverse_dynamics_regressor(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    param_entries: Sequence[SysIdParameterEntry],
    context: AnalyticalDynamicsContext,
    *,
    baseline_link_inertia_lc: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Build exact inverse-dynamics columns by evaluating RNEA inertial bases.

    Params rows 4-9 are origin-referenced inertia (see
    ``analytical_presolve._origin_referenced_inertia``), so the mass-scale and
    CoM-offset bases include the parallel-axis coupling through the baseline CoM
    (``context.link_coms``). The columns are the exact ``d(tau)/d(theta)`` at the
    baseline theta (mass scale 1, zero CoM delta).

    Args:
        positions: Joint positions with shape ``(T, N)``.
        velocities: Joint velocities with shape ``(T, N)``.
        accelerations: Joint accelerations with shape ``(T, N)``.
        param_entries: Selected parameter entries in theta order.
        context: Serial-articulation dynamics model and baseline properties.
        baseline_link_inertia_lc: Per-link baseline Log-Cholesky inertia vectors.

    Returns:
        Inverse-dynamics regressor with shape ``(T * N, M)``.
    """
    times, num_dof = velocities.shape
    regressor = np.zeros((times * num_dof, len(param_entries)), dtype=np.float64)
    masses = context.link_masses
    if masses is None:
        masses = np.ones(context.num_links, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64).reshape(-1)
    coms = context.link_coms
    if coms is None or np.asarray(coms).size == 0:
        coms = np.zeros((context.num_links, 3), dtype=np.float64)
    coms = np.asarray(coms, dtype=np.float64).reshape(-1, 3)
    source_link_indices = _context_source_link_indices(context)
    source_to_context = {int(source_idx): ctx_idx for ctx_idx, source_idx in enumerate(source_link_indices)}

    def add_body_basis(link_idx: int, body_params: np.ndarray, col: int, scale: float = 1.0) -> None:
        inertial_basis = np.zeros((context.num_links, 10), dtype=np.float64)
        inertial_basis[link_idx] = body_params * scale
        for t in range(times):
            tau = rnea_inverse_dynamics(context, positions[t], velocities[t], accelerations[t], inertial_basis)
            start = t * num_dof
            regressor[start : start + num_dof, col] += tau[:num_dof]

    for p_idx, entry in enumerate(param_entries):
        if entry.param_type == SysIdParameterType.LINK_MASS:
            if entry.link_index == GLOBAL_LINK_INDEX:
                links = range(context.num_links)
            else:
                ctx_idx = source_to_context.get(int(entry.link_index))
                links = () if ctx_idx is None else (ctx_idx,)
            for link_idx in links:
                if 0 <= link_idx < context.num_links:
                    # Mass scale s multiplies m = m0*s, h = m*c, and the parallel-axis
                    # part of the origin-referenced inertia (rows 4-9), so with a
                    # nonzero baseline CoM c the basis carries all three couplings.
                    com = coms[min(link_idx, coms.shape[0] - 1)]
                    basis = np.zeros(10, dtype=np.float64)
                    basis[0] = 1.0
                    basis[1:4] = com
                    _pack_symmetric_inertia(basis, float(com @ com) * np.eye(3, dtype=np.float64) - np.outer(com, com))
                    add_body_basis(link_idx, basis, p_idx, scale=masses[min(link_idx, len(masses) - 1)])
            continue

        if entry.param_type in (
            SysIdParameterType.LINK_COM_OFFSET_X,
            SysIdParameterType.LINK_COM_OFFSET_Y,
            SysIdParameterType.LINK_COM_OFFSET_Z,
        ):
            axis = {
                SysIdParameterType.LINK_COM_OFFSET_X: 1,
                SysIdParameterType.LINK_COM_OFFSET_Y: 2,
                SysIdParameterType.LINK_COM_OFFSET_Z: 3,
            }[entry.param_type]
            ctx_idx = source_to_context.get(int(entry.link_index))
            if ctx_idx is not None and 0 <= ctx_idx < context.num_links:
                # A CoM offset along e_k moves h = m*c and shifts the origin-referenced
                # inertia: d/dc_k [m(|c|^2 1 - c c^T)] = m(2 c_k 1 - e_k c^T - c e_k^T).
                mass = masses[min(ctx_idx, len(masses) - 1)]
                com = coms[min(ctx_idx, coms.shape[0] - 1)]
                unit = np.zeros(3, dtype=np.float64)
                unit[axis - 1] = 1.0
                basis = np.zeros(10, dtype=np.float64)
                basis[axis] = mass
                _pack_symmetric_inertia(
                    basis,
                    mass
                    * (2.0 * com[axis - 1] * np.eye(3, dtype=np.float64) - np.outer(unit, com) - np.outer(com, unit)),
                )
                add_body_basis(ctx_idx, basis, p_idx)
            continue

        if entry.param_type == SysIdParameterType.LINK_INERTIA_LOG_CHOLESKY:
            ctx_idx = source_to_context.get(int(entry.link_index))
            if ctx_idx is not None and 0 <= ctx_idx < context.num_links and baseline_link_inertia_lc is not None:
                baseline = baseline_link_inertia_lc.get(entry.link_index)
                if baseline is None:
                    continue
                local = log_cholesky_jacobian(baseline)[:, entry.component_index].reshape(3, 3)
                basis = np.zeros(10, dtype=np.float64)
                basis[4] = local[0, 0]
                basis[5] = 0.5 * (local[0, 1] + local[1, 0])
                basis[6] = 0.5 * (local[0, 2] + local[2, 0])
                basis[7] = local[1, 1]
                basis[8] = 0.5 * (local[1, 2] + local[2, 1])
                basis[9] = local[2, 2]
                add_body_basis(ctx_idx, basis, p_idx)

    return regressor


def _context_source_link_indices(context: AnalyticalDynamicsContext) -> np.ndarray:
    if context.source_link_indices is None:
        return np.arange(context.num_links, dtype=np.int64)
    values = np.asarray(context.source_link_indices, dtype=np.int64).reshape(-1)
    if values.shape[0] != context.num_links:
        return np.arange(context.num_links, dtype=np.int64)
    return values


def finite_difference_acceleration(times: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Estimate joint accelerations from velocity samples via central differences.

    Args:
        times: Telemetry sample timestamps.
        velocities: Joint velocities with shape ``(T, N)``.

    Returns:
        Estimated accelerations with the same shape as ``velocities``.
    """
    if velocities.shape[0] < 2:
        return np.zeros_like(velocities)
    sample_times = np.asarray(times, dtype=np.float64).reshape(-1)
    if sample_times.shape[0] != velocities.shape[0]:
        raise ValueError(
            f"Timestamp count ({sample_times.shape[0]}) must match velocity samples ({velocities.shape[0]})."
        )
    safe_dt = np.maximum(np.diff(sample_times), 1e-8)
    safe_times = np.concatenate(([sample_times[0]], sample_times[0] + np.cumsum(safe_dt)))
    edge_order = 2 if velocities.shape[0] >= 3 else 1
    return np.gradient(velocities, safe_times, axis=0, edge_order=edge_order)


def assemble_body_regressor(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    num_links: int,
) -> np.ndarray:
    """Assemble a stacked body regressor for per-link mass and diagonal inertia.

    Legacy fallback used by standalone unit tests. Optimizer analytical mode uses
    :class:`AnalyticalDynamicsContext` and ``assemble_inverse_dynamics_regressor``.

    Args:
        positions: Shape ``(T, N)``.
        velocities: Shape ``(T, N)``.
        accelerations: Shape ``(T, N)``.
        num_links: Number of articulated links.

    Returns:
        Regressor matrix with shape ``(T * N, 4 * num_links)``.
    """
    times = positions.shape[0]
    num_dof = positions.shape[1]
    num_params = 4 * num_links
    regressor = np.zeros((times * num_dof, num_params), dtype=np.float64)

    for t in range(times):
        for j in range(num_dof):
            row = t * num_dof + j
            link_idx = min(j, num_links - 1)
            qdd = accelerations[t, j]
            qd = velocities[t, j]
            base = link_idx * 4
            regressor[row, base + 0] = abs(qdd) + 1e-6
            regressor[row, base + 1] = abs(qdd) * 0.5
            regressor[row, base + 2] = abs(qdd) * 0.5
            regressor[row, base + 3] = abs(qd) * abs(qdd) + 1e-6

    return regressor


def assemble_joint_regressor(  # noqa: D417
    velocities: np.ndarray,
    param_entries: Sequence[SysIdParameterEntry],
    positions: np.ndarray | None = None,
    commands: np.ndarray | None = None,
    *,
    baseline_joint_friction: Mapping[int, float] | None = None,
    baseline_joint_stiffness: Mapping[int, float] | None = None,
    baseline_joint_damping: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Assemble joint regressor columns for friction, damping, and stiffness parameters.

    Args:
        velocities: Shape ``(T, N)``.
        param_entries: Optimized parameter list in theta order.
        positions: Optional measured joint positions.
        commands: Optional joint-position targets used as the stiffness reference.
        baseline_joint_friction: Authored friction multiplied by each friction scale.
        baseline_joint_stiffness: Authored stiffness multiplied by each stiffness scale.
        baseline_joint_damping: Authored damping multiplied by each damping scale.

    Returns:
        Regressor with shape ``(T * N, M)`` aligned with ``param_entries``.
    """  # noqa: DOC101, DOC103
    times, num_dof = velocities.shape
    num_params = len(param_entries)
    regressor = np.zeros((times * num_dof, num_params), dtype=np.float64)

    for p_idx, entry in enumerate(param_entries):
        if entry.dof_index < 0 or entry.dof_index >= num_dof:
            continue
        j = entry.dof_index
        if entry.param_type == SysIdParameterType.JOINT_FRICTION:
            baseline = float((baseline_joint_friction or {}).get(j, 1.0))
            for t in range(times):
                row = t * num_dof + j
                regressor[row, p_idx] = -baseline * np.sign(velocities[t, j])
        elif entry.param_type == SysIdParameterType.JOINT_DAMPING:
            baseline = float((baseline_joint_damping or {}).get(j, 1.0))
            for t in range(times):
                row = t * num_dof + j
                regressor[row, p_idx] = -baseline * velocities[t, j]
        elif entry.param_type == SysIdParameterType.JOINT_STIFFNESS and positions is not None:
            baseline = float((baseline_joint_stiffness or {}).get(j, 1.0))
            reference = commands if commands is not None else np.zeros_like(positions)
            for t in range(times):
                row = t * num_dof + j
                regressor[row, p_idx] = -baseline * (positions[t, j] - reference[t, j])

    return regressor


def inertia_log_cholesky_parameter_jacobian(
    param_entries: Iterable[SysIdParameterEntry],
    baseline_link_inertia_lc: dict[int, np.ndarray],
) -> np.ndarray:
    """Build block-diagonal ``d(flat(I)) / d(theta)`` for Log-Cholesky inertia entries.

    Returns:
        Matrix with shape ``(9 * num_active_links, M)`` where ``M = len(param_entries)``.

    Args:
        param_entries: Selected parameter entries in theta order.
        baseline_link_inertia_lc: Per-link baseline Log-Cholesky inertia vectors.
    """
    entries = list(param_entries)
    num_params = len(entries)
    active_links = sorted(
        {e.link_index for e in entries if e.param_type == SysIdParameterType.LINK_INERTIA_LOG_CHOLESKY}
    )
    if not active_links:
        return np.zeros((0, num_params), dtype=np.float64)

    jacobian = np.zeros((9 * len(active_links), num_params), dtype=np.float64)
    link_to_block = {link_idx: block for block, link_idx in enumerate(active_links)}

    for p_idx, entry in enumerate(entries):
        if entry.param_type != SysIdParameterType.LINK_INERTIA_LOG_CHOLESKY:
            continue
        link_idx = entry.link_index
        if link_idx not in link_to_block:
            continue
        baseline = baseline_link_inertia_lc.get(link_idx, np.zeros(LOG_CHOLESKY_DIM, dtype=np.float64))
        local_jac = log_cholesky_jacobian(baseline)
        block = link_to_block[link_idx]
        row_start = block * 9
        jacobian[row_start : row_start + 9, p_idx] = local_jac[:, entry.component_index]

    return jacobian


def estimate_analytical_jacobian(  # noqa: D417
    positions: np.ndarray,
    velocities: np.ndarray,
    times: np.ndarray,
    param_entries: list[SysIdParameterEntry],
    *,
    num_links: int,
    baseline_link_inertia_lc: dict[int, np.ndarray] | None = None,
    residual_dim: int,
    dynamics_context: AnalyticalDynamicsContext | None = None,
    commands: np.ndarray | None = None,
    baseline_joint_friction: Mapping[int, float] | None = None,
    baseline_joint_stiffness: Mapping[int, float] | None = None,
    baseline_joint_damping: Mapping[int, float] | None = None,
) -> torch.Tensor | None:
    """Estimate a parameter Jacobian from regressor matrices for Newton/LM backends.

    The assembled matrix is in the torque (inverse-dynamics) domain: each row is
    ``d(joint torque) / d(theta)``. It is only a valid Levenberg-Marquardt Jacobian
    when the optimizer residual is itself torque-domain; callers must not use it
    against position/velocity trajectory residuals.

    Args:
        commands: Per-step joint command/target signal ``(T, N)`` used as the
            stiffness reference ``d(tau)/d(stiffness) = command - position``. When
            ``None`` the reference defaults to zero, which zeros the stiffness
            columns -- pass the measured command signal to identify stiffness.
        baseline_joint_friction: Authored friction values keyed by DOF index.
        baseline_joint_stiffness: Authored stiffness values keyed by DOF index.
        baseline_joint_damping: Authored damping values keyed by DOF index.

    Returns ``None`` when no regressor columns are available. Otherwise returns a
    ``(residual_dim, M)`` tensor. A row mismatch is rejected instead of silently
    truncating or padding a Jacobian that is not aligned with the residual vector.

    Returns:
        Torque-domain Jacobian with shape ``(residual_dim, M)``, or ``None`` when no columns are available.
    """  # noqa: DOC101, DOC103
    if not param_entries:
        return None

    accelerations = finite_difference_acceleration(times, velocities)
    if dynamics_context is not None:
        body = assemble_inverse_dynamics_regressor(
            positions,
            velocities,
            accelerations,
            param_entries,
            dynamics_context,
            baseline_link_inertia_lc=baseline_link_inertia_lc,
        )
    else:
        # The legacy body regressor is not aligned to ``param_entries`` and must
        # not be merged into a production Jacobian. Without a dynamics context,
        # retain only the explicitly aligned joint columns and let LM fall back
        # to finite differences when no such columns are available.
        _ = num_links
        body = np.zeros((velocities.shape[0] * velocities.shape[1], len(param_entries)), dtype=np.float64)
    joint = assemble_joint_regressor(
        velocities,
        param_entries,
        positions=positions,
        commands=commands,
        baseline_joint_friction=baseline_joint_friction,
        baseline_joint_stiffness=baseline_joint_stiffness,
        baseline_joint_damping=baseline_joint_damping,
    )

    # Combine simplified body/joint regressors into one matrix aligned with theta.
    combined_rows = min(body.shape[0], joint.shape[0])
    if combined_rows == 0:
        return None

    combined = np.zeros((combined_rows, len(param_entries)), dtype=np.float64)
    if body.shape[1] > 0:
        cols = min(body.shape[1], len(param_entries))
        combined[:, :cols] += body[:combined_rows, :cols]
    combined += joint[:combined_rows, :]

    if not np.any(combined):
        return None

    if combined_rows != residual_dim:
        raise ValueError(f"Analytical Jacobian rows ({combined_rows}) must match residual dimension ({residual_dim}).")
    return torch.tensor(combined, dtype=torch.float32)
