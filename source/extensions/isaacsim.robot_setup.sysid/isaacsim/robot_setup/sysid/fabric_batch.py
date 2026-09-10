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

"""Batched articulation state reads for GPU-parallel candidate evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _warp_array_to_numpy(data) -> np.ndarray:  # noqa: ANN001
    import warp as wp

    if isinstance(data, wp.array):
        return np.array(data.numpy(), dtype=np.float32, copy=True)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(np.float32, copy=True)
    return np.asarray(data, dtype=np.float32)


def _to_torch(data: Any, device: torch.device, num_cols: int) -> torch.Tensor:
    """Copy state into an owning tensor without staging GPU data through NumPy.

    Args:
        data: Tensor-compatible articulation state.
        device: Device that owns the returned tensor.
        num_cols: Expected number of state columns.

    Returns:
        Owning, two-dimensional state tensor.
    """
    if isinstance(data, torch.Tensor):
        tensor = data.detach().to(device=device, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        return tensor.clone()
    try:
        import warp as wp

        if isinstance(data, wp.array):
            tensor = wp.to_torch(data).to(device=device, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.reshape(1, -1)
            # Articulation views reuse their backing buffers on later physics
            # steps.  A device-to-device clone preserves this sample without a
            # GPU -> host -> GPU synchronization.
            return tensor.clone()
    except (AttributeError, ImportError, RuntimeError, TypeError):
        pass
    arr = _warp_array_to_numpy(data)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return torch.as_tensor(arr, device=device, dtype=torch.float32)


def read_batched_joint_state(
    articulation: Any,
    *,
    num_envs: int,
    dof_indices: list[int],
    device: torch.device,
    include_effort: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Read positions, velocities, and measured efforts for all envs in one articulation view call.

    Args:
        articulation: Value supplied for ``articulation``.
        num_envs: Value supplied for ``num_envs``.
        dof_indices: Value supplied for ``dof_indices``.
        device: Device on which tensors are allocated.
        include_effort: Whether to read measured joint efforts.

    Returns:
        Result produced by the operation.
    """
    env_ids = list(range(num_envs))
    pos = articulation.get_dof_positions(indices=env_ids, dof_indices=dof_indices)
    vel = articulation.get_dof_velocities(indices=env_ids, dof_indices=dof_indices)
    pos_t = _to_torch(pos, device, len(dof_indices))
    vel_t = _to_torch(vel, device, len(dof_indices))
    effort_t = None
    if not include_effort:
        return pos_t, vel_t, effort_t
    try:
        efforts = articulation.get_dof_projected_joint_forces(indices=env_ids, dof_indices=dof_indices)
        effort_t = _to_torch(efforts, device, len(dof_indices))
    except (AttributeError, NotImplementedError, TypeError):
        try:
            efforts = articulation.get_dof_efforts(indices=env_ids, dof_indices=dof_indices)
            effort_t = _to_torch(efforts, device, len(dof_indices))
        except (AttributeError, NotImplementedError, TypeError):
            effort_t = None
    return pos_t, vel_t, effort_t


def read_batched_link_pose(
    articulation: Any,
    *,
    num_envs: int,
    link_index: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Read world poses in the public xyz+xyzw convention.

    Args:
        articulation: Articulation view that owns the requested link.
        num_envs: Number of batched environments.
        link_index: Link index in the articulation view.
        device: Torch device for the returned tensor.

    Returns:
        Pose tensor with shape ``(num_envs, 7)``, or ``None`` when the
        articulation cannot select the requested link.
    """
    if link_index < 0:
        return None
    try:
        positions, orientations = articulation.get_world_poses(
            indices=list(range(num_envs)),
            link_indices=[link_index],
        )
        pos = _to_torch(positions, device, 3)
        quat = _to_torch(orientations, device, 4)
        if pos.ndim == 3:
            pos = pos[:, 0, :]
            quat = quat[:, 0, :]
        quat_xyzw = quat[..., [1, 2, 3, 0]]
        return torch.cat([pos, quat_xyzw], dim=-1)
    except TypeError:
        # This articulation view cannot select a specific link (no link_indices
        # support); get_world_poses then returns per-environment root poses.
        # Indexing those by link_index would return an arbitrary environment's
        # root pose mislabeled as the link pose, so report it as unavailable
        # rather than feeding wrong data into the end-effector residual.
        return None
    except (AttributeError, NotImplementedError):
        return None


def read_batched_contact_proxy(
    articulation: Any,
    *,
    num_envs: int,
    link_index: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Reaction-force proxy for contact: a link's incoming joint force ``(num_envs, 3)``.

    This is not a measured external contact force. It returns the link's incoming
    joint constraint/reaction force, which is generally nonzero even without any
    external contact. It is only a coarse stand-in used for the CONTACT_FORCE
    residual when no contact sensor is available; callers and users should treat
    that residual channel as approximate.

    Args:
        articulation: Value supplied for ``articulation``.
        num_envs: Value supplied for ``num_envs``.
        link_index: Value supplied for ``link_index``.
        device: Device on which tensors are allocated.

    Returns:
        Result produced by the operation.
    """
    if link_index < 0:
        return None
    try:
        forces, _torques = articulation.get_link_incoming_joint_force(
            indices=list(range(num_envs)),
            link_indices=[link_index],
        )
        force = _to_torch(forces, device, 3)
        if force.ndim == 3:
            force = force[:, 0, :]
        return force
    except (AttributeError, NotImplementedError, TypeError):
        return None
