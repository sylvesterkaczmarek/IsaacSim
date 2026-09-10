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

# NOTE: Temporary vendor copy — this implementation is in the process of moving to
# ``newton.controllers`` in upcoming releases of ``isaacsim.pip.newton``.

"""ControllerHolonomicDrive — holonomic (omni/mecanum) wheel controller."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .controller import Controller
from .utils import _allocate_namespace


@wp.kernel
def _clamp_twist_kernel(
    twist_in: wp.array[wp.float32],
    max_linear_speed: wp.array[wp.float32],
    max_angular_speed: wp.array[wp.float32],
    twist_out: wp.array[wp.float32],
) -> None:
    """Clamp a body twist [vx, vy, wz] to speed limits (single thread).

    The planar linear speed ``sqrt(vx² + vy²)`` is uniformly scaled to stay
    within ``max_linear_speed``.  ``wz`` is independently clamped to
    ``±max_angular_speed``.

    Args:
        twist_in: Input twist command.
        max_linear_speed: Maximum allowed linear speed.
        max_angular_speed: Maximum allowed angular speed.
        twist_out: Clamped output twist.
    """
    vx = twist_in[0]
    vy = twist_in[1]
    wz = twist_in[2]

    lin_max = max_linear_speed[0]
    linear_sq = vx * vx + vy * vy
    if linear_sq > lin_max * lin_max:
        scale = lin_max / wp.sqrt(linear_sq)
        vx = vx * scale
        vy = vy * scale

    ang_max = max_angular_speed[0]
    wz = wp.clamp(wz, -ang_max, ang_max)

    twist_out[0] = vx
    twist_out[1] = vy
    twist_out[2] = wz


@wp.kernel
def _holonomic_drive_kernel(
    clamped_twist: wp.array[wp.float32],
    kinematics_m: wp.array[wp.float32],
    kinematics_k: wp.array[wp.float32],
    max_wheel_speed: wp.array[wp.float32],
    output_indices: wp.array[wp.uint32],
    output: wp.array[wp.float32],
) -> None:
    """Compute phi_dot for one wheel and write to output (one thread per wheel).

    ``kinematics_m`` is stored row-major with stride 3: row ``i`` occupies
    indices ``[3*i, 3*i+1, 3*i+2]``.

    ``phi_dot_i = K_i * (M[i,0]*vx + M[i,1]*vy + M[i,2]*wz)``

    Each wheel's speed is independently clamped to ``±max_wheel_speed[0]``
    after the multiply.

    Args:
        clamped_twist: Clamped twist values.
        kinematics_m: Linear kinematics matrix.
        kinematics_k: Angular kinematics matrix.
        max_wheel_speed: Maximum allowed wheel speed.
        output_indices: Indices selecting output values.
        output: Output buffer to populate.
    """
    i = wp.tid()
    vx = clamped_twist[0]
    vy = clamped_twist[1]
    wz = clamped_twist[2]

    contact_speed = kinematics_m[i * 3 + 0] * vx + kinematics_m[i * 3 + 1] * vy + kinematics_m[i * 3 + 2] * wz
    phi_dot = kinematics_k[i] * contact_speed
    phi_dot = wp.clamp(phi_dot, -max_wheel_speed[0], max_wheel_speed[0])

    output[output_indices[i]] = phi_dot


class ControllerHolonomicDrive(Controller):
    """Holonomic-drive controller for N-wheel omni / mecanum robots.

    Converts a body twist command ``[vx, vy, wz]`` into per-wheel angular
    velocities using precomputed inverse-kinematic matrices::

        phi_dot_i = K_i * (M[i, :] @ [vx, vy, wz])

    where ``M`` (N×3) maps the body twist to per-wheel contact speeds and
    ``K`` (N,) converts contact speeds to wheel joint angular velocities.
    Both matrices are baked in at construction from fixed wheel geometry.

    Speed limiting is applied in two stages:

    1. **Pre-multiply**: the planar linear speed is uniformly scaled down to
       ``max_linear_speed`` (preserving vx/vy ratio); ``wz`` is clamped to
       ``±max_angular_speed``.
    2. **Post-multiply**: each wheel speed is independently clamped to
       ``±max_wheel_speed``.  Unlike the pre-multiply stage, this clamp can
       distort the commanded motion direction; set it large to disable.

    Args:
        num_wheels: Number of wheel joints controlled.
        kinematics_m: Row-major kinematic matrix ``M``. Maps ``[vx, vy, wz]``
            to per-wheel no-slip-axis contact speeds.
        kinematics_k: Conversion diagonal ``K``. Converts each contact speed
            to a wheel joint angular velocity.
        default_dof_indices: Output slot index for each wheel DOF, length
            ``num_wheels``. The output array is sized
            ``max(default_dof_indices) + 1``; slots outside these indices are
            left untouched.
        max_linear_speed: Planar linear speed limit [m/s]. ``None`` means no
            limit.
        max_angular_speed: Yaw-rate limit [rad/s]. ``None`` means no limit.
        max_wheel_speed: Per-wheel angular velocity clamp [rad/s]. ``None``
            means no limit.
        twist_command_attr: Input port name for the body twist command.
        joint_target_qd_attr: Output port name for wheel angular velocities.
        device: Warp device for internal buffers.
        requires_grad: Allocate buffers with gradient support.

    Raises:
        ValueError: If ``num_wheels < 1`` or ``default_dof_indices`` has the
            wrong length.
        TypeError: If ``default_dof_indices`` is not a ``wp.array[uint32]``.
    """

    def __init__(
        self,
        *,
        num_wheels: int,
        kinematics_m: np.ndarray | wp.array,
        kinematics_k: np.ndarray | wp.array,
        default_dof_indices: wp.array,
        max_linear_speed: wp.array | None = None,
        max_angular_speed: wp.array | None = None,
        max_wheel_speed: wp.array | None = None,
        twist_command_attr: str = "twist_command",
        joint_target_qd_attr: str = "joint_target_qd",
        device: Any = None,
        requires_grad: bool = False,
    ) -> None:
        if num_wheels < 1:
            raise ValueError(f"num_wheels must be >= 1, got {num_wheels}.")
        if not isinstance(default_dof_indices, wp.array) or default_dof_indices.dtype != wp.uint32:
            raise TypeError("default_dof_indices must be wp.array[uint32].")
        if int(default_dof_indices.size) != num_wheels:
            raise ValueError(
                f"default_dof_indices length {default_dof_indices.size} must equal num_wheels = {num_wheels}."
            )

        self._num_wheels = num_wheels
        self._device = device if device is not None else wp.get_device()
        self._requires_grad = requires_grad
        self._twist_attr = twist_command_attr
        self._output_attr = joint_target_qd_attr
        self._output_idx = default_dof_indices

        m_np = kinematics_m.numpy() if isinstance(kinematics_m, wp.array) else np.asarray(kinematics_m)
        k_np = kinematics_k.numpy() if isinstance(kinematics_k, wp.array) else np.asarray(kinematics_k)

        # Store M flattened row-major so the kernel can index with ``i*3+j``.
        self._kinematics_m = wp.array(
            m_np.reshape(-1).astype(np.float32),
            dtype=wp.float32,
            device=self._device,
        )
        self._kinematics_k = wp.array(
            k_np.astype(np.float32),
            dtype=wp.float32,
            device=self._device,
        )

        inf = float("inf")
        self._max_linear_speed = (
            wp.full(1, value=inf, dtype=wp.float32, device=self._device)
            if max_linear_speed is None
            else max_linear_speed
        )
        self._max_angular_speed = (
            wp.full(1, value=inf, dtype=wp.float32, device=self._device)
            if max_angular_speed is None
            else max_angular_speed
        )
        self._max_wheel_speed = (
            wp.full(1, value=inf, dtype=wp.float32, device=self._device) if max_wheel_speed is None else max_wheel_speed
        )

        # Internal intermediate buffer: clamped twist shared across the two kernels.
        self._clamped_twist = wp.zeros(3, dtype=wp.float32, device=self._device, requires_grad=requires_grad)

        output_size = int(np.max(default_dof_indices.numpy())) + 1
        self._input_specs: list[tuple[str, Any, int]] = [(self._twist_attr, wp.float32, 3)]
        self._output_specs: list[tuple[str, Any, int]] = [(self._output_attr, wp.float32, output_size)]

    @property
    def num_wheels(self) -> int:
        """Return the number of wheel joints this controller drives."""
        return self._num_wheels

    @property
    def device(self) -> Any:
        """Return the Warp device used for internal buffers."""
        return self._device

    @property
    def requires_grad(self) -> bool:
        """Return whether internal buffers were allocated with gradient support."""
        return self._requires_grad

    def is_graphable(self) -> bool:
        """Return ``True``; this controller supports CUDA graph capture.

        Returns:
            Whether graphable.
        """
        return True

    def input(self) -> Any:
        """Allocate and return a fresh input namespace for this controller.

        Returns:
            The resulting value.
        """
        return _allocate_namespace(self._input_specs, self._device, self._requires_grad)

    def output(self) -> Any:
        """Allocate and return a fresh output namespace for this controller.

        Returns:
            The resulting value.
        """
        return _allocate_namespace(self._output_specs, self._device, self._requires_grad)

    def compute(
        self,
        *,
        inputs: Any,
        outputs: Any,
        dt: float,
    ) -> None:
        """Compute per-wheel angular velocities from the body twist and write them to outputs.

        Args:
            inputs: Namespace whose ``twist_command`` field holds a ``[vx, vy, wz]``
                array. Speed clamping is applied before the kinematic multiply.
            outputs: Namespace whose ``joint_target_qd`` field receives the computed
                wheel angular velocities.
            dt: Timestep in seconds. Unused; present for interface compatibility.
        """
        twist = getattr(inputs, self._twist_attr)
        out = getattr(outputs, self._output_attr)

        wp.launch(
            _clamp_twist_kernel,
            dim=1,
            inputs=[twist, self._max_linear_speed, self._max_angular_speed],
            outputs=[self._clamped_twist],
            device=self._device,
        )
        wp.launch(
            _holonomic_drive_kernel,
            dim=self._num_wheels,
            inputs=[
                self._clamped_twist,
                self._kinematics_m,
                self._kinematics_k,
                self._max_wheel_speed,
                self._output_idx,
            ],
            outputs=[out],
            device=self._device,
        )
