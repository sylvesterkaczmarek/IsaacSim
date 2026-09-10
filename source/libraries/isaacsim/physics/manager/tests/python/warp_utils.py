# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Provide Warp array helpers and kernels for physics tensor tests."""

from __future__ import annotations

import warp as wp


@wp.kernel
def _arange_k(a: wp.array(dtype=wp.int32)) -> None:
    tid = wp.tid()
    a[tid] = tid


def arange(n: int, device: str = "cpu") -> wp.array:
    """Create a synchronized sequence of 32-bit integer indices.

    Args:
        n: Number of indices.
        device: Warp device that receives the array.

    Returns:
        Warp array containing integers from zero through ``n - 1``.

    """
    a = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(kernel=_arange_k, dim=n, inputs=[a], device=device)
    wp.synchronize()
    return a


@wp.kernel
def _linspace_k(a: wp.array(dtype=wp.float32), offset: wp.float32, step: wp.float32) -> None:
    tid = wp.tid()
    a[tid] = offset + float(tid) * step


def linspace(
    n: int,
    start: float,
    end: float,
    include_end: bool = False,
    include_start: bool = True,
    device: str = "cpu",
) -> wp.array:
    """Create a synchronized evenly spaced floating-point sequence.

    Args:
        n: Number of values.
        start: Lower sequence bound.
        end: Upper sequence bound.
        include_end: Whether the upper bound is part of the sequence.
        include_start: Whether the lower bound is part of the sequence.
        device: Warp device that receives the array.

    Returns:
        Warp array containing the generated sequence.

    """
    d = n - 1
    if not include_start:
        d += 1
    if not include_end:
        d += 1

    step = (end - start) / d
    if not include_start:
        offset = start + step
    else:
        offset = start

    a = wp.empty(n, dtype=wp.float32, device=device)
    wp.launch(kernel=_linspace_k, dim=n, inputs=[a, offset, step], device=device)
    wp.synchronize()
    return a


@wp.kernel
def _fill_float32_k(a: wp.array(dtype=wp.float32), value: wp.float32) -> None:
    tid = wp.tid()
    a[tid] = value


def fill_float32(n: int, value: float = 0.0, device: str = "cpu") -> wp.array:
    """Create a synchronized scalar array filled with one value.

    Args:
        n: Number of scalar elements.
        value: Value written to every element.
        device: Warp device that receives the array.

    Returns:
        Filled single-precision Warp array.

    """
    a = wp.empty(n, dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_float32_k, dim=n, inputs=[a, value], device=device)
    wp.synchronize()
    return a


@wp.kernel
def _fill_vec3_k(a: wp.array(dtype=wp.vec3), value: wp.vec3) -> None:
    tid = wp.tid()
    a[tid] = value


def fill_vec3(n: int, value: wp.vec3 = wp.vec3(0.0, 0.0, 0.0), device: str = "cpu") -> wp.array:
    """Create a synchronized vector array filled with one value.

    Args:
        n: Number of vector elements.
        value: Value written to every element.
        device: Warp device that receives the array.

    Returns:
        Filled three-component vector Warp array.

    """
    a = wp.empty(n, dtype=wp.vec3, device=device)
    wp.launch(kernel=_fill_vec3_k, dim=n, inputs=[a, value], device=device)
    wp.synchronize()
    return a


@wp.kernel
def _random_k(seed: int, a: wp.array(dtype=float), lower: float, upper: float) -> None:
    tid = wp.tid()
    state = wp.rand_init(seed, tid)
    a[tid] = wp.randf(state, lower, upper)


def random(n: int, lower: float = 0.0, upper: float = 1.0, device: str = "cpu", seed: int = 42) -> wp.array:
    """Create synchronized uniformly distributed random values.

    Args:
        n: Number of random values.
        lower: Inclusive lower distribution bound.
        upper: Exclusive upper distribution bound.
        device: Warp device that receives the array.
        seed: Random-number seed.

    Returns:
        Warp array containing the generated values.

    """
    a = wp.zeros(n, dtype=float, device=device)
    wp.launch(kernel=_random_k, dim=n, inputs=[seed, a, lower, upper], device=device)
    wp.synchronize()
    return a


@wp.kernel
def _compute_dof_forces_k(
    pos: wp.array(dtype=float, ndim=2),
    vel: wp.array(dtype=float, ndim=2),
    force: wp.array(dtype=float, ndim=2),
    stiffness: float,
    damping: float,
) -> None:
    i, j = wp.tid()
    pos_target = 0.0
    force[i, j] = stiffness * (pos_target - pos[i, j]) - damping * vel[i, j]


def compute_dof_forces(
    pos: wp.array,
    vel: wp.array,
    force: wp.array,
    stiffness: float,
    damping: float,
    device: str = "cpu",
) -> None:
    """Compute spring-damper forces into a caller-provided array.

    Args:
        pos: Current degree-of-freedom positions.
        vel: Current degree-of-freedom velocities.
        force: Output array that receives computed forces.
        stiffness: Position-error gain.
        damping: Velocity gain.
        device: Warp device used to launch the kernel.

    """
    wp.launch(
        kernel=_compute_dof_forces_k, dim=force.shape, inputs=[pos, vel, force, stiffness, damping], device=device
    )
    wp.synchronize()


@wp.kernel
def _pack_wrench_k(
    f: wp.array(dtype=wp.vec3),
    tau: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.float32, ndim=2),
) -> None:
    i = wp.tid()
    out[i, 0] = f[i][0]
    out[i, 1] = f[i][1]
    out[i, 2] = f[i][2]
    out[i, 3] = tau[i][0]
    out[i, 4] = tau[i][1]
    out[i, 5] = tau[i][2]
    out[i, 6] = pos[i][0]
    out[i, 7] = pos[i][1]
    out[i, 8] = pos[i][2]


def pack_wrench(
    forces: wp.array,
    torques: wp.array | None = None,
    positions: wp.array | None = None,
) -> wp.array:
    """Pack force, torque, and application-position arrays by row.

    Args:
        forces: Force vectors as a vector array or flat scalar array.
        torques: Optional torque vectors in the same supported layouts.
        positions: Optional force application positions in the same layouts.

    Returns:
        Warp array with force, torque, and position components in nine columns.

    """
    device = forces.device

    def _to_vec3(x: wp.array | None, n: int) -> wp.array:
        if x is None:
            return wp.zeros(n, dtype=wp.vec3, device=device)
        if x.dtype == wp.vec3:
            if x.shape[0] != n:
                raise ValueError(f"pack_wrench: row count mismatch (forces={n}, got {x.shape[0]})")
            return x
        if x.size != n * 3:
            raise ValueError(f"pack_wrench: row count mismatch (forces={n}, got {x.size} elements, expected {n * 3})")
        return wp.array(x.numpy().reshape(n, 3), dtype=wp.vec3, device=device)

    if forces.dtype != wp.vec3:
        n = forces.size // 3
        forces = _to_vec3(forces, n)
    else:
        n = forces.shape[0]

    tau_wp = _to_vec3(torques, n)
    pos_wp = _to_vec3(positions, n)

    out = wp.zeros((n, 9), dtype=wp.float32, device=device)
    wp.launch(_pack_wrench_k, dim=n, inputs=[forces, tau_wp, pos_wp, out], device=device)
    wp.synchronize()
    return out


def wp_allclose(a: object, b: object, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Compare Warp or NumPy-compatible values element by element.

    Args:
        a: First array, scalar, or sequence.
        b: Second array, scalar, or sequence.
        rtol: Relative comparison tolerance.
        atol: Absolute comparison tolerance.

    Returns:
        True when all corresponding values satisfy the tolerances.

    """
    import numpy as _np

    if isinstance(a, wp.array):
        a = a.numpy()
    if isinstance(b, wp.array):
        b = b.numpy()
    return bool(_np.allclose(a, b, rtol=rtol, atol=atol))
