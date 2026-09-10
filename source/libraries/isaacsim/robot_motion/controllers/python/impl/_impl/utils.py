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

"""Internal helpers for Newton controllers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp


def _idx_max(idx: wp.array) -> int:
    """Return one past the maximum index value in ``idx``.

    Args:
        idx: Index to normalize.

    Returns:
        The resulting value.
    """
    return int(np.max(idx.numpy())) + 1


def _normalize(v: Any, name: str) -> np.ndarray:
    """Return a normalized unit-length vector from any array-like input.

    Args:
        v: Value to normalize. ``wp.array`` inputs are converted to numpy first.
        name: Parameter name used in error messages.

    Returns:
        Normalized vector of shape ``(3,)``.

    Raises:
        ValueError: If ``v`` is not 3-element or has zero magnitude.
    """
    if isinstance(v, wp.array):
        v = v.numpy()
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-element vector, got shape {arr.shape}.")
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return arr / norm


def _normalize_indices(
    idx: Any,
    default_idx: wp.array,
    *,
    name: str,
) -> wp.array:
    if idx is None:
        return default_idx
    if (not isinstance(idx, wp.array)) or (idx.dtype != wp.uint32):
        raise TypeError(f"Port '{name}': idx must be wp.array[uint32] or None, got {type(idx).__name__}.")
    if idx.size != default_idx.size:
        raise TypeError(
            f"Port '{name}': indices must be the same size as default_dof_indices: {idx.size} != {default_idx.size}."
        )
    return idx


def _normalize_parameter_port(
    value: Any,
    expected_size: int,
    dtype: Any,
    device: Any,
    requires_grad: bool,
    *,
    name: str,
) -> tuple[str | None, wp.array | None]:
    if isinstance(value, str):
        return value, None
    if isinstance(value, wp.array):
        if value.size != expected_size:
            raise ValueError(f"Port '{name}': baked array length {value.size} must equal {expected_size}.")
        if value.dtype != dtype:
            raise TypeError(f"Port '{name}': baked array dtype {value.dtype} must equal {dtype}.")
        baked = wp.zeros(expected_size, dtype=dtype, device=device, requires_grad=requires_grad)
        wp.copy(baked, value)
        return None, baked
    raise TypeError(f"Port '{name}': must be wp.array[{dtype}] or str (attr name); got {type(value).__name__}.")


def _allocate_namespace(
    specs: list[tuple[str, Any, int]],
    device: Any,
    requires_grad: bool,
) -> SimpleNamespace:
    merged: dict[str, tuple[Any, int]] = {}
    for attr, dtype, size in specs:
        if attr in merged:
            prev_dtype, prev_size = merged[attr]
            if prev_dtype != dtype:
                raise TypeError(f"Field '{attr}': two ports declared incompatible dtypes ({prev_dtype} vs {dtype}).")
            merged[attr] = (dtype, max(prev_size, size))
        else:
            merged[attr] = (dtype, size)
    return SimpleNamespace(
        **{
            attr: wp.zeros(size, dtype=dtype, device=device, requires_grad=requires_grad)
            for attr, (dtype, size) in merged.items()
        }
    )
