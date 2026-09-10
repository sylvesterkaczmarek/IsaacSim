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

"""Provide shared assertions and resource lookup for Isaac Sim tests."""

import os
import pathlib

import numpy as np
import warp as wp
from warp._src.types import np_dtype_to_warp_type


def finite_float_elements(*, min_value: float = -1e6, max_value: float = 1e6) -> dict:
    """Handle finite float elements.

    Args:
        min_value: Minimum random value.
        max_value: Maximum random value.

    Returns:
        The resulting value.
    """
    return {
        "allow_nan": False,
        "allow_infinity": False,
        "min_value": np.float32(min_value),
        "max_value": np.float32(max_value),
    }


def resolve_resource_path(path: str) -> str:
    """Resolve resource path.

    Args:
        path: Filesystem path to process.

    Returns:
        The resulting value.
    """
    resource_root = os.environ.get("ISAACSIM_TEST_RESOURCE_ROOT", "")
    if not resource_root:
        raise RuntimeError("ISAACSIM_TEST_RESOURCE_ROOT environment variable is not set")
    resource_dir = pathlib.Path(resource_root)
    if not (resource_dir / "TEST_RESOURCES.md").exists():
        raise RuntimeError(
            f"Unable to resolve test resources path as no 'TEST_RESOURCES.md' file was found under '{resource_dir}'"
        )
    resource_path = resource_dir / path
    if not resource_path.exists():
        raise FileNotFoundError(f"Resource path '{resource_path}' does not exist")
    return str(resource_path)


def check_array(
    a: wp.array | list[wp.array],
    *,
    shape: list[int] | None = None,
    dtype: type | None = None,
    device: str | wp.Device | None = None,
) -> None:
    """Check that the given array matches the expected shape, dtype, and device.

    Args:
        a: Value to check.
        shape: Expected array shape.
        dtype: Expected data type.
        device: Expected device.
    """
    for i, x in enumerate(a if isinstance(a, (list, tuple)) else [a]):
        assert isinstance(x, wp.array), f"check_array [{i}]: {repr(x)} ({type(x)}) is not a Warp array"
        if shape is not None:
            assert tuple(x.shape) == tuple(
                shape
            ), f"check_array [{i}]: Unexpected shape: expected {shape}, got {x.shape}"
        if dtype is not None:
            assert x.dtype == np_dtype_to_warp_type.get(
                dtype, dtype
            ), f"check_array [{i}]: Unexpected dtype: expected {dtype}, got {x.dtype}"
        if device is not None:
            assert x.device == wp.get_device(
                device
            ), f"check_array [{i}]: Unexpected device: expected {device}, got {x.device}"


def check_allclose(
    a: wp.array | np.ndarray | list[wp.array | np.ndarray],
    b: wp.array | np.ndarray | list[wp.array | np.ndarray],
    *,
    rtol: float = 1e-03,
    atol: float = 1e-05,
) -> None:
    """Check that the given arrays are all close.

    Args:
        a: Value to check.
        b: Expected value to compare against.
        rtol: Relative tolerance for value comparisons.
        atol: Absolute tolerance for value comparisons.
    """
    a = a if isinstance(a, (list, tuple)) else [a]
    b = b if isinstance(b, (list, tuple)) else [b]
    assert len(a) == len(b), f"check_allclose: Unexpected input length: {len(a)} != {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if isinstance(x, wp.array):
            x = x.numpy()
        if isinstance(y, wp.array):
            y = y.numpy()
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=f"Iteration {i}" if len(a) > 1 else "")


def check_equal(
    a: wp.array | np.ndarray | list[wp.array | np.ndarray] | list[str] | list[list[str]],
    b: wp.array | np.ndarray | list[wp.array | np.ndarray] | list[str] | list[list[str]],
) -> None:
    """Check that the given arrays or lists of strings are equal.

    Args:
        a: Value to check.
        b: Expected value to compare against.
    """
    # lists
    if isinstance(a, list) or isinstance(b, list):
        assert isinstance(a, list) and isinstance(
            b, list
        ), f"check_equal: Unexpected input type. Expected list, got: {type(a)} and {type(b)}"
        assert len(a) == len(b), f"check_equal: Unexpected input length: {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            check_equal(x, y)
    # strings
    elif isinstance(a, str) or isinstance(b, str):
        assert isinstance(a, str) and isinstance(
            b, str
        ), f"check_equal: Unexpected input type. Expected string, got: {type(a)} and {type(b)}"
        assert a == b
    # arrays
    elif isinstance(a, (wp.array, np.ndarray)) or isinstance(b, (wp.array, np.ndarray)):
        assert isinstance(a, (wp.array, np.ndarray)) and isinstance(
            b, (wp.array, np.ndarray)
        ), f"check_equal: Unexpected input type. Expected array, got: {type(a)} and {type(b)}"
        if isinstance(a, wp.array):
            a = a.numpy()
        if isinstance(b, wp.array):
            b = b.numpy()
        np.testing.assert_equal(a, b)
    else:
        raise ValueError(f"check_equal: Unexpected types for a ({type(a)}) and/or b ({type(b)})")
