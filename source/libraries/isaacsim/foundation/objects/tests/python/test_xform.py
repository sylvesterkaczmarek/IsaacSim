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

"""Test xform behavior."""

from typing import Any

import hypothesis
import hypothesis.extra.numpy
import isaacsim_test
import numpy as np
import pytest
import warp as wp
from isaacsim.foundation.objects import Xform

from .fixtures import stage  # noqa: F401 - imported so pytest can discover the fixture

"""
Utility functions.
"""


def _get_paths(stage: Any, populate: Any) -> Any:
    paths = [f"/World/Prim{i}" for i in range(5)]
    if populate:
        for path in paths:
            stage.define_prim(path, "Xform")
        paths = "/World/Prim.*"
    return paths


def _normalize_quaternions(quats: np.ndarray) -> np.ndarray:
    """Normalise (N, 4) quaternion array, mirroring ``GfQuat::GetNormalized()``.

    The norm is computed in float64 because squaring small float32 values underflows into the
    subnormal range and destroys precision. Rows shorter than ``GF_MIN_VECTOR_LENGTH`` become
    identity, which is what USD does.

    Args:
        quats: Quats values.

    Returns:
        The resulting value.
    """
    quats = quats.astype(np.float64)
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    degenerate = norms < 1e-10  # GF_MIN_VECTOR_LENGTH
    normalized = quats / np.where(degenerate, 1.0, norms)
    identity = np.zeros_like(normalized)
    identity[:, 0] = 1.0
    return np.where(degenerate, identity, normalized).astype(np.float32)


"""
Test cases.
"""


@hypothesis.given(
    positions=hypothesis.extra.numpy.arrays(
        dtype=np.float32, shape=(5, 3), elements=isaacsim_test.finite_float_elements()
    ),
    orientations=hypothesis.extra.numpy.arrays(
        dtype=np.float32, shape=(5, 4), elements=isaacsim_test.finite_float_elements(min_value=0.0, max_value=1.0)
    ),
)
@pytest.mark.parametrize("populate", [True, False])
def test_world_poses(capsys: Any, stage: Any, populate: Any, positions: Any, orientations: Any) -> None:
    """Test world poses.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
        populate: Whether to populate the test stage.
        positions: Positions values.
        orientations: Orientations values.
    """
    orientations = _normalize_quaternions(orientations)
    prims = Xform(_get_paths(stage, populate))
    # initial get
    output_positions, output_orientations = prims.get_world_poses()
    isaacsim_test.check_array(output_positions, shape=(5, 3), dtype=wp.float64)
    isaacsim_test.check_array(output_orientations, shape=(5, 4), dtype=wp.float64)
    # round-trip
    prims.set_world_poses(positions, orientations)
    output_positions, output_orientations = prims.get_world_poses()
    isaacsim_test.check_array(output_positions, shape=(5, 3), dtype=wp.float64)
    isaacsim_test.check_array(output_orientations, shape=(5, 4), dtype=wp.float64)
    isaacsim_test.check_allclose(positions, output_positions)
    isaacsim_test.check_allclose(orientations, output_orientations)


@hypothesis.given(
    translations=hypothesis.extra.numpy.arrays(
        dtype=np.float32, shape=(5, 3), elements=isaacsim_test.finite_float_elements()
    ),
    orientations=hypothesis.extra.numpy.arrays(
        dtype=np.float32, shape=(5, 4), elements=isaacsim_test.finite_float_elements(min_value=0.0, max_value=1.0)
    ),
)
@pytest.mark.parametrize("populate", [True, False])
def test_local_poses(capsys: Any, stage: Any, populate: Any, translations: Any, orientations: Any) -> None:
    """Test local poses.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
        populate: Whether to populate the test stage.
        translations: Translations values.
        orientations: Orientations values.
    """
    orientations = _normalize_quaternions(orientations)
    prims = Xform(_get_paths(stage, populate))
    # initial get
    output_translations, output_orientations = prims.get_local_poses()
    isaacsim_test.check_array(output_translations, shape=(5, 3), dtype=wp.float64)
    isaacsim_test.check_array(output_orientations, shape=(5, 4), dtype=wp.float64)
    # round-trip
    prims.set_local_poses(translations, orientations)
    output_translations, output_orientations = prims.get_local_poses()
    isaacsim_test.check_array(output_translations, shape=(5, 3), dtype=wp.float64)
    isaacsim_test.check_array(output_orientations, shape=(5, 4), dtype=wp.float64)
    isaacsim_test.check_allclose(translations, output_translations)
    isaacsim_test.check_allclose(orientations, output_orientations)


@hypothesis.given(
    scales=hypothesis.extra.numpy.arrays(
        dtype=np.float32, shape=(5, 3), elements=isaacsim_test.finite_float_elements(min_value=0.001)
    ),
)
@pytest.mark.parametrize("populate", [True, False])
def test_local_scales(capsys: Any, stage: Any, populate: Any, scales: Any) -> None:
    """Test local scales.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
        populate: Whether to populate the test stage.
        scales: Scales values.
    """
    prims = Xform(_get_paths(stage, populate))
    # initial get
    output = prims.get_local_scales()
    isaacsim_test.check_array(output, shape=(5, 3), dtype=wp.float64)
    # round-trip
    prims.set_local_scales(scales)
    output = prims.get_local_scales()
    isaacsim_test.check_array(output, shape=(5, 3), dtype=wp.float64)
    isaacsim_test.check_allclose(scales, output)


@hypothesis.given(
    values=hypothesis.extra.numpy.arrays(dtype=np.bool_, shape=(5, 1)),
)
@pytest.mark.parametrize("populate", [True, False])
def test_visibilities(capsys: Any, stage: Any, populate: Any, values: Any) -> None:
    """Test visibilities.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
        populate: Whether to populate the test stage.
        values: Values exercised by the test.
    """
    prims = Xform(_get_paths(stage, populate))
    # initial get
    output = prims.get_visibilities()
    isaacsim_test.check_array(output, shape=(5, 1), dtype=wp.uint8)
    # round-trip
    prims.set_visibilities(values)
    output = prims.get_visibilities()
    isaacsim_test.check_array(output, shape=(5, 1), dtype=wp.uint8)
    isaacsim_test.check_equal(values, output)
