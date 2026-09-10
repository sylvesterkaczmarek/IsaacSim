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

"""Unit tests for USD parameter snapshot conversion."""

from __future__ import annotations

import math

import numpy as np
import omni.kit.test
from isaacsim.robot_setup.sysid.usd_parameter_io import (
    LinkUsdSnapshot,
    read_link_usd_snapshot,
    write_link_usd_snapshot,
)
from pxr import Sdf, Usd


class UsdParameterIoTests(omni.kit.test.AsyncTestCase):
    """Verify USD attributes preserve body-frame physical parameters."""

    async def test_inertia_principal_axes_round_trip_preserves_body_tensor(self) -> None:
        """A rotated principal frame must round-trip to the same body-frame tensor."""
        stage = Usd.Stage.CreateInMemory()
        link_path = "/World/Link"
        stage.DefinePrim(link_path, "Xform")

        angle = 0.61
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotation = np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        diagonal = np.array([0.011, 0.023, 0.037], dtype=np.float64)
        principal_axes_quat = np.array(
            [math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)],
            dtype=np.float64,
        )
        expected_body_inertia = rotation @ np.diag(diagonal) @ rotation.T
        snapshot = LinkUsdSnapshot(
            path=link_path,
            mass=2.0,
            com_offset=np.zeros(3, dtype=np.float64),
            diagonal_inertia=diagonal,
            contact_offset=0.0,
            principal_axes_quat=principal_axes_quat,
        )

        write_link_usd_snapshot(stage, snapshot)
        recovered = read_link_usd_snapshot(stage, link_path)

        np.testing.assert_allclose(recovered.diagonal_inertia, diagonal, rtol=0.0, atol=1e-8)
        np.testing.assert_allclose(recovered.inertia_matrix, expected_body_inertia, rtol=0.0, atol=1e-8)

    async def test_link_write_preserves_contact_offset_by_default(self) -> None:
        """Ordinary parameter writeback must not author runtime-removed contact tuning."""
        stage = Usd.Stage.CreateInMemory()
        link_path = "/World/Link"
        prim = stage.DefinePrim(link_path, "Xform")
        contact_attr = prim.CreateAttribute(
            "physxCollision:contactOffset",
            Sdf.ValueTypeNames.Float,
            custom=False,
        )
        contact_attr.Set(0.75)
        snapshot = LinkUsdSnapshot(
            path=link_path,
            mass=2.0,
            com_offset=np.zeros(3, dtype=np.float64),
            diagonal_inertia=np.ones(3, dtype=np.float64),
            contact_offset=0.25,
        )

        write_link_usd_snapshot(stage, snapshot)
        self.assertAlmostEqual(float(contact_attr.Get()), 0.75)

        write_link_usd_snapshot(stage, snapshot, write_contact=True)
        self.assertAlmostEqual(float(contact_attr.Get()), 0.25)
