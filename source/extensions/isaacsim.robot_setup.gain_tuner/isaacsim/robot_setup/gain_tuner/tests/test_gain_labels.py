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

"""Unit tests for per-source gain-field label selection."""

from __future__ import annotations

import omni.kit.test
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner.gain_sources import (
    DAMPING_LABEL,
    KD_LABEL,
    KI_LABEL,
    KP_LABEL,
    MJC_DAMPING_LABEL,
    MJC_STIFFNESS_LABEL,
    STIFFNESS_LABEL,
)


class TestGainLabels(omni.kit.test.AsyncTestCase):
    """Pure per-source gain-label selection (``gain_labels_for_source``)."""

    async def test_physics_drive_labels(self) -> None:
        """PhysicsDrive uses stiffness/damping and never exposes a Ki label."""
        labels = gain_tuner.gain_labels_for_source(gain_tuner.GainSource.PHYSICS_DRIVE)
        self.assertEqual((labels.kp_label, labels.kd_label, labels.ki_label), (STIFFNESS_LABEL, DAMPING_LABEL, None))

    async def test_actuator_pd_labels(self) -> None:
        """A PD actuator uses Kp/Kd and no Ki label."""
        labels = gain_tuner.gain_labels_for_source(gain_tuner.GainSource.ACTUATOR, is_pid=False)
        self.assertEqual((labels.kp_label, labels.kd_label, labels.ki_label), (KP_LABEL, KD_LABEL, None))

    async def test_actuator_pid_labels(self) -> None:
        """A PID actuator uses Kp/Kd/Ki."""
        labels = gain_tuner.gain_labels_for_source(gain_tuner.GainSource.ACTUATOR, is_pid=True)
        self.assertEqual((labels.kp_label, labels.kd_label, labels.ki_label), (KP_LABEL, KD_LABEL, KI_LABEL))

    async def test_none_source_falls_back_to_stiffness(self) -> None:
        """The NONE source falls back to stiffness/damping labels."""
        labels = gain_tuner.gain_labels_for_source(gain_tuner.GainSource.NONE)
        self.assertEqual((labels.kp_label, labels.kd_label, labels.ki_label), (STIFFNESS_LABEL, DAMPING_LABEL, None))

    async def test_mujoco_labels(self) -> None:
        """MuJoCo-native gains use the mjc stiffness/damping labels and no Ki."""
        labels = gain_tuner.gain_labels_for_source(gain_tuner.GainSource.MUJOCO)
        self.assertEqual(
            (labels.kp_label, labels.kd_label, labels.ki_label), (MJC_STIFFNESS_LABEL, MJC_DAMPING_LABEL, None)
        )
