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

# ruff: noqa: D101, D102

"""Tests for machine-readable post-solve tuning assessment helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import omni.kit.test
from isaacsim.robot_setup.sysid.tuning_assessment import (
    ASSESSMENT_PASS,
    ASSESSMENT_WARN,
    _check_validation_holdout,
    _recommended_next_parameters,
    _training_normalized_cost,
)


def _cost_config() -> SimpleNamespace:
    trajectory = SimpleNamespace(
        commands=np.zeros((4, 1), dtype=np.float64),
        positions=np.zeros((4, 1), dtype=np.float64),
        torques=None,
        end_effector_poses=None,
        contact_forces=None,
    )
    weights = SimpleNamespace(
        position_weight=1.0,
        velocity_weight=1.0,
        torque_weight=0.0,
        end_effector_pose_weight=0.0,
        contact_force_weight=0.0,
    )
    return SimpleNamespace(
        trajectory=trajectory,
        max_rollout_steps=4,
        residual_weight_config=weights,
        training_segments=[],
        training_chunks=[],
    )


class TuningAssessmentCostTests(omni.kit.test.AsyncTestCase):
    async def test_single_and_segmented_training_costs_use_same_units(self) -> None:
        selected = SimpleNamespace(cost=4.0)
        config = _cost_config()

        # Four steps x two enabled signals gives eight residuals.
        self.assertEqual(_training_normalized_cost(config, selected), 0.5)

        # Segmented optimizer status costs are already per-residual weighted means.
        selected.cost = 0.5
        config.training_segments = [object()]
        self.assertEqual(_training_normalized_cost(config, selected), 0.5)

        # The legacy training_chunks path uses the same segmented optimizer contract.
        config.training_segments = []
        config.training_chunks = [config.trajectory]
        self.assertEqual(_training_normalized_cost(config, selected), 0.5)


class ValidationHoldoutAssessmentTests(omni.kit.test.AsyncTestCase):
    async def test_metrics_not_from_configured_holdout_have_distinct_issue(self) -> None:
        checks = {}
        issues = []
        prepared = SimpleNamespace(validation_chunks=[])
        result = SimpleNamespace(validation_artifact_source="full_telemetry_fallback")

        _check_validation_holdout(checks, issues, prepared, result, [object()])

        self.assertEqual(checks["validation_holdout"]["status"], ASSESSMENT_WARN)
        self.assertEqual([issue.code for issue in issues], ["no_configured_holdout"])
        self.assertIn(
            "add_independent_validation_chunks_before_broadening_parameters",
            _recommended_next_parameters(None, issues, [object()]),
        )

    async def test_missing_metrics_have_distinct_issue(self) -> None:
        checks = {}
        issues = []
        prepared = SimpleNamespace(validation_chunks=[object()])
        result = SimpleNamespace(validation_artifact_source="configured_chunks")

        _check_validation_holdout(checks, issues, prepared, result, [])

        self.assertEqual(checks["validation_holdout"]["status"], ASSESSMENT_WARN)
        self.assertEqual([issue.code for issue in issues], ["no_validation_metrics_at_all"])

    async def test_configured_holdout_with_metrics_passes(self) -> None:
        checks = {}
        issues = []
        prepared = SimpleNamespace(validation_chunks=[object()])
        result = SimpleNamespace(validation_artifact_source="configured_chunks")

        _check_validation_holdout(checks, issues, prepared, result, [object()])

        self.assertEqual(checks["validation_holdout"]["status"], ASSESSMENT_PASS)
        self.assertEqual(issues, [])
