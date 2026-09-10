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

"""Tests for the actionable Check-panel suggestions (pure logic, no UI widgets)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import omni.kit.test
from isaacsim.robot_setup.sysid.ui.check_suggestions import (
    SUGGESTION_KIND_COMMAND_ALIGNMENT,
    SUGGESTION_KIND_FEEDFORWARD,
    extract_check_suggestions,
    format_presolve_summary,
    suppressed_issue_codes,
)


def _report(*, presolve: Any = None, telemetry_quality: Any = None) -> Any:
    return SimpleNamespace(presolve=presolve, telemetry_quality=telemetry_quality or {})


def _feedforward_payload(recommended: Any = "inverse_dynamics", configured: Any = "none") -> Any:
    return {
        "rank": 10,
        "condition": 1.2e3,
        "target_source": "measured_torque",
        "active_column_count": 12,
        "feedforward": {
            "residual_rms": {"none": 12.4, "gravity": 3.1, "inverse_dynamics": 0.9},
            "recommended": recommended,
            "configured": configured,
        },
    }


def _command_lag_payload(*, suggested: Any = 0.5167, common_samples: Any = 15.5) -> Any:
    return {
        "command_lag": {
            "dt_seconds": 1.0 / 30.0,
            "common_mode_lag_samples": common_samples,
            "common_mode_lag_seconds": common_samples / 30.0,
            "residual_spread_samples": 0.8,
            "suggested_command_alignment_seconds": suggested,
        }
    }


class ExtractSuggestionsTests(omni.kit.test.AsyncTestCase):
    """Represent ExtractSuggestionsTests."""

    async def test_feedforward_suggested_only_on_mismatch(self) -> None:
        """Verify feedforward suggested only on mismatch."""
        suggestions = extract_check_suggestions(_report(presolve=_feedforward_payload()))
        self.assertEqual(len(suggestions), 1)
        suggestion = suggestions[0]
        self.assertEqual(suggestion.kind, SUGGESTION_KIND_FEEDFORWARD)
        self.assertEqual(suggestion.value, "inverse_dynamics")
        self.assertIn("inverse_dynamics", suggestion.apply_label)
        self.assertIn("0.9", suggestion.detail)

        matching = _feedforward_payload(recommended="none", configured="none")
        self.assertEqual(extract_check_suggestions(_report(presolve=matching)), [])

    async def test_command_delay_suggested_only_above_threshold(self) -> None:
        """Verify command delay suggested only above threshold."""
        suggestions = extract_check_suggestions(_report(telemetry_quality=_command_lag_payload()))
        self.assertEqual(len(suggestions), 1)
        suggestion = suggestions[0]
        self.assertEqual(suggestion.kind, SUGGESTION_KIND_COMMAND_ALIGNMENT)
        # The value is the additive delta measured on the loaded (already-shifted)
        # trajectory, not an absolute setting.
        self.assertAlmostEqual(float(suggestion.value), 0.5167)

        below = _command_lag_payload(suggested=0.03, common_samples=1.0)
        self.assertEqual(extract_check_suggestions(_report(telemetry_quality=below)), [])

        missing = _command_lag_payload(suggested=None, common_samples=15.5)
        self.assertEqual(extract_check_suggestions(_report(telemetry_quality=missing)), [])

    async def test_negative_common_mode_lag_still_suggests(self) -> None:
        """Verify negative common mode lag still suggests."""
        payload = _command_lag_payload(suggested=-0.2, common_samples=-6.0)
        suggestions = extract_check_suggestions(_report(telemetry_quality=payload))
        self.assertEqual(len(suggestions), 1)
        self.assertAlmostEqual(float(suggestions[0].value), -0.2)

    async def test_suppressed_issue_codes_cover_both_kinds(self) -> None:
        """Verify suppressed issue codes cover both kinds."""
        suggestions = extract_check_suggestions(
            _report(presolve=_feedforward_payload(), telemetry_quality=_command_lag_payload())
        )
        self.assertEqual(len(suggestions), 2)
        codes = suppressed_issue_codes(suggestions)
        self.assertIn("feedforward_mode_mismatch", codes)
        self.assertIn("telemetry_command_lag_common_mode", codes)

    async def test_empty_report_yields_no_suggestions(self) -> None:
        """Verify empty report yields no suggestions."""
        self.assertEqual(extract_check_suggestions(_report()), [])


class PresolveSummaryTests(omni.kit.test.AsyncTestCase):
    """Represent PresolveSummaryTests."""

    async def test_summary_formats_rank_condition_and_target(self) -> None:
        """Verify summary formats rank condition and target."""
        text = format_presolve_summary(_feedforward_payload())
        self.assertIn("rank 10/12", text)
        self.assertIn("1.20e+03", text)
        self.assertIn("measured_torque", text)

    async def test_summary_empty_without_payload(self) -> None:
        """Verify summary empty without payload."""
        self.assertEqual(format_presolve_summary(None), "")
        self.assertEqual(format_presolve_summary({}), "")
        self.assertEqual(format_presolve_summary({"warnings": []}), "")
