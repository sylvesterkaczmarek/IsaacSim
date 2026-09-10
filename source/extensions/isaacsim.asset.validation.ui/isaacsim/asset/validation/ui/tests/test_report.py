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

"""Tests for validation report serialization."""

import json
import tempfile
from pathlib import Path

import omni.kit.test
from isaacsim.asset.validation.ui.controller import RequirementResult, RequirementStatus
from isaacsim.asset.validation.ui.report import build_report, location_text, write_report
from omni.asset_validator.core import Issue, IssueSeverity
from pxr import Sdf


def _make_result(issues: tuple[Issue, ...]) -> RequirementResult:
    return RequirementResult(
        feature_id="TEST_FEATURE",
        feature_name="Test Feature",
        optional_feature=False,
        code="RC.001",
        name="Clean folder",
        guidance="Remove unexpected files.",
        documentation_path="requirements/rc-001",
        category="RobotCore",
        rule_name="CleanFolder",
        status=RequirementStatus.FAIL if issues else RequirementStatus.PASS,
        issues=issues,
    )


class TestValidationReport(omni.kit.test.AsyncTestCase):
    """Verify report contents and file output."""

    async def test_formats_single_and_multiple_locations(self) -> None:
        """Format Sdf paths and lists of locations into readable text."""
        self.assertEqual("", location_text(None))
        self.assertEqual("/World/Robot", location_text(Sdf.Path("/World/Robot")))
        self.assertEqual(
            "/World/Robot, /World/Cube",
            location_text([Sdf.Path("/World/Robot"), Sdf.Path("/World/Cube")]),
        )

    async def test_report_contains_summary_and_findings(self) -> None:
        """Summarize statuses and serialize each finding."""
        issue = Issue(message="Folder contains an unexpected file.", severity=IssueSeverity.FAILURE)
        report = build_report((_make_result((issue,)),), None, "/tmp/example.usd", ("TEST_FEATURE",))

        self.assertEqual("/tmp/example.usd", report["asset"])
        self.assertEqual(1, report["summary"]["Fail"])
        self.assertEqual(1, report["summary"]["findings"])
        self.assertEqual(["TEST_FEATURE"], report["selected_features"])
        finding = report["requirements"][0]["findings"][0]
        self.assertEqual("Failure", finding["severity"])
        self.assertEqual("Folder contains an unexpected file.", finding["message"])

    async def test_write_report_forces_json_suffix(self) -> None:
        """Write the report to disk using a .json suffix."""
        report = build_report((_make_result(()),), None, "/tmp/example.usd")
        with tempfile.TemporaryDirectory() as directory:
            destination = write_report(Path(directory) / "report", report)

            self.assertEqual(".json", destination.suffix)
            self.assertEqual(report, json.loads(destination.read_text(encoding="utf-8")))
