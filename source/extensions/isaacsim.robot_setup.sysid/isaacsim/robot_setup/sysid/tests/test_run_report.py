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

# ruff: noqa: ANN001, D101, D102

"""Tests for run-report schemas and artifact helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import omni.kit.test
import torch
from isaacsim.robot_setup.sysid.parameter_types import (
    SysIdParameterEntry,
    SysIdParameterType,
)
from isaacsim.robot_setup.sysid.resource_loader import read_schema_resource
from isaacsim.robot_setup.sysid.run_report import (
    SysIdRunReport,
    _command_delay_payload,
    _contained_artifact_path,
    _parameter_rows,
    save_validation_series_overlay_png,
)
from isaacsim.robot_setup.sysid.schema_validation import (
    validate_sysid_run_report_payload,
)


def _empty_report() -> SysIdRunReport:
    return SysIdRunReport(
        report_schema_version=3,
        run_spec={},
        schema_issues=[],
        telemetry_quality={},
        backend="lm",
        simulation_engine="isaac_sim",
        newton_config=None,
        robot_prim_path="/World/Robot",
        train_chunks=[],
        validation_metrics=[],
        validation_summary={},
        validation_artifacts={},
        resampling_diagnostics=[],
        tuning_assessment={},
        parameters=[],
        command_alignment_seconds=0.0,
        actuator_command_delay_seconds={},
        parameters_written=False,
        provenance_path="",
        final_status=None,
        last_accepted_status=None,
        selected_status=None,
    )


class RunReportTests(omni.kit.test.AsyncTestCase):
    async def test_run_report_payload_matches_version_three_contract(self) -> None:
        payload = json.loads(json.dumps(_empty_report().to_dict()))

        self.assertEqual(validate_sysid_run_report_payload(payload), [])
        schema = json.loads(read_schema_resource("sysid_run_report.schema.json"))
        self.assertEqual(schema["properties"]["report_schema_version"]["const"], 3)
        self.assertEqual(set(schema["required"]), set(payload))

    async def test_run_report_validator_rejects_wrong_version(self) -> None:
        payload = _empty_report().to_dict()
        payload["report_schema_version"] = 2

        issues = validate_sysid_run_report_payload(payload)

        self.assertEqual([issue.path for issue in issues if issue.severity == "error"], ["$.report_schema_version"])

    async def test_validation_overlay_writes_png(self) -> None:
        rollout = SimpleNamespace(
            times=[0.0, 0.1, 0.2],
            measured_positions=[[0.0, 0.1, 0.2]],
            simulated_positions=[[0.0, 0.11, 0.19]],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlay.png"

            written, skipped = save_validation_series_overlay_png(
                [rollout],
                path,
                measured_attr="measured_positions",
                simulated_attr="simulated_positions",
                title="Validation",
                ylabel_prefix="q",
            )

            self.assertEqual(skipped, "")
            self.assertEqual(Path(written), path)
            self.assertTrue(path.is_file())

    async def test_requested_animation_path_is_contained_by_artifact_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root.parent / "outside.mp4"

            resolved = _contained_artifact_path(root, str(outside), default_name="default.mp4")
            escaped = _contained_artifact_path(root, "../escaped.gif", default_name="default.mp4")

            self.assertEqual(resolved, root / "outside.mp4")
            self.assertEqual(escaped, root / "escaped.gif")

    async def test_command_delay_payload_quantizes_to_physics_steps(self) -> None:
        entry = SysIdParameterEntry(SysIdParameterType.ACTUATOR_COMMAND_DELAY_SECONDS, dof_index=0)
        prepared = SimpleNamespace(
            config=SimpleNamespace(param_entries=[entry]),
            bridge=SimpleNamespace(actuator_metadata={"physics_dt": 0.01}),
        )
        result = SimpleNamespace(selected_status=SimpleNamespace(theta=[0.026]))

        payload = _command_delay_payload(prepared, result)

        self.assertEqual(payload["rows"][0]["applied_steps"], 3)
        self.assertAlmostEqual(payload["rows"][0]["effective_seconds"], 0.03)

    async def test_parameter_rows_merge_confidence_by_index(self) -> None:
        entry = SysIdParameterEntry(SysIdParameterType.JOINT_FRICTION, dof_index=0)
        prepared = SimpleNamespace(
            config=SimpleNamespace(
                theta_initial=torch.tensor([1.0]),
                param_entries=[entry],
                trajectory=SimpleNamespace(num_joints=1),
            )
        )
        result = SimpleNamespace(
            selected_status=SimpleNamespace(theta=[1.2]),
            parameter_confidence={
                "entries": [
                    {
                        "index": 0,
                        "verdict": "well_constrained",
                        "ten_percent_range": 0.05,
                        "step": 0.01,
                        "at_noise_floor": False,
                    }
                ]
            },
        )

        rows = _parameter_rows(prepared, result)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["delta"], 0.2)
        self.assertEqual(rows[0]["confidence"]["verdict"], "well_constrained")
        self.assertAlmostEqual(rows[0]["confidence"]["ten_percent_range"], 0.05)
