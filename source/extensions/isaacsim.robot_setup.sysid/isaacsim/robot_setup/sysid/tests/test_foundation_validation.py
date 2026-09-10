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

"""Validation tests for portable SysID foundation contracts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import omni.kit.test
from isaacsim.robot_setup.sysid.parameter_types import (
    SysIdParameterEntry,
    SysIdParameterType,
)
from isaacsim.robot_setup.sysid.provenance import (
    ProvenanceWriter,
    SysIdProvenanceRecord,
    split_parameters_by_category,
)
from isaacsim.robot_setup.sysid.residual_config import ResidualWeightConfig
from isaacsim.robot_setup.sysid.residual_engine import (
    ResidualSignalType,
    ResidualSpec,
    WeightedResidualEngine,
)
from isaacsim.robot_setup.sysid.resource_loader import read_schema_resource
from isaacsim.robot_setup.sysid.schema_validation import (
    validate_sysid_run_spec_payload,
    validate_topic_mapping_payload,
)


class FoundationValidationTests(omni.kit.test.AsyncTestCase):
    """Reject invalid numeric values before they reach an optimizer."""

    async def test_schema_rejects_nonfinite_solver_and_residual_numbers(self) -> None:
        """Report NaN and infinity wherever finite numeric values are required."""
        issues = validate_sysid_run_spec_payload(
            {
                "schema_version": 2,
                "solver": {
                    "epsilon": float("nan"),
                    "damping_initial": float("inf"),
                    "cma_seed": -1,
                    "bo_seed": -2,
                    "gd_seed": -3,
                },
                "residuals": {
                    "position_weight": 1.0,
                    "velocity_weight": float("nan"),
                },
            }
        )

        self.assertEqual(
            {issue.path for issue in issues if issue.severity == "error"},
            {
                "$.solver.epsilon",
                "$.solver.damping_initial",
                "$.solver.cma_seed",
                "$.solver.bo_seed",
                "$.solver.gd_seed",
                "$.residuals.velocity_weight",
            },
        )

    async def test_residual_config_rejects_negative_or_nonfinite_weights(self) -> None:
        """Reject invalid channel and sample weights at construction."""
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "position_weight"):
                    ResidualWeightConfig(position_weight=value)

        with self.assertRaisesRegex(ValueError, "sample_weights"):
            ResidualWeightConfig(sample_weights=np.asarray([1.0, -0.1]))

    async def test_residual_engine_rejects_invalid_direct_specs(self) -> None:
        """Protect callers that construct a residual engine without a config."""
        with self.assertRaisesRegex(ValueError, "position"):
            WeightedResidualEngine((ResidualSpec(ResidualSignalType.POSITION, -1.0),))

    async def test_schema_rejects_unparseable_chunks_and_missing_parameter_type(
        self,
    ) -> None:
        """Keep validation errors ahead of raw run-spec conversion failures."""
        issues = validate_sysid_run_spec_payload(
            {
                "schema_version": 2,
                "telemetry": {
                    "chunks": [
                        {
                            "role": "train",
                            "start": "not-a-number",
                            "end": 1.0,
                            "weight": float("nan"),
                        }
                    ]
                },
                "parameters": {
                    "selected": [
                        {
                            "min": 0.0,
                            "initial": 0.5,
                            "max": 1.0,
                        }
                    ]
                },
            }
        )

        self.assertEqual(
            {issue.path for issue in issues if issue.severity == "error"},
            {
                "$.telemetry.chunks[0].start",
                "$.telemetry.chunks[0].weight",
                "$.parameters.selected[0].param_type",
            },
        )

    async def test_topic_mapping_recognizes_torque_semantics(self) -> None:
        """Keep the validator and bundled topic-map schema aligned with the loader."""
        issues = validate_topic_mapping_payload(
            {
                "position_topic": "/joint_states",
                "torque_semantics": "external",
            }
        )
        self.assertEqual(issues, [])

        schema = json.loads(read_schema_resource("topic_map.schema.json"))
        self.assertEqual(schema["properties"]["torque_semantics"]["enum"], ["link_side", "external"])

        issues = validate_topic_mapping_payload(
            {
                "position_topic": "/joint_states",
                "torque_topic": "/joint_torque",
            }
        )
        self.assertEqual(
            [issue.path for issue in issues if issue.severity == "error"],
            ["$.torque_semantics"],
        )

    async def test_provenance_rejects_parameter_count_mismatch(self) -> None:
        """Never write a silently truncated parameter audit record."""
        entries = [SysIdParameterEntry(SysIdParameterType.JOINT_FRICTION, 0)]

        with self.assertRaisesRegex(ValueError, "count mismatch"):
            split_parameters_by_category(entries, [], num_joints=1)

    async def test_schema_validates_sampling_residuals_and_newton_substeps(
        self,
    ) -> None:
        """Keep warn-first validation aligned with the bundled schemas."""
        issues = validate_sysid_run_spec_payload(
            {
                "schema_version": 2,
                "solver": {"max_rollout_steps": 1},
                "residuals": {"sample_weights": [1.0, float("nan"), -1.0]},
                "simulation": {"newton": {"featherstone_substeps": 0}},
            }
        )

        self.assertEqual(
            {issue.path for issue in issues if issue.severity == "error"},
            {
                "$.solver.max_rollout_steps",
                "$.residuals.sample_weights[1]",
                "$.residuals.sample_weights[2]",
                "$.simulation.newton.featherstone_substeps",
            },
        )
        run_schema = json.loads(read_schema_resource("sysid_run_spec.schema.json"))
        telemetry_properties = run_schema["properties"]["telemetry"]["properties"]
        self.assertIn("auto_split", telemetry_properties)
        self.assertIn("auto_split_train_fraction", telemetry_properties)
        self.assertIn("auto_split_min_chunk_seconds", telemetry_properties)
        residual_schema = json.loads(read_schema_resource("residual_weights.schema.json"))
        self.assertTrue(residual_schema["$id"].endswith("/residual_weights.schema.json"))

    async def test_provenance_json_sidecar_round_trips(self) -> None:
        """Write and reload a provenance record through the public sidecar API."""
        record = SysIdProvenanceRecord(
            schema_version="1",
            dataset_source_type="csv",
            dataset_source_path="run.csv",
            optimized_at="2026-07-29T00:00:00+00:00",
            optimizer_backend="levenberg_marquardt",
            final_cost=0.25,
            iterations=4,
            robot_prim_path="/World/Robot",
            extra={"accepted": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            written = ProvenanceWriter().export_json_sidecar(str(path), record)
            restored = SysIdProvenanceRecord.from_dict(json.loads(Path(written).read_text(encoding="utf-8")))

        self.assertEqual(restored.to_dict(), record.to_dict())
