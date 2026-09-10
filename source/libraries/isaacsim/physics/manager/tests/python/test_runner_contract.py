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

"""Verify error propagation from the shared scenario runner."""

from __future__ import annotations

from unittest.mock import patch

import _legacy_runner
import pytest


class _Scenario:
    def __init__(self, test_case: object, device: object) -> None:
        self.test_case = test_case
        self.device = device


class _AttributeErrorRunner:
    def __init__(self, scenario: object, engine: str, frontend: str) -> None:
        self.scenario = scenario
        self.engine = engine
        self.frontend = frontend

    def start(self) -> None:
        raise AttributeError("missing adapter method")

    def simulate(self) -> None:
        raise AssertionError("simulate must not run after start fails")

    def stop(self) -> None:
        return


class TestRunnerContract:
    """Check that runner failures remain visible to pytest."""

    def test_attribute_error_is_not_converted_to_skip(self) -> None:
        """Verify that adapter attribute errors propagate instead of skipping."""
        with patch.object(_legacy_runner, "RunnerInMemory", _AttributeErrorRunner):
            try:
                _legacy_runner.run_scenario(
                    self,
                    _Scenario,
                    "newton",
                    _legacy_runner.cpu_device(),
                )
            except BaseException as exc:
                assert isinstance(exc, AttributeError)
                assert str(exc) == "missing adapter method"
            else:
                pytest.fail("run_scenario did not propagate the runner failure")
