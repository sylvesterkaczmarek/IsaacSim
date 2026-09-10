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

"""Tests that the headless SysID tools resolve artifact output paths safely."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import omni.kit.test

_TOOLS_DIR = Path(__file__).resolve().parents[4] / "tools"


def _load_tool(name: str) -> ModuleType:
    path = _TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}_path_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load tool module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_TOOLS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class TestHeadlessToolPaths(omni.kit.test.AsyncTestCase):
    """Path-resolution helpers used by the standalone headless tools."""

    def test_artifact_directory_defaults_to_result_parent(self) -> None:
        """The artifact directory defaults to the result JSON's resolved parent."""
        tool = _load_tool("headless_sysid_solve")
        with tempfile.TemporaryDirectory(prefix="sysid_artifacts_") as tmp_dir:
            root = Path(tmp_dir)
            result_path = root / "friction_result.json"
            self.assertEqual(tool._artifact_output_dir(str(result_path)), root.resolve())

    def test_apply_tool_rejects_the_source_as_its_own_output(self) -> None:
        """Reject copying a stage onto itself (source == output)."""
        tool = _load_tool("headless_sysid_apply_theta_to_stage")
        source = Path("robot.usda")
        with self.assertRaisesRegex(ValueError, "must be different"):
            tool._copy_stage(source, source, overwrite=True)
