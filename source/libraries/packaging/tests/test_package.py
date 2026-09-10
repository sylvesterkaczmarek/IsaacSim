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

"""Test native module archive metadata validation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


def _load_package_module() -> ModuleType:
    """Load the native packaging tool from its source path.

    Returns:
        The resulting value.
    """
    path = Path(__file__).parents[1] / "package.py"
    spec = importlib.util.spec_from_file_location("isaacsim_module_package_tool", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load package tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PACKAGE = _load_package_module()


class NativePackageTests(unittest.TestCase):
    """Test native archive package-manifest contracts."""

    def test_load_manifest_accepts_exact_shared_version_dependencies(self) -> None:
        """Preserve lockstep internal dependencies in native artifacts."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stage = root / "stage"
            manifest_path = stage / "share" / "isaacsim" / "packages" / "acme_robotics" / "package.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "acme_robotics",
                        "version": "1.2.3",
                        "complete": True,
                        "modules": ["acme.robotics"],
                        "python_imports": [],
                        "dependencies": {"acme_core": "==1.2.3"},
                    }
                ),
                encoding="utf-8",
            )
            arguments = _PACKAGE._ArchiveArguments("sdk", root / "build", root / "dist", "acme_robotics", "", "cmake")

            manifest = _PACKAGE._load_manifest(arguments, stage)

            self.assertEqual(manifest["dependencies"], {"acme_core": "==1.2.3"})

    def test_load_manifest_rejects_non_lockstep_dependency(self) -> None:
        """Reject internal metadata that accepts another package release."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stage = root / "stage"
            manifest_path = stage / "share" / "isaacsim" / "packages" / "acme_robotics" / "package.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "acme_robotics",
                        "version": "1.2.3",
                        "complete": True,
                        "modules": ["acme.robotics"],
                        "python_imports": [],
                        "dependencies": {"acme_core": ">=1.2,<2"},
                    }
                ),
                encoding="utf-8",
            )
            arguments = _PACKAGE._ArchiveArguments("sdk", root / "build", root / "dist", "acme_robotics", "", "cmake")

            with self.assertRaisesRegex(RuntimeError, "exact shared-version"):
                _PACKAGE._load_manifest(arguments, stage)


if __name__ == "__main__":
    unittest.main()
