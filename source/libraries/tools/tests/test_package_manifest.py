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

"""Test the shared standalone-library package manifest contract."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MANIFEST_CONTRACT_PATH = Path(__file__).resolve().parents[2] / "package_manifest.py"
MANIFEST_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "isaacsim_package_manifest_tests", MANIFEST_CONTRACT_PATH
)
if MANIFEST_CONTRACT_SPEC is None or MANIFEST_CONTRACT_SPEC.loader is None:
    raise RuntimeError(f"Cannot load package manifest contract: {MANIFEST_CONTRACT_PATH}")
manifest_contract = importlib.util.module_from_spec(MANIFEST_CONTRACT_SPEC)
sys.modules[MANIFEST_CONTRACT_SPEC.name] = manifest_contract
MANIFEST_CONTRACT_SPEC.loader.exec_module(manifest_contract)


class PackageManifestTests(unittest.TestCase):
    """Validate shared package identity and inventory relationships."""

    def _write_manifest(self, root: Path, **updates: object) -> Path:
        manifest: dict[str, object] = {
            "name": "acme_runtime",
            "version": "7.0.0.dev0",
            "complete": True,
            "dependencies": {},
            "modules": ["acme.runtime.logging"],
            "python_imports": [],
        }
        manifest.update(updates)
        path = root / "package.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_accepts_complete_package_inventory(self) -> None:
        """Test accepts complete package inventory."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_manifest(Path(temporary_directory))

            manifest = manifest_contract.load_package_manifest(
                path,
                "acme_runtime",
                expected_version="7.0.0.dev0",
            )

            self.assertEqual(manifest["modules"], ["acme.runtime.logging"])

    def test_rejects_missing_module_inventory(self) -> None:
        """Test rejects missing module inventory."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_manifest(Path(temporary_directory))
            manifest = json.loads(path.read_text(encoding="utf-8"))
            del manifest["modules"]
            path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "invalid module inventory"):
                manifest_contract.load_package_manifest(path, "acme_runtime")

    def test_rejects_duplicate_modules(self) -> None:
        """Test rejects duplicate modules."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_manifest(
                Path(temporary_directory),
                modules=["acme.runtime.logging", "acme.runtime.logging"],
            )

            with self.assertRaisesRegex(RuntimeError, "invalid module inventory"):
                manifest_contract.load_package_manifest(path, "acme_runtime")


if __name__ == "__main__":
    unittest.main()
