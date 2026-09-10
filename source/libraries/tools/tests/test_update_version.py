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

"""Test standalone library version synchronization."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

UPDATE_VERSION_PATH = Path(__file__).parents[1] / "update_version.py"
UPDATE_VERSION_SPEC = importlib.util.spec_from_file_location("isaacsim_update_version", UPDATE_VERSION_PATH)
if UPDATE_VERSION_SPEC is None or UPDATE_VERSION_SPEC.loader is None:
    raise RuntimeError(f"Cannot load version tool from {UPDATE_VERSION_PATH}")
update_version = importlib.util.module_from_spec(UPDATE_VERSION_SPEC)
sys.modules[UPDATE_VERSION_SPEC.name] = update_version
UPDATE_VERSION_SPEC.loader.exec_module(update_version)


class VersionSynchronizationTests(unittest.TestCase):
    """Test shared version updates and consistency checks."""

    def setUp(self) -> None:
        """Handle setUp."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.root.joinpath("VERSION").write_text("6.1.0\n", encoding="utf-8")
        self._write_project("isaacsim/common", "isaacsim_common", [])
        self.consumer = self._write_project(
            "isaacsim/foundation",
            "isaacsim-foundation",
            ["usd-exchange==2.3.0"],
            ["isaacsim_common"],
        )

    def tearDown(self) -> None:
        """Handle tearDown."""
        self.temporary_directory.cleanup()

    def _write_project(
        self,
        relative_path: str,
        name: str,
        dependencies: list[str],
        internal_dependencies: list[str] | None = None,
    ) -> Path:
        path = self.root / relative_path / "pyproject.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        dependency_lines = "\n".join(f'    "{dependency}",' for dependency in dependencies)
        dynamic = '["dependencies"]' if internal_dependencies else "[]"
        dynamic_metadata = ""
        if internal_dependencies:
            internal_names = ", ".join(f'"{dependency}"' for dependency in internal_dependencies)
            dynamic_metadata = (
                "\n[[tool.dynamic-metadata]]\n"
                'provider = { path = "../../packaging", module = "isaacsim_libraries_metadata" }\n'
                'field = "dependencies"\n'
                f"names = [{internal_names}]\n"
            )
        path.write_text(
            f'[project]\nname = "{name}"\ndynamic = {dynamic}\ndependencies = [\n{dependency_lines}\n]\n'
            f"{dynamic_metadata}",
            encoding="utf-8",
        )
        return path

    def test_update_version_changes_only_the_shared_version(self) -> None:
        """Update only the canonical version file when dependencies are dynamic."""
        consumer_before = self.consumer.read_text(encoding="utf-8")

        update_version.update_version(self.root, "7.0.0a1")

        self.assertEqual(self.root.joinpath("VERSION").read_text(encoding="utf-8"), "7.0.0a1\n")
        self.assertEqual(self.consumer.read_text(encoding="utf-8"), consumer_before)

    def test_check_version_sync_rejects_a_static_internal_pin(self) -> None:
        """Reject internal requirements declared in the static dependency list."""
        self._write_project(
            "isaacsim/foundation",
            "isaacsim-foundation",
            ["isaacsim_common==6.1.0", "usd-exchange==2.3.0"],
        )

        with self.assertRaisesRegex(update_version.VersionError, "must use the shared-version metadata provider"):
            update_version.check_version_sync(self.root)

    def test_check_version_sync_rejects_an_unknown_internal_dependency(self) -> None:
        """Reject dynamic internal requirements that name no library distribution."""
        self._write_project(
            "isaacsim/foundation",
            "isaacsim-foundation",
            ["usd-exchange==2.3.0"],
            ["isaacsim-missing"],
        )

        with self.assertRaisesRegex(update_version.VersionError, "Unknown internal dependency"):
            update_version.check_version_sync(self.root)

    def test_update_version_rejects_a_noncanonical_version(self) -> None:
        """Test update version rejects a noncanonical version."""
        with self.assertRaisesRegex(update_version.VersionError, "Invalid canonical PEP 440"):
            update_version.update_version(self.root, "7.0-pre-alpha")


if __name__ == "__main__":
    unittest.main()
