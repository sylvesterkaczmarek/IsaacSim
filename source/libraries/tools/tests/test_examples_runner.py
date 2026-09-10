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

"""Test developer environment handling in the examples runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

EXAMPLES_RUNNER_PATH = Path(__file__).resolve().parents[3] / "examples" / "examples.py"
EXAMPLES_RUNNER_SPEC = importlib.util.spec_from_file_location("isaacsim_examples_runner", EXAMPLES_RUNNER_PATH)
if EXAMPLES_RUNNER_SPEC is None or EXAMPLES_RUNNER_SPEC.loader is None:
    raise RuntimeError(f"Cannot load examples runner from {EXAMPLES_RUNNER_PATH}")
examples_runner = importlib.util.module_from_spec(EXAMPLES_RUNNER_SPEC)
sys.modules[EXAMPLES_RUNNER_SPEC.name] = examples_runner
previous_bytecode_setting = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    EXAMPLES_RUNNER_SPEC.loader.exec_module(examples_runner)
finally:
    sys.dont_write_bytecode = previous_bytecode_setting

EXAMPLES_INTEGRATION_PATH = Path(__file__).resolve().parents[2] / "testing" / "examples" / "run_examples_test.py"
examples_integration = None
if sys.version_info >= (3, 11):
    EXAMPLES_INTEGRATION_SPEC = importlib.util.spec_from_file_location(
        "isaacsim_examples_integration", EXAMPLES_INTEGRATION_PATH
    )
    if EXAMPLES_INTEGRATION_SPEC is None or EXAMPLES_INTEGRATION_SPEC.loader is None:
        raise RuntimeError(f"Cannot load examples integration runner from {EXAMPLES_INTEGRATION_PATH}")
    examples_integration = importlib.util.module_from_spec(EXAMPLES_INTEGRATION_SPEC)
    sys.modules[EXAMPLES_INTEGRATION_SPEC.name] = examples_integration
    EXAMPLES_INTEGRATION_SPEC.loader.exec_module(examples_integration)


@unittest.skipIf(examples_integration is None, "Examples integration tests require Python 3.11 or newer")
class ExamplesIntegrationCleanupTests(unittest.TestCase):
    """Test retention of examples integration diagnostics and handoff artifacts."""

    def _run_phase(self, phase: str, *, error: Exception | None = None) -> bool:
        """Run a stubbed integration phase and report whether its test root remains.

        Args:
            phase: Example execution phase.
            error: Exception raised by the example.

        Returns:
            The resulting value.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            test_root = Path(temporary_directory) / "examples-test"
            arguments = SimpleNamespace(phase=phase, test_root=test_root)

            def run_phase(_arguments: object) -> None:
                test_root.mkdir()
                (test_root / "diagnostic.txt").touch()
                if error is not None:
                    raise error

            selected_function = {
                "all": "_run_full_integration",
                "build": "_prepare_prebuilt_examples",
                "test": "_test_prebuilt_examples",
            }[phase]
            with (
                mock.patch.object(examples_integration, "_parse_arguments", return_value=arguments),
                mock.patch.object(examples_integration, selected_function, side_effect=run_phase),
            ):
                if error is None:
                    self.assertEqual(examples_integration.main(), 0)
                else:
                    with self.assertRaises(type(error)):
                        examples_integration.main()
            return test_root.exists()

    def test_main_removes_successful_full_integration_tree(self) -> None:
        """Remove scratch output after the complete integration phase succeeds."""
        self.assertFalse(self._run_phase("all"))

    def test_main_removes_successful_prebuilt_test_tree(self) -> None:
        """Remove handed-off scratch output after the test phase succeeds."""
        self.assertFalse(self._run_phase("test"))

    def test_main_preserves_successful_build_phase_tree(self) -> None:
        """Keep artifacts that a later test phase must consume."""
        self.assertTrue(self._run_phase("build"))

    def test_main_preserves_failed_integration_tree(self) -> None:
        """Keep scratch output when a phase fails so it can be diagnosed."""
        self.assertTrue(self._run_phase("all", error=RuntimeError("failure")))


class DeveloperArtifactStateTests(unittest.TestCase):
    """Test rejection of absent, partial, and changed developer artifacts."""

    def _write_state(self, repository_root: Path, build_dir: Path, *, targets: list[str]) -> Path:
        """Write matching completed-build state and return its tracked input.

        Args:
            repository_root: Repository root directory.
            build_dir: Directory containing built examples.
            targets: Selected example targets.

        Returns:
            The resulting value.
        """
        source_root = repository_root / "source" / "libraries"
        source_root.mkdir(parents=True, exist_ok=True)
        manifest = repository_root / "pixi.toml"
        manifest.write_text("locked\n", encoding="utf-8")
        developer_environment = build_dir / examples_runner.DEVELOPER_ENVIRONMENT_DIRECTORY
        developer_environment.mkdir(exist_ok=True)
        (build_dir / "artifact-state.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "configuration": "Release",
                    "profile": "standard",
                    "dependency_profile": "locked",
                    "inputs": {"pixi.toml": hashlib.sha256(manifest.read_bytes()).hexdigest()},
                    "source_digest": examples_runner._library_source_digest(source_root),
                    "targets": targets,
                    "developer_environment": {
                        "path": examples_runner.DEVELOPER_ENVIRONMENT_DIRECTORY,
                        "components": ["isaacsim_common-runtime"],
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_validate_developer_artifact_state_accepts_matching_full_build(self) -> None:
        """Accept a full build whose locked inputs have not changed."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            build_dir = repository_root / "build"
            build_dir.mkdir()
            self._write_state(repository_root, build_dir, targets=["all"])

            actual_environment = examples_runner._validate_developer_artifact_state(
                repository_root, build_dir, "Release"
            )

            self.assertEqual(actual_environment, build_dir / examples_runner.DEVELOPER_ENVIRONMENT_DIRECTORY)

    def test_validate_developer_artifact_state_rejects_changed_input(self) -> None:
        """Require a rebuild after a locked manifest changes."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            build_dir = repository_root / "build"
            build_dir.mkdir()
            manifest = self._write_state(repository_root, build_dir, targets=["all"])
            manifest.write_text("changed\n", encoding="utf-8")

            with self.assertRaisesRegex(examples_runner.ExampleError, "changed since the build"):
                examples_runner._validate_developer_artifact_state(repository_root, build_dir, "Release")

    def test_validate_developer_artifact_state_rejects_partial_build(self) -> None:
        """Require a full library build before materializing developer examples."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            build_dir = repository_root / "build"
            build_dir.mkdir()
            self._write_state(repository_root, build_dir, targets=["isaacsim_common"])

            with self.assertRaisesRegex(examples_runner.ExampleError, "stale or partial"):
                examples_runner._validate_developer_artifact_state(repository_root, build_dir, "Release")

    def test_validate_developer_artifact_state_rejects_changed_library_sources(self) -> None:
        """Reject changed Python, C, and C++ library sources until the libraries are rebuilt."""
        for source_name in ("Module.py", "Module.c", "Module.cpp"):
            with self.subTest(source_name=source_name), tempfile.TemporaryDirectory() as temporary_directory:
                repository_root = Path(temporary_directory)
                build_dir = repository_root / "build"
                source_root = repository_root / "source" / "libraries"
                build_dir.mkdir()
                source_root.mkdir(parents=True)
                source_path = source_root / source_name
                source_path.write_text("original\n", encoding="utf-8")
                self._write_state(repository_root, build_dir, targets=["all"])
                source_path.write_text("changed\n", encoding="utf-8")

                with self.assertRaisesRegex(examples_runner.ExampleError, "sources changed since the build"):
                    examples_runner._validate_developer_artifact_state(repository_root, build_dir, "Release")

    def test_validate_developer_artifact_state_accepts_changed_example_sources(self) -> None:
        """Keep the environment valid after Python, C, and C++ example changes."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            build_dir = repository_root / "build"
            example_root = repository_root / "source" / "examples"
            build_dir.mkdir()
            example_root.mkdir(parents=True)
            source_paths = [example_root / name for name in ("main.py", "Main.c", "Main.cpp")]
            for source_path in source_paths:
                source_path.write_text("original\n", encoding="utf-8")
            self._write_state(repository_root, build_dir, targets=["all"])

            for source_path in source_paths:
                source_path.write_text("changed\n", encoding="utf-8")
                examples_runner._validate_developer_artifact_state(repository_root, build_dir, "Release")


class DeveloperEnvironmentTests(unittest.TestCase):
    """Test read-only use of the complete build-owned developer environment."""

    def test_create_developer_environment_uses_one_unified_prefix(self) -> None:
        """Expose native and Python surfaces without invoking installation."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prefix = root / "developer-environment"
            runtime_dependencies = root / "runtime-dependencies"
            for directory in (prefix / "bin", prefix / "lib", prefix / "python", runtime_dependencies):
                directory.mkdir(parents=True)
            executable = root / "python"
            executable.touch()
            developer_build = examples_runner._DeveloperBuild(
                library_build_dir=root,
                cmake=executable,
                python=executable,
                python_install_dir="python",
                python_runtime_dependencies=runtime_dependencies,
                developer_environment=prefix,
                generator="Ninja",
                generator_platform=None,
                generator_toolset=None,
                make_program=None,
            )

            with mock.patch.object(examples_runner.subprocess, "run") as run:
                first = examples_runner._create_developer_environment(developer_build)
                second = examples_runner._create_developer_environment(developer_build)

            self.assertEqual(first["CMAKE_PREFIX_PATH"], str(prefix))
            self.assertEqual(first["PYTHONPATH"].split(examples_runner.os.pathsep)[0], str(prefix / "python"))
            self.assertEqual(first, second)
            run.assert_not_called()

    def test_validate_developer_artifact_state_rejects_missing_environment(self) -> None:
        """Reject completed state whose build-owned environment was removed."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            build_dir = repository_root / "build"
            build_dir.mkdir()
            fixture = DeveloperArtifactStateTests()
            fixture._write_state(repository_root, build_dir, targets=["all"])
            (build_dir / examples_runner.DEVELOPER_ENVIRONMENT_DIRECTORY).rmdir()

            with self.assertRaisesRegex(examples_runner.ExampleError, "missing or redirected"):
                examples_runner._validate_developer_artifact_state(repository_root, build_dir, "Release")

    def test_validate_developer_artifact_state_rejects_redirected_environment(self) -> None:
        """Reject a developer environment redirected outside its build tree."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            build_dir = repository_root / "build"
            external_environment = repository_root / "external"
            build_dir.mkdir()
            external_environment.mkdir()
            fixture = DeveloperArtifactStateTests()
            fixture._write_state(repository_root, build_dir, targets=["all"])
            environment_path = build_dir / examples_runner.DEVELOPER_ENVIRONMENT_DIRECTORY
            environment_path.rmdir()
            try:
                environment_path.symlink_to(external_environment, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Directory symlinks are unavailable: {error}")

            with self.assertRaisesRegex(examples_runner.ExampleError, "missing or redirected"):
                examples_runner._validate_developer_artifact_state(repository_root, build_dir, "Release")


class DocumentationCatalogTests(unittest.TestCase):
    """Test documentation catalog generation through the real examples parser."""

    def setUp(self) -> None:
        """Handle setUp."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.library_catalog = self.root / "libraries.json"
        self.library_catalog.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "release_version": "7.0.0.dev0",
                    "distributions": [
                        {
                            "name": "isaacsim_common",
                            "complete": True,
                            "modules": [{"name": "isaacsim.common.logging"}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        """Handle tearDown."""
        self.temporary_directory.cleanup()

    def _write_example(
        self,
        relative_path: str,
        example_id: str,
        *,
        published: bool = True,
        run_path: str = "main.py",
    ) -> None:
        example_root = self.root / relative_path
        example_root.mkdir(parents=True)
        example_root.joinpath("README.md").write_text(f"# {example_id}\n", encoding="utf-8")
        entry_point = example_root / run_path
        entry_point.parent.mkdir(parents=True, exist_ok=True)
        entry_point.write_text("print('ok')\n", encoding="utf-8")
        example_root.joinpath("example.toml").write_text(
            f"""id = "{example_id}"
title = "{example_id}"
summary = "Catalog fixture."
owners = ["isaacsim.common.logging"]
topics = ["python"]
published = {str(published).lower()}

[build]
adapter = "none"

[run]
adapter = "python"
path = "{run_path}"

[requirements]
system = ["python"]

[[requirements.modules]]
name = "isaacsim_common"
surfaces = ["python"]
""",
            encoding="utf-8",
        )

    def test_catalog_is_deterministic_and_filters_unpublished_examples(self) -> None:
        """Test catalog is deterministic and filters unpublished examples."""
        self._write_example("libraries/demo/zeta", "zeta")
        self._write_example("libraries/demo/alpha", "alpha")
        self._write_example("libraries/demo/hidden", "hidden", published=False)

        first = examples_runner._build_catalog(self.root, self.library_catalog)
        second = examples_runner._build_catalog(self.root, self.library_catalog)

        self.assertEqual(first, second)
        self.assertEqual(first["release_version"], "7.0.0.dev0")
        self.assertEqual([entry["id"] for entry in first["examples"]], ["alpha", "zeta"])
        self.assertEqual(first["examples"][0]["languages"], ["python"])

    def test_catalog_preserves_nested_entry_point(self) -> None:
        """Test catalog preserves nested entry point."""
        self._write_example("libraries/demo/nested", "nested", run_path="scripts/main.py")

        catalog = examples_runner._build_catalog(self.root, self.library_catalog)

        self.assertEqual(catalog["examples"][0]["entry_point"], "scripts/main.py")

    def test_catalog_preserves_series_order_and_level(self) -> None:
        """Test catalog preserves series order and level."""
        self._write_example("series/learn/first", "learn.first")
        self._write_example("series/learn/second", "learn.second")
        series_root = self.root / "series" / "learn"
        series_root.joinpath("README.md").write_text("# Learn\n", encoding="utf-8")
        series_root.joinpath("series.toml").write_text(
            """id = "learn"
title = "Learn"
summary = "Ordered learning."

[[step]]
example = "learn.second"
level = "intermediate"

[[step]]
example = "learn.first"
level = "beginner"
""",
            encoding="utf-8",
        )

        catalog = examples_runner._build_catalog(self.root, self.library_catalog)

        self.assertEqual(catalog["series"][0]["steps"], ["learn.second", "learn.first"])
        examples = {entry["id"]: entry for entry in catalog["examples"]}
        self.assertEqual(examples["learn.second"]["series_position"], 1)
        self.assertEqual(examples["learn.second"]["level"], "intermediate")

    def test_catalog_rejects_unknown_library_cross_references(self) -> None:
        """Test catalog rejects unknown library cross references."""
        self._write_example("libraries/demo/invalid", "invalid")
        manifest = self.root / "libraries" / "demo" / "invalid" / "example.toml"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace("isaacsim_common", "missing_distribution"),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(examples_runner.ExampleError, "unknown distributions"):
            examples_runner._build_catalog(self.root, self.library_catalog)

    def test_catalog_rejects_unknown_library_owner(self) -> None:
        """Test catalog rejects unknown library owner."""
        self._write_example("libraries/demo/invalid", "invalid")
        manifest = self.root / "libraries" / "demo" / "invalid" / "example.toml"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace("isaacsim.common.logging", "isaacsim.missing.module"),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(examples_runner.ExampleError, "unknown public API owners"):
            examples_runner._build_catalog(self.root, self.library_catalog)

    def test_catalog_rejects_references_to_incomplete_distributions(self) -> None:
        """Test catalog rejects references to incomplete distributions."""
        self._write_example("libraries/demo/invalid", "invalid")
        catalog = json.loads(self.library_catalog.read_text(encoding="utf-8"))
        catalog["distributions"][0]["complete"] = False
        self.library_catalog.write_text(json.dumps(catalog), encoding="utf-8")

        with self.assertRaisesRegex(examples_runner.ExampleError, "unknown distributions"):
            examples_runner._build_catalog(self.root, self.library_catalog)

    def test_catalog_rejects_unsupported_library_catalog_schema(self) -> None:
        """Test catalog rejects unsupported library catalog schema."""
        self._write_example("libraries/demo/invalid", "invalid")
        catalog = json.loads(self.library_catalog.read_text(encoding="utf-8"))
        catalog["schema_version"] = 2
        self.library_catalog.write_text(json.dumps(catalog), encoding="utf-8")

        with self.assertRaisesRegex(examples_runner.ExampleError, "Unsupported library documentation catalog schema"):
            examples_runner._build_catalog(self.root, self.library_catalog)


if __name__ == "__main__":
    unittest.main()
