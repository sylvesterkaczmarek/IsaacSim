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

"""Materialize and test the examples through their public repository runner."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import textwrap
import tomllib
from pathlib import Path

EXAMPLES_TEST_PHASE_VARIABLE = "ISAACSIM_EXAMPLES_TEST_PHASE"
TRANSIENT_SOURCE_DIRECTORIES = ("__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache")


def _copy_authored_tree(source: Path, destination: Path) -> None:
    """Copy authored example files without ignored tool caches.

    Args:
        source: Source path.
        destination: Destination path.
    """
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(*TRANSIENT_SOURCE_DIRECTORIES),
    )


def _parse_arguments() -> argparse.Namespace:
    """Parse CTest-provided integration settings.

    Returns:
        Parsed integration settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", type=Path, required=True)
    parser.add_argument("--library-build-dir", type=Path, required=True)
    parser.add_argument("--examples-dir", type=Path, required=True)
    parser.add_argument("--system-requirements-file", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--python-install-dir", required=True)
    parser.add_argument("--python-runtime-deps-dir", type=Path, required=True)
    parser.add_argument("--target-deps-dir", type=Path, required=True)
    parser.add_argument("--readelf", default="")
    parser.add_argument("--package-group", action="append", default=[])
    parser.add_argument("--generator", required=True)
    parser.add_argument("--generator-platform", default="")
    parser.add_argument("--generator-toolset", default="")
    parser.add_argument("--make-program", default="")
    parser.add_argument("--config", default="Release")
    parser.add_argument(
        "--phase",
        choices=("all", "build", "test"),
        default=os.environ.get(EXAMPLES_TEST_PHASE_VARIABLE, "all"),
        help=(
            "Run the complete integration test, prepare and build a portable test tree, or test an existing tree. "
            f"Defaults to ${EXAMPLES_TEST_PHASE_VARIABLE}, then 'all'."
        ),
    )
    return parser.parse_args()


def _run_checked(
    description: str,
    command: list[str],
    environment: dict[str, str] | None = None,
    working_directory: Path | None = None,
) -> str:
    """Run a command and return stdout or raise with captured diagnostics.

    Args:
        description: User-facing operation description.
        command: Command and arguments.
        environment: Optional process environment.
        working_directory: Optional command working directory.

    Returns:
        Captured standard output.
    """
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        cwd=working_directory,
        env=environment,
        text=True,
    )
    if result.returncode != 0:
        rendered_command = subprocess.list2cmdline(command)
        raise RuntimeError(
            f"{description} failed ({result.returncode})\nCommand: {rendered_command}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def _run_expect_failure(
    description: str,
    command: list[str],
    expected_text: str,
    environment: dict[str, str] | None = None,
    working_directory: Path | None = None,
) -> None:
    """Run a command and require a failure with an expected diagnostic.

    Args:
        description: User-facing operation description.
        command: Command and arguments.
        expected_text: Diagnostic text required in captured output.
        environment: Optional process environment.
        working_directory: Optional command working directory.
    """
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        cwd=working_directory,
        env=environment,
        text=True,
    )
    output = result.stdout + result.stderr
    if result.returncode == 0 or expected_text not in output:
        rendered_command = subprocess.list2cmdline(command)
        raise RuntimeError(
            f"{description} did not fail as expected\nCommand: {rendered_command}\n"
            f"Expected diagnostic: {expected_text}\n{output}"
        )


def _read_example_paths(examples_dir: Path) -> tuple[str, ...]:
    """Discover published example paths needed for materialization checks.

    Args:
        examples_dir: Root of the examples collection.

    Returns:
        Published relative example paths in deterministic path order.
    """
    paths: list[str] = []
    for manifest_path in sorted(examples_dir.rglob("example.toml")):
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        published = manifest.get("published", True)
        if type(published) is not bool:
            raise ValueError(f"{manifest_path}.published must be a boolean")
        if published:
            paths.append(manifest_path.parent.relative_to(examples_dir).as_posix())
    if not paths:
        raise ValueError(f"{examples_dir} has no published examples")
    return tuple(paths)


def _read_required_surfaces(examples_dir: Path) -> dict[str, set[str]]:
    """Read the direct module surfaces declared by the source manifests.

    Args:
        examples_dir: Root of the examples collection.

    Returns:
        Required surfaces indexed by module distribution.
    """
    requirements: dict[str, set[str]] = {}
    for relative_path in _read_example_paths(examples_dir):
        example_root = examples_dir / relative_path
        manifest_path = example_root / "example.toml"
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        requirement_table = manifest.get("requirements")
        modules = requirement_table.get("modules") if isinstance(requirement_table, dict) else None
        if not isinstance(modules, list) or not modules:
            raise ValueError(f"{manifest_path} has no module requirements")
        for module in modules:
            if not isinstance(module, dict):
                raise ValueError(f"{manifest_path} has an invalid module requirement")
            name = module.get("name")
            surfaces = module.get("surfaces")
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(surfaces, list)
                or not surfaces
                or any(not isinstance(surface, str) or surface not in {"native_sdk", "python"} for surface in surfaces)
            ):
                raise ValueError(f"{manifest_path} has an invalid module requirement")
            requirements.setdefault(name, set()).update(surfaces)
    return requirements


def _install_component(arguments: argparse.Namespace, prefix: Path, component: str) -> None:
    """Install one staged module component into an isolated prefix.

    Args:
        arguments: CTest-provided integration settings.
        prefix: Isolated installation prefix.
        component: CMake install component.
    """
    _run_checked(
        f"Install {component}",
        [
            str(arguments.cmake),
            "--install",
            str(arguments.library_build_dir),
            "--prefix",
            str(prefix),
            "--component",
            component,
            "--config",
            arguments.config,
        ],
    )


def _install_declared_requirements(
    arguments: argparse.Namespace,
    examples_dir: Path,
    sdk_prefix: Path,
    python_prefix: Path,
) -> None:
    """Install exactly the module surfaces declared by published examples.

    Args:
        arguments: CTest-provided integration settings.
        examples_dir: Root of the examples collection.
        sdk_prefix: Isolated native SDK prefix.
        python_prefix: Isolated Python distribution prefix.
    """
    requirements = _read_required_surfaces(examples_dir)
    unavailable_groups = requirements.keys() - set(arguments.package_group)
    if unavailable_groups:
        raise ValueError(f"Examples require unavailable module groups: {', '.join(sorted(unavailable_groups))}")

    for module_name, surfaces in sorted(requirements.items()):
        _install_component(arguments, sdk_prefix, f"{module_name}-runtime")
        if "native_sdk" in surfaces:
            _install_component(arguments, sdk_prefix, f"{module_name}-development")
        if "python" in surfaces:
            _install_component(arguments, python_prefix, f"{module_name}-python")


def _reject_installed_build_paths(
    arguments: argparse.Namespace,
    sdk_prefix: Path,
    python_prefix: Path,
) -> None:
    """Reject Isaac Sim binaries that encode the build dependency directory.

    Args:
        arguments: CTest-provided integration settings.
        sdk_prefix: Isolated native SDK prefix.
        python_prefix: Isolated Python distribution prefix.
    """
    if os.name == "nt" or not arguments.readelf:
        return

    forbidden_prefix = str(arguments.target_deps_dir.resolve())
    candidates = [
        *sdk_prefix.glob("lib/libisaacsim*.so*"),
        *python_prefix.rglob("libisaacsim*.so*"),
        *python_prefix.rglob("_bindings*.so"),
    ]
    checked_files: set[Path] = set()
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in checked_files or not resolved_candidate.is_file():
            continue
        checked_files.add(resolved_candidate)
        dynamic_section = _run_checked(
            f"Inspect installed binary {candidate}",
            [arguments.readelf, "-d", str(resolved_candidate)],
        )
        for line in dynamic_section.splitlines():
            if ("RPATH" in line or "RUNPATH" in line) and forbidden_prefix in line:
                raise RuntimeError(f"Installed binary contains a build dependency RUNPATH: {candidate}")


def _runner_arguments(arguments: argparse.Namespace, build_root: Path) -> list[str]:
    """Create the runner's global CMake arguments.

    Args:
        arguments: CTest-provided integration settings.
        build_root: Isolated example build root.

    Returns:
        Global runner arguments.
    """
    result = [
        "--build-root",
        str(build_root),
        "--cmake",
        str(arguments.cmake),
        "--config",
        arguments.config,
        "--generator",
        arguments.generator,
    ]
    if arguments.generator_platform:
        result.extend(["--generator-platform", arguments.generator_platform])
    if arguments.generator_toolset:
        result.extend(["--generator-toolset", arguments.generator_toolset])
    if arguments.make_program:
        result.extend(["--make-program", arguments.make_program])
    return result


def _runtime_environment(
    arguments: argparse.Namespace,
    sdk_prefix: Path,
    python_prefix: Path,
) -> dict[str, str]:
    """Create the installed-component environment consumed by examples.

    Args:
        arguments: CTest-provided integration settings.
        sdk_prefix: Installed native SDK prefix.
        python_prefix: Installed Python distribution prefix.

    Returns:
        Runtime environment for the materialized examples.
    """
    environment = os.environ.copy()
    environment["CMAKE_PREFIX_PATH"] = str(sdk_prefix)
    environment["PATH"] = os.pathsep.join((str(sdk_prefix / "bin"), environment.get("PATH", "")))
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(python_prefix / arguments.python_install_dir),
            str(arguments.python_runtime_deps_dir),
        )
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _validate_materialized_roots(examples_dir: Path) -> None:
    """Validate each published leaf as an independent installer payload.

    Args:
        examples_dir: Root of the materialized examples collection.
    """
    for relative_path in _read_example_paths(examples_dir):
        example_root = examples_dir / relative_path
        if not example_root.is_dir():
            raise ValueError(f"Example root does not exist: {example_root}")
        readme = example_root / "README.md"
        if not readme.is_file():
            raise ValueError(f"Example root has no README.md: {example_root}")
        readme_text = readme.read_text(encoding="utf-8")
        for forbidden_text in ("source/examples", "_build/examples", "../../README.md"):
            if forbidden_text in readme_text:
                raise ValueError(f"{readme} contains collection-only text: {forbidden_text}")


def _materialize_selected_workspace(
    source_examples: Path,
    system_requirements_file: Path,
    destination_examples: Path,
    example_paths: tuple[str, ...],
    series_paths: tuple[str, ...] = (),
) -> None:
    """Create the minimal selected workspace defined by the Installer SDD.

    Args:
        source_examples: Validated source examples collection.
        system_requirements_file: Release-wide system requirements to materialize.
        destination_examples: Destination `source/examples` directory.
        example_paths: Example roots to include in the selected workspace.
        series_paths: Complete series roots to include in the selected workspace.
    """
    destination_examples.mkdir(parents=True)
    destination_libraries = destination_examples.parent / "libraries"
    destination_libraries.mkdir()
    shutil.copy2(system_requirements_file, destination_libraries / system_requirements_file.name)
    for filename in ("README.md", "examples.py"):
        shutil.copy2(source_examples / filename, destination_examples / filename)
    for relative_path in series_paths:
        _copy_authored_tree(source_examples / relative_path, destination_examples / relative_path)
    for relative_path in example_paths:
        destination = destination_examples / relative_path
        if not destination.exists():
            _copy_authored_tree(source_examples / relative_path, destination)


def _add_series_fixture(examples_dir: Path) -> None:
    """Add a materialized-only series that exercises ordering and level validation.

    Args:
        examples_dir: Root of the materialized examples collection.
    """
    series_root = examples_dir / "series" / "runner_smoke"
    step_root = series_root / "introduction"
    no_test_root = series_root / "independent_practice"
    step_root.mkdir(parents=True)
    no_test_root.mkdir()
    (series_root / "README.md").write_text("# Runner smoke series\n", encoding="utf-8")
    (series_root / "series.toml").write_text(
        textwrap.dedent("""\
            id = "runner_smoke"
            title = "Runner Smoke"
            summary = "Exercise the series manifest contract."

            [[step]]
            example = "runner_smoke.introduction"
            level = "beginner"

            [[step]]
            example = "runner_smoke.independent_practice"
            level = "expert"
            """),
        encoding="utf-8",
    )
    (step_root / "README.md").write_text("# Runner smoke introduction\n", encoding="utf-8")
    (step_root / "main.py").write_text(
        textwrap.dedent("""\
            import os
            import sys

            valid_arguments = (
                ["--include", "normal", "--include", "normal", ""],
                ["--include", "first", "--include", "second", ""],
            )
            if sys.argv[1:] not in valid_arguments:
                raise SystemExit(f"Unexpected arguments: {sys.argv[1:]}")
            if os.environ.get("ISAACSIM_RUNNER_SMOKE") != "enabled":
                raise SystemExit("Missing test environment overlay")
            print("Runner smoke series passed.")
            """),
        encoding="utf-8",
    )
    (step_root / "example.toml").write_text(
        textwrap.dedent("""\
            id = "runner_smoke.introduction"
            title = "Runner Smoke Introduction"
            summary = "Exercise a co-located series step."
            owners = ["isaacsim.common.logging"]
            topics = ["testing"]

            [build]
            adapter = "none"

            [run]
            adapter = "python"
            path = "main.py"
            arguments = ["--include", "normal", "--include", "normal", ""]

            [requirements]
            system = ["python"]

            [[requirements.modules]]
            name = "isaacsim_common"
            surfaces = ["python"]

            [[test]]
            name = "inherited"
            stdout_contains = ["Runner smoke series passed."]

            [test.environment]
            ISAACSIM_RUNNER_SMOKE = "enabled"

            [[test]]
            name = "override"
            arguments = ["--include", "first", "--include", "second", ""]
            stdout_contains = ["Runner smoke series passed."]

            [test.environment]
            ISAACSIM_RUNNER_SMOKE = "enabled"
            """),
        encoding="utf-8",
    )
    (no_test_root / "README.md").write_text("# Runner smoke independent practice\n", encoding="utf-8")
    (no_test_root / "main.py").write_text(
        textwrap.dedent("""\
            import sys

            print(f"Arguments: {sys.argv[1:]!r}")
            """),
        encoding="utf-8",
    )
    (no_test_root / "example.toml").write_text(
        textwrap.dedent("""\
            id = "runner_smoke.independent_practice"
            title = "Runner Smoke Independent Practice"
            summary = "Exercise a series step without automated test configuration."
            owners = ["isaacsim.common.logging"]
            topics = ["testing"]

            [build]
            adapter = "none"

            [run]
            adapter = "python"
            path = "main.py"

            [requirements]
            system = ["python"]

            [[requirements.modules]]
            name = "isaacsim_common"
            surfaces = ["python"]
            """),
        encoding="utf-8",
    )


def _add_failure_fixture(examples_dir: Path) -> None:
    """Add tests that intentionally fail through assertions and timeout handling.

    Args:
        examples_dir: Root of the materialized examples collection.
    """
    example_root = examples_dir / "libraries" / "runner_failure"
    example_root.mkdir(parents=True)
    (example_root / "README.md").write_text("# Runner failure fixture\n", encoding="utf-8")
    (example_root / "main.py").write_text(
        textwrap.dedent("""\
            import subprocess
            import sys
            import time

            if sys.argv[1:] == ["--spawn-child"]:
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "import signal,time;"
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                        "time.sleep(30)",
                    ]
                )
                time.sleep(30)
            print("Actual output.")
            """),
        encoding="utf-8",
    )
    (example_root / "example.toml").write_text(
        textwrap.dedent("""\
            id = "runner_failure"
            title = "Runner Failure"
            summary = "Exercise runner failure reporting."
            owners = ["isaacsim.common.logging"]
            topics = ["testing"]

            [build]
            adapter = "none"

            [run]
            adapter = "python"
            path = "main.py"

            [requirements]
            system = ["python"]

            [[requirements.modules]]
            name = "isaacsim_common"
            surfaces = ["python"]

            [[test]]
            name = "assertion_failure"
            stdout_contains = ["Missing output."]

            [[test]]
            name = "timeout"
            arguments = ["--spawn-child"]
            timeout_seconds = 1
            """),
        encoding="utf-8",
    )

    unpublished_root = examples_dir / "libraries" / "runner_unpublished"
    unpublished_root.mkdir()
    (unpublished_root / "example.toml").write_text("published = false\n", encoding="utf-8")


def _exercise_invalid_source_rejections(
    arguments: argparse.Namespace,
    environment: dict[str, str],
) -> None:
    """Exercise fail-closed validation with isolated malformed source copies.

    Args:
        arguments: CTest-provided integration settings.
        environment: Runtime environment for runner subprocesses.
    """
    invalid_root = arguments.test_root / "invalid-source"
    cases: list[tuple[str, str, str]] = [
        ("published", 'id = "hello_world.c"', 'published = "yes"\nid = "hello_world.c"'),
        ("target", 'targets = ["hello_world_c"]', 'targets = ["--clean-first"]'),
        ("requirement", 'surfaces = ["native_sdk"]', 'surfaces = ["python"]'),
    ]
    for name, old_text, new_text in cases:
        fixture = invalid_root / name
        _copy_authored_tree(arguments.examples_dir, fixture)
        path = fixture / "libraries" / "isaacsim_common" / "logging" / "hello_world" / "c" / "example.toml"
        text = path.read_text(encoding="utf-8")
        if old_text not in text:
            raise RuntimeError(f"Cannot prepare {name} validation fixture")
        path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        expected_text_by_name = {
            "published": "published must be a boolean",
            "target": "targets contains an invalid CMake target",
            "requirement": "CMake adapter but declares no native_sdk",
        }
        _run_expect_failure(
            f"Reject invalid {name} source",
            [
                str(arguments.python),
                str(fixture / "examples.py"),
                "--build-root",
                str(arguments.test_root / "invalid-build" / name),
                "build",
                "hello_world.python",
            ],
            expected_text_by_name[name],
            environment,
            arguments.test_root,
        )

    environment_fixture = invalid_root / "environment"
    _copy_authored_tree(arguments.examples_dir, environment_fixture)
    environment_manifest = (
        environment_fixture / "libraries" / "isaacsim_common" / "logging" / "hello_world" / "python" / "example.toml"
    )
    environment_manifest.write_text(
        environment_manifest.read_text(encoding="utf-8") + '\n[test.environment]\n"BAD=KEY" = "value"\n',
        encoding="utf-8",
    )
    _run_expect_failure(
        "Reject a non-portable environment variable name",
        [
            str(arguments.python),
            str(environment_fixture / "examples.py"),
            "--build-root",
            str(arguments.test_root / "invalid-build" / "environment"),
            "test",
            "hello_world.python",
        ],
        "environment must contain portable variable names",
        environment,
        arguments.test_root,
    )

    generated_fixture = invalid_root / "generated"
    _copy_authored_tree(arguments.examples_dir, generated_fixture)
    (generated_fixture / ".ruff_cache").mkdir()
    _run_expect_failure(
        "Reject generated source output",
        [
            str(arguments.python),
            str(generated_fixture / "examples.py"),
            "--build-root",
            str(arguments.test_root / "invalid-build" / "generated"),
            "build",
            "hello_world.python",
        ],
        "Example source contains generated output",
        environment,
        arguments.test_root,
    )

    if os.name != "nt":
        symlink_fixture = invalid_root / "symlink"
        _copy_authored_tree(arguments.examples_dir, symlink_fixture)
        symlink_path = symlink_fixture / "linked_main.py"
        symlink_path.symlink_to(
            symlink_fixture / "libraries" / "isaacsim_common" / "logging" / "hello_world" / "python" / "main.py"
        )
        _run_expect_failure(
            "Reject a source symlink after materialization",
            [
                str(arguments.python),
                str(symlink_fixture / "examples.py"),
                "--build-root",
                str(arguments.test_root / "invalid-build" / "symlink"),
                "build",
                "hello_world.python",
            ],
            "Example source must not contain symlinks",
            environment,
            arguments.test_root,
        )


def _prepare_prebuilt_examples(arguments: argparse.Namespace) -> None:
    """Install, materialize, and build the published examples for a later test job.

    Args:
        arguments: CTest-provided integration settings.
    """
    if arguments.test_root.exists():
        shutil.rmtree(arguments.test_root)

    materialized_repository = arguments.test_root / "materialized"
    materialized_examples = materialized_repository / "source" / "examples"
    materialized_libraries = materialized_repository / "source" / "libraries"
    sdk_prefix = arguments.test_root / "sdk"
    python_prefix = arguments.test_root / "python-package"
    build_root = materialized_repository / "_build" / "examples"

    _install_declared_requirements(
        arguments,
        arguments.examples_dir,
        sdk_prefix,
        python_prefix,
    )
    _reject_installed_build_paths(arguments, sdk_prefix, python_prefix)
    _copy_authored_tree(arguments.examples_dir, materialized_examples)
    materialized_libraries.mkdir()
    shutil.copy2(
        arguments.system_requirements_file,
        materialized_libraries / arguments.system_requirements_file.name,
    )
    _validate_materialized_roots(materialized_examples)

    environment = _runtime_environment(arguments, sdk_prefix, python_prefix)
    _run_checked(
        "Build all materialized examples",
        [
            str(arguments.python),
            str(materialized_examples / "examples.py"),
            *_runner_arguments(arguments, build_root),
            "build",
        ],
        environment,
        materialized_repository,
    )


def _test_prebuilt_examples(arguments: argparse.Namespace) -> None:
    """Test the published examples prepared by a compiler-equipped build job.

    Args:
        arguments: CTest-provided integration settings.
    """
    materialized_repository = arguments.test_root / "materialized"
    materialized_examples = materialized_repository / "source" / "examples"
    runner = materialized_examples / "examples.py"
    sdk_prefix = arguments.test_root / "sdk"
    python_prefix = arguments.test_root / "python-package"
    build_root = materialized_repository / "_build" / "examples"
    if not build_root.is_dir():
        raise RuntimeError(
            f"The prebuilt examples test tree is incomplete; run the build phase first. Missing: {build_root}"
        )

    for path in (materialized_repository / "source", sdk_prefix, python_prefix):
        if path.exists():
            shutil.rmtree(path)
    _copy_authored_tree(arguments.examples_dir, materialized_examples)
    materialized_libraries = materialized_repository / "source" / "libraries"
    materialized_libraries.mkdir()
    shutil.copy2(
        arguments.system_requirements_file,
        materialized_libraries / arguments.system_requirements_file.name,
    )
    _install_declared_requirements(
        arguments,
        materialized_examples,
        sdk_prefix,
        python_prefix,
    )
    _reject_installed_build_paths(arguments, sdk_prefix, python_prefix)

    environment = _runtime_environment(arguments, sdk_prefix, python_prefix)
    _run_checked(
        "Test all prebuilt materialized examples",
        [
            str(arguments.python),
            str(runner),
            *_runner_arguments(arguments, build_root),
            "test",
            "--no-build",
        ],
        environment,
        materialized_repository,
    )


def _run_full_integration(arguments: argparse.Namespace) -> None:
    """Validate and execute every materialized example in one job.

    Args:
        arguments: CTest-provided integration settings.
    """
    if arguments.test_root.exists():
        shutil.rmtree(arguments.test_root)
    test_root_libraries = arguments.test_root / "source" / "libraries"
    test_root_libraries.mkdir(parents=True)
    shutil.copy2(
        arguments.system_requirements_file,
        test_root_libraries / arguments.system_requirements_file.name,
    )
    sdk_prefix = arguments.test_root / "sdk"
    python_prefix = arguments.test_root / "python-package"
    materialized_repository = arguments.test_root / "materialized"
    materialized_examples = materialized_repository / "source" / "examples"
    build_root = materialized_repository / "_build" / "examples"
    _install_declared_requirements(
        arguments,
        arguments.examples_dir,
        sdk_prefix,
        python_prefix,
    )
    _reject_installed_build_paths(arguments, sdk_prefix, python_prefix)

    environment = _runtime_environment(arguments, sdk_prefix, python_prefix)
    selected_repository = arguments.test_root / "selected-example"
    selected_examples = selected_repository / "source" / "examples"
    _materialize_selected_workspace(
        arguments.examples_dir,
        arguments.system_requirements_file,
        selected_examples,
        ("libraries/isaacsim_common/logging/hello_world/python",),
    )
    selected_runner = selected_examples / "examples.py"
    selected_sdk_prefix = arguments.test_root / "selected-example-sdk"
    selected_python_prefix = arguments.test_root / "selected-example-python-package"
    _install_declared_requirements(
        arguments,
        selected_examples,
        selected_sdk_prefix,
        selected_python_prefix,
    )
    selected_environment = _runtime_environment(arguments, selected_sdk_prefix, selected_python_prefix)
    _run_checked(
        "Test a minimal installer-selected example workspace",
        [
            str(arguments.python),
            str(selected_runner),
            *_runner_arguments(arguments, selected_repository / "_build" / "examples"),
            "test",
        ],
        selected_environment,
        selected_repository,
    )

    _copy_authored_tree(arguments.examples_dir, materialized_examples)
    materialized_libraries = materialized_repository / "source" / "libraries"
    materialized_libraries.mkdir()
    shutil.copy2(
        arguments.system_requirements_file,
        materialized_libraries / arguments.system_requirements_file.name,
    )
    runner = materialized_examples / "examples.py"
    common_arguments = [
        str(arguments.python),
        str(runner),
        *_runner_arguments(arguments, build_root),
    ]
    _run_checked(
        "Validate the untouched examples collection and test its examples",
        [*common_arguments, "test"],
        environment,
        materialized_repository,
    )
    _validate_materialized_roots(materialized_examples)
    _add_series_fixture(materialized_examples)
    selected_series_repository = arguments.test_root / "selected-series"
    selected_series_examples = selected_series_repository / "source" / "examples"
    _materialize_selected_workspace(
        materialized_examples,
        arguments.system_requirements_file,
        selected_series_examples,
        (
            "series/runner_smoke/introduction",
            "series/runner_smoke/independent_practice",
        ),
        ("series/runner_smoke",),
    )
    selected_series_sdk_prefix = arguments.test_root / "selected-series-sdk"
    selected_series_python_prefix = arguments.test_root / "selected-series-python-package"
    _install_declared_requirements(
        arguments,
        selected_series_examples,
        selected_series_sdk_prefix,
        selected_series_python_prefix,
    )
    selected_series_environment = _runtime_environment(
        arguments,
        selected_series_sdk_prefix,
        selected_series_python_prefix,
    )
    _run_checked(
        "Test a minimal installer-selected series workspace",
        [
            str(arguments.python),
            str(selected_series_examples / "examples.py"),
            *_runner_arguments(arguments, selected_series_repository / "_build" / "examples"),
            "test",
        ],
        selected_series_environment,
        selected_series_repository,
    )
    all_test_output = _run_checked(
        "Build and test all examples",
        [*common_arguments, "test"],
        environment,
        materialized_repository,
    )
    if "Examples environment: installed" not in all_test_output:
        raise RuntimeError(f"Materialized examples did not select installed mode\n{all_test_output}")
    if "runner_smoke.independent_practice: skipped" not in all_test_output:
        raise RuntimeError(f"Default-all test did not report the unconfigured example as skipped\n{all_test_output}")
    _run_checked(
        "Run one named test configuration with an explicit installed environment",
        [*common_arguments, "test", "--installed", "runner_smoke.introduction:override"],
        environment,
        materialized_repository,
    )
    _run_expect_failure(
        "Reject developer mode outside a source checkout",
        [*common_arguments, "run", "--dev", "runner_smoke.independent_practice"],
        "--dev requires examples.py to be run from an Isaac Sim source checkout",
        environment,
        materialized_repository,
    )
    _run_expect_failure(
        "Reject conflicting environment overrides",
        [*common_arguments, "--dev", "run", "--installed", "runner_smoke.independent_practice"],
        "--dev and --installed cannot be used together",
        environment,
        materialized_repository,
    )
    passthrough_output = _run_checked(
        "Run one example with argument passthrough",
        [
            *common_arguments,
            "run",
            "runner_smoke.independent_practice",
            "--",
            "--integration-smoke",
            "two words",
        ],
        environment,
        materialized_repository,
    )
    if "Arguments: ['--integration-smoke', 'two words']" not in passthrough_output:
        raise RuntimeError(f"Run did not preserve passthrough arguments\n{passthrough_output}")
    script_path_output = _run_checked(
        "Run one example by its declared Python script path",
        [
            *common_arguments,
            "run",
            "series/runner_smoke/independent_practice/main.py",
            "--",
            "--integration-smoke",
            "two words",
        ],
        environment,
        materialized_repository,
    )
    if "Arguments: ['--integration-smoke', 'two words']" not in script_path_output:
        raise RuntimeError(f"Script-path run did not preserve passthrough arguments\n{script_path_output}")
    _add_failure_fixture(materialized_examples)
    _run_expect_failure(
        "Report a failed output assertion",
        [*common_arguments, "test", "runner_failure:assertion_failure"],
        "stdout does not contain: Missing output.",
        environment,
        materialized_repository,
    )
    _run_expect_failure(
        "Terminate a timed-out test",
        [*common_arguments, "test", "runner_failure:timeout"],
        "timed out after 1 seconds",
        environment,
        materialized_repository,
    )
    _run_expect_failure(
        "Exclude an unpublished example",
        [*common_arguments, "test", "runner_unpublished"],
        "Unknown example ID: runner_unpublished",
        environment,
        materialized_repository,
    )
    _run_expect_failure(
        "Reject a build root inside source/examples",
        [
            str(arguments.python),
            str(runner),
            "--build-root",
            str(materialized_examples / "generated_output"),
            "build",
            "hello_world.python",
        ],
        "Build root must remain outside source/examples",
        environment,
        materialized_repository,
    )
    _exercise_invalid_source_rejections(arguments, environment)


def main() -> int:
    """Run the selected examples integration phase.

    Returns:
        Process exit code.
    """
    arguments = _parse_arguments()
    if arguments.phase == "build":
        _prepare_prebuilt_examples(arguments)
    elif arguments.phase == "test":
        _test_prebuilt_examples(arguments)
    else:
        _run_full_integration(arguments)
    if arguments.phase != "build" and arguments.test_root.exists():
        shutil.rmtree(arguments.test_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
