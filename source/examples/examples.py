# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build, run, and test published Isaac Sim examples."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # The developer bootstrap may initially run under Python 3.10.
    tomllib = None

DEFAULT_TEST_TIMEOUT_SECONDS = 60
PROCESS_TERMINATION_TIMEOUT_SECONDS = 5
DEVELOPER_ENVIRONMENT_VARIABLE = "ISAACSIM_EXAMPLES_DEVELOPER_ENVIRONMENT"
DEVELOPER_BOOTSTRAP_PREFIX = "bootstrap:"
DEVELOPER_ENVIRONMENT_DIRECTORY = "developer-environment"
DEVELOPER_ENVIRONMENT_LOCK_SUFFIX = "developer-environment.lock"
EXAMPLE_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*")
NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
TARGET_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*")
ENVIRONMENT_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ALLOWED_EXAMPLE_GROUPS = frozenset({"libraries", "series"})
ALLOWED_LEVELS = frozenset({"beginner", "intermediate", "expert"})
ALLOWED_MODULE_SURFACES = frozenset({"native_sdk", "python"})
SYSTEM_REQUIREMENT_NAMES = frozenset({"c", "cmake", "cpp", "python", "python_development"})
SYSTEM_REQUIREMENTS_FIELDS = frozenset(
    {"c_standard", "cmake_minimum_version", "cmake_policy_maximum_version", "cpp_standard", "python_version"}
)
VERSION_PATTERN = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))?")
GENERATED_DIRECTORY_NAMES = frozenset(
    {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "_build", "build", "CMakeFiles", "dist"}
)
GENERATED_FILE_NAMES = frozenset(
    {
        "CMakeCache.txt",
        "Makefile",
        ".ninja_deps",
        ".ninja_log",
        "build.ninja",
        "cmake_install.cmake",
        "compile_commands.json",
    }
)
GENERATED_FILE_SUFFIXES = frozenset({".a", ".dll", ".dylib", ".exe", ".lib", ".o", ".obj", ".pyc", ".pyd", ".so"})


class ExampleError(RuntimeError):
    """Error raised for an invalid example or failed example operation."""


@dataclass(frozen=True)
class _BuildConfiguration:
    """Structured build configuration for an example."""

    adapter: str
    targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RunConfiguration:
    """Structured run configuration for an example."""

    adapter: str
    target: str | None = None
    path: Path | None = None
    arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class _TestConfiguration:
    """Test overlay for an example's normal run configuration."""

    name: str
    arguments: tuple[str, ...] | None
    timeout_seconds: int
    expected_exit_code: int
    stdout_contains: tuple[str, ...]
    stderr_contains: tuple[str, ...]
    environment: dict[str, str]


@dataclass(frozen=True)
class _ModuleRequirement:
    """One direct Isaac Sim module distribution requirement."""

    name: str
    surfaces: tuple[str, ...]


@dataclass(frozen=True)
class _Requirements:
    """Validated module and system requirements."""

    modules: tuple[_ModuleRequirement, ...]
    system: tuple[str, ...]


@dataclass(frozen=True)
class _SystemRequirements:
    """Release-wide system toolchain requirements."""

    cmake_minimum_version: str
    cmake_policy_maximum_version: str
    c_standard: int
    cpp_standard: int
    python_version: str


@dataclass(frozen=True)
class _Example:
    """Validated example definition."""

    id: str
    title: str
    summary: str
    owners: tuple[str, ...]
    categories: tuple[str, ...]
    topics: tuple[str, ...]
    root: Path
    build: _BuildConfiguration
    run: _RunConfiguration
    requirements: _Requirements
    tests: tuple[_TestConfiguration, ...]


@dataclass(frozen=True)
class _SeriesStep:
    """One validated step in an ordered example series."""

    example_id: str
    level: str


@dataclass(frozen=True)
class _Series:
    """Validated ordered example series."""

    id: str
    title: str
    summary: str
    root: Path
    steps: tuple[_SeriesStep, ...]


@dataclass(frozen=True)
class _CommandResult:
    """Captured command result."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class _DeveloperBuild:
    """Configured library build used to create a developer environment."""

    library_build_dir: Path
    cmake: Path
    python: Path
    python_install_dir: str
    python_runtime_dependencies: Path
    developer_environment: Path
    generator: str
    generator_platform: str | None
    generator_toolset: str | None
    make_program: str | None


def _validate_keys(
    table: dict[str, Any],
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    """Validate the exact set of keys accepted by a TOML table.

    Args:
        table: Table to validate.
        required: Required field names.
        optional: Optional field names.
        context: User-facing location of the table.
    """
    missing = required - table.keys()
    unknown = table.keys() - required - optional
    if missing:
        raise ExampleError(f"{context} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ExampleError(f"{context} has unknown fields: {', '.join(sorted(unknown))}")


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML document and report parse failures with its path.

    Args:
        path: TOML document path.

    Returns:
        Parsed top-level table.
    """
    if tomllib is None:
        raise ExampleError(f"Reading {path} requires Python 3.11 or newer")
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ExampleError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExampleError(f"{path} must contain a TOML table")
    return value


def _read_string(table: dict[str, Any], field: str, context: str) -> str:
    """Read a required non-empty string.

    Args:
        table: Table containing the field.
        field: Field name to read.
        context: User-facing location of the table.

    Returns:
        Validated string value.
    """
    value = table.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ExampleError(f"{context}.{field} must be a non-empty string")
    return value


def _read_string_list(
    table: dict[str, Any],
    field: str,
    context: str,
    *,
    required: bool = True,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Read a list of unique strings.

    Args:
        table: Table containing the field.
        field: Field name to read.
        context: User-facing location of the table.
        required: Whether the field must be present.
        allow_empty: Whether an empty list is valid.

    Returns:
        Validated string values.
    """
    value = table.get(field)
    if value is None and not required:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ExampleError(f"{context}.{field} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise ExampleError(f"{context}.{field} must not be empty")
    if len(value) != len(set(value)):
        raise ExampleError(f"{context}.{field} must not contain duplicates")
    return tuple(value)


def _read_argument_list(
    table: dict[str, Any],
    field: str,
    context: str,
    *,
    required: bool = True,
) -> tuple[str, ...]:
    """Read an ordered command-line argument list.

    Args:
        table: Table containing the field.
        field: Field name to read.
        context: User-facing location of the table.
        required: Whether the field must be present.

    Returns:
        Argument values exactly as authored.
    """
    value = table.get(field)
    if value is None and not required:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ExampleError(f"{context}.{field} must be a list of strings")
    return tuple(value)


def _load_system_requirements(path: Path) -> _SystemRequirements:
    """Load the release-wide system toolchain requirements.

    Args:
        path: Requirements file shipped beside the modules.

    Returns:
        Validated system requirements.
    """
    table = _read_toml(path)
    context = str(path)
    _validate_keys(table, set(SYSTEM_REQUIREMENTS_FIELDS), set(), context)
    for field in ("cmake_minimum_version", "cmake_policy_maximum_version", "python_version"):
        value = _read_string(table, field, context)
        if VERSION_PATTERN.fullmatch(value) is None:
            raise ExampleError(f"{context}.{field} must be a two- or three-component version")
    if str(table["python_version"]).count(".") != 1:
        raise ExampleError(f"{context}.python_version must identify a Python major and minor version")
    minimum_cmake_parts = tuple(int(component) for component in str(table["cmake_minimum_version"]).split("."))
    maximum_cmake_parts = tuple(int(component) for component in str(table["cmake_policy_maximum_version"]).split("."))
    minimum_cmake = minimum_cmake_parts + (0,) * (3 - len(minimum_cmake_parts))
    maximum_cmake = maximum_cmake_parts + (0,) * (3 - len(maximum_cmake_parts))
    if maximum_cmake < minimum_cmake:
        raise ExampleError(f"{context}.cmake_policy_maximum_version must not precede cmake_minimum_version")
    for field in ("c_standard", "cpp_standard"):
        value = table[field]
        if type(value) is not int or value <= 0:
            raise ExampleError(f"{context}.{field} must be a positive integer")
    return _SystemRequirements(
        cmake_minimum_version=str(table["cmake_minimum_version"]),
        cmake_policy_maximum_version=str(table["cmake_policy_maximum_version"]),
        c_standard=int(table["c_standard"]),
        cpp_standard=int(table["cpp_standard"]),
        python_version=str(table["python_version"]),
    )


def _validate_python_version(requirements: _SystemRequirements) -> None:
    """Require the runner's Python to match the release-wide version.

    Args:
        requirements: Release-wide system requirements.
    """
    actual_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_version != requirements.python_version:
        raise ExampleError(
            f"The examples require Python {requirements.python_version}, "
            f"but {sys.executable} is Python {actual_version}"
        )


def _read_cmake_cache(path: Path) -> dict[str, str]:
    """Read string values from a configured CMake cache.

    Args:
        path: CMake cache path.

    Returns:
        Cache values indexed by variable name.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ExampleError(f"Cannot read the configured library build {path}: {error}") from error

    values: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.partition("=")[0]:
            continue
        key_and_type, _, value = line.partition("=")
        key, _, _ = key_and_type.partition(":")
        values[key] = value
    return values


def _library_source_digest(root: Path) -> str:
    """Fingerprint authored library sources using the build tool's exclusions.

    Args:
        root: Root directory to process.

    Returns:
        The resulting value.
    """
    excluded = frozenset({"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist"})
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts) or not path.is_file():
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_developer_artifact_state(repository_root: Path, library_build_dir: Path, config: str) -> Path:
    """Validate successful full-build state and return its developer environment.

    Args:
        repository_root: Repository root directory.
        library_build_dir: Directory containing built Isaac Sim libraries.
        config: Build configuration name.

    Returns:
        The resulting value.
    """
    artifact_path = library_build_dir / "artifact-state.json"
    try:
        artifact_state = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExampleError(
            f"Developer artifacts are not current at {library_build_dir}; "
            f"run the library build before examples: {error}"
        ) from error
    if not isinstance(artifact_state, dict):
        raise ExampleError(f"Developer build state is invalid at {library_build_dir}")
    if (
        artifact_state.get("schema_version") != 1
        or artifact_state.get("configuration") != config
        or artifact_state.get("profile") != "standard"
        or artifact_state.get("dependency_profile") != "locked"
        or artifact_state.get("targets") != ["all"]
    ):
        raise ExampleError(f"Developer artifacts are stale or partial at {library_build_dir}; rebuild source/libraries")
    inputs = artifact_state.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ExampleError(f"Developer artifact inputs are invalid at {artifact_path}")
    for relative_name, expected_digest in inputs.items():
        if not isinstance(relative_name, str) or not isinstance(expected_digest, str):
            raise ExampleError(f"Developer artifact inputs are invalid at {artifact_path}")
        relative_path = Path(relative_name)
        candidate = (repository_root / relative_path).resolve()
        if (
            relative_path.is_absolute()
            or not candidate.is_relative_to(repository_root.resolve())
            or not candidate.is_file()
        ):
            raise ExampleError(f"Developer configure input is unsafe or missing: {relative_name}")
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise ExampleError(f"Developer configure input changed since the build: {relative_name}")
    expected_source_digest = artifact_state.get("source_digest")
    source_root = repository_root / "source" / "libraries"
    if (
        not isinstance(expected_source_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_source_digest) is None
        or _library_source_digest(source_root) != expected_source_digest
    ):
        raise ExampleError("Developer library sources changed since the build; rebuild source/libraries")
    developer_environment = artifact_state.get("developer_environment")
    if not isinstance(developer_environment, dict) or set(developer_environment) != {"components", "path"}:
        raise ExampleError("Developer environment state is missing; rebuild source/libraries")
    relative_path = developer_environment["path"]
    components = developer_environment["components"]
    if (
        relative_path != DEVELOPER_ENVIRONMENT_DIRECTORY
        or not isinstance(components, list)
        or not components
        or any(not isinstance(component, str) or not component for component in components)
    ):
        raise ExampleError("Developer environment state is invalid; rebuild source/libraries")
    environment_path = library_build_dir / relative_path
    is_junction = getattr(environment_path, "is_junction", None)
    if not environment_path.is_dir() or environment_path.is_symlink() or (is_junction is not None and is_junction()):
        raise ExampleError(f"Developer environment is missing or redirected: {environment_path}")
    return environment_path


@contextlib.contextmanager
def _developer_environment_lock(build_directory: Path, *, exclusive: bool) -> Iterator[None]:
    """Coordinate environment readers with replacement and cleanup.

    Args:
        build_directory: Directory containing the CMake build.
        exclusive: Whether to acquire an exclusive lock.

    Yields:
        Control while the developer-environment lock is held.
    """
    lock_path = build_directory.parent / f".{build_directory.name}-{DEVELOPER_ENVIRONMENT_LOCK_SUFFIX}"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import ctypes
            import msvcrt
            from ctypes import wintypes

            class Overlapped(ctypes.Structure):
                _fields_ = (
                    ("internal", ctypes.c_size_t),
                    ("internal_high", ctypes.c_size_t),
                    ("offset", wintypes.DWORD),
                    ("offset_high", wintypes.DWORD),
                    ("event", wintypes.HANDLE),
                )

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            lock_file_ex = kernel32.LockFileEx
            lock_file_ex.argtypes = (
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.POINTER(Overlapped),
            )
            lock_file_ex.restype = wintypes.BOOL
            unlock_file_ex = kernel32.UnlockFileEx
            unlock_file_ex.argtypes = (
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.POINTER(Overlapped),
            )
            unlock_file_ex.restype = wintypes.BOOL

            overlapped = Overlapped()
            flags = 0x2 if exclusive else 0  # ``LOCKFILE_EXCLUSIVE_LOCK``.
            handle = msvcrt.get_osfhandle(lock_file.fileno())
            if not lock_file_ex(handle, flags, 0, 1, 0, ctypes.byref(overlapped)):
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                yield
            finally:
                if not unlock_file_ex(handle, 0, 1, 0, ctypes.byref(overlapped)):
                    raise ctypes.WinError(ctypes.get_last_error())
        else:
            import fcntl

            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), mode)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_developer_build(repository_root: Path, config: str) -> _DeveloperBuild:
    """Read the configured library build selected for developer execution.

    Args:
        repository_root: Isaac Sim repository root.
        config: Requested build configuration.

    Returns:
        Configured build tools and paths.
    """
    library_build_dir = repository_root / "_cmake_build" / f"isaacsim-libraries-{config.lower()}"
    cache_path = library_build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        build_command = "source\\libraries\\build.bat" if os.name == "nt" else "source/libraries/build.sh"
        raise ExampleError(
            f"Developer mode requires a configured {config} library build at {library_build_dir}. "
            f"Run {build_command} {'-d' if config.lower() == 'debug' else '-r'} first."
        )
    cache = _read_cmake_cache(cache_path)
    configured_type = cache.get("CMAKE_BUILD_TYPE", "")
    if configured_type.lower() != config.lower():
        raise ExampleError(
            f"Developer build {library_build_dir} is configured as {configured_type or 'an unknown configuration'}, "
            f"not {config}."
        )

    required_values = (
        "CMAKE_COMMAND",
        "CMAKE_GENERATOR",
        "ISAACSIM_PYTHON_INSTALL_DIR",
        "ISAACSIM_PYTHON_RUNTIME_DEPS_DIR",
        "Python_EXECUTABLE",
    )
    missing_values = [name for name in required_values if not cache.get(name)]
    if missing_values:
        raise ExampleError(f"Developer build cache is missing values: {', '.join(missing_values)}")

    cmake = Path(cache["CMAKE_COMMAND"])
    python = Path(cache["Python_EXECUTABLE"])
    python_runtime_dependencies = Path(cache["ISAACSIM_PYTHON_RUNTIME_DEPS_DIR"])
    python_install_dir = Path(cache["ISAACSIM_PYTHON_INSTALL_DIR"])
    if python_install_dir.is_absolute() or any(part in {".", ".."} for part in python_install_dir.parts):
        raise ExampleError("Developer build cache has an invalid ISAACSIM_PYTHON_INSTALL_DIR")
    missing_paths = [path for path in (cmake, python) if not path.is_file()]
    if not python_runtime_dependencies.is_dir():
        missing_paths.append(python_runtime_dependencies)
    if missing_paths:
        raise ExampleError(
            "Developer build dependencies are incomplete; rebuild source/libraries. Missing: "
            + ", ".join(str(path) for path in missing_paths)
        )

    developer_environment = _validate_developer_artifact_state(repository_root, library_build_dir, config)
    for required_directory in (
        developer_environment / "lib" / "cmake",
        developer_environment / python_install_dir,
    ):
        if not required_directory.is_dir():
            raise ExampleError(f"Developer environment is incomplete; rebuild source/libraries: {required_directory}")

    return _DeveloperBuild(
        library_build_dir=library_build_dir,
        cmake=cmake,
        python=python,
        python_install_dir=str(python_install_dir),
        python_runtime_dependencies=python_runtime_dependencies,
        developer_environment=developer_environment,
        generator=cache["CMAKE_GENERATOR"],
        generator_platform=cache.get("CMAKE_GENERATOR_PLATFORM") or None,
        generator_toolset=cache.get("CMAKE_GENERATOR_TOOLSET") or None,
        make_program=cache.get("CMAKE_MAKE_PROGRAM") or None,
    )


def _resolve_example_file(example_root: Path, value: str, context: str) -> Path:
    """Resolve a manifest file path within an example root.

    Args:
        example_root: Root containing the example.
        value: POSIX-style manifest path.
        context: User-facing location of the field.

    Returns:
        Resolved path beneath the example root.
    """
    relative_path = PurePosixPath(value)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {".", ".."} for part in relative_path.parts)
    ):
        raise ExampleError(f"{context} must be a relative path within the example")
    path = (example_root / Path(*relative_path.parts)).resolve()
    try:
        path.relative_to(example_root.resolve())
    except ValueError as error:
        raise ExampleError(f"{context} escapes the example root") from error
    return path


def _load_build_configuration(table: Any, example_root: Path, context: str) -> _BuildConfiguration:
    """Load and validate a build adapter.

    Args:
        table: Build adapter table.
        example_root: Root containing the example.
        context: User-facing location of the table.

    Returns:
        Validated build configuration.
    """
    if not isinstance(table, dict):
        raise ExampleError(f"{context} must be a table")
    adapter = _read_string(table, "adapter", context)
    if adapter == "none":
        _validate_keys(table, {"adapter"}, set(), context)
        return _BuildConfiguration(adapter=adapter)
    if adapter != "cmake":
        raise ExampleError(f"{context}.adapter is unsupported: {adapter}")

    _validate_keys(table, {"adapter", "targets"}, set(), context)
    targets = _read_string_list(table, "targets", context)
    if any(TARGET_PATTERN.fullmatch(target) is None for target in targets):
        raise ExampleError(f"{context}.targets contains an invalid CMake target")
    if not (example_root / "CMakeLists.txt").is_file():
        raise ExampleError(f"{context} requires {example_root / 'CMakeLists.txt'}")
    return _BuildConfiguration(adapter=adapter, targets=targets)


def _load_run_configuration(table: Any, example_root: Path, context: str) -> _RunConfiguration:
    """Load and validate a run adapter.

    Args:
        table: Run adapter table.
        example_root: Root containing the example.
        context: User-facing location of the table.

    Returns:
        Validated run configuration.
    """
    if not isinstance(table, dict):
        raise ExampleError(f"{context} must be a table")
    adapter = _read_string(table, "adapter", context)
    arguments = _read_argument_list(table, "arguments", context, required=False)
    if adapter == "executable":
        _validate_keys(table, {"adapter", "target"}, {"arguments"}, context)
        target = _read_string(table, "target", context)
        if TARGET_PATTERN.fullmatch(target) is None:
            raise ExampleError(f"{context}.target is invalid: {target}")
        return _RunConfiguration(adapter=adapter, target=target, arguments=arguments)
    if adapter == "python":
        _validate_keys(table, {"adapter", "path"}, {"arguments"}, context)
        script_path = _resolve_example_file(example_root, _read_string(table, "path", context), f"{context}.path")
        if not script_path.is_file():
            raise ExampleError(f"{context}.path does not exist: {script_path}")
        return _RunConfiguration(adapter=adapter, path=script_path, arguments=arguments)
    raise ExampleError(f"{context}.adapter is unsupported: {adapter}")


def _load_test_configurations(value: Any, context: str) -> tuple[_TestConfiguration, ...]:
    """Load and validate test overlays.

    Args:
        value: Test array value.
        context: User-facing location of the array.

    Returns:
        Validated test configurations.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ExampleError(f"{context} must be an array of tables")

    tests: list[_TestConfiguration] = []
    names: set[str] = set()
    optional_fields = {
        "arguments",
        "environment",
        "expected_exit_code",
        "stderr_contains",
        "stdout_contains",
        "timeout_seconds",
    }
    for index, table in enumerate(value):
        test_context = f"{context}[{index}]"
        if not isinstance(table, dict):
            raise ExampleError(f"{test_context} must be a table")
        _validate_keys(table, {"name"}, optional_fields, test_context)
        name = _read_string(table, "name", test_context)
        if NAME_PATTERN.fullmatch(name) is None:
            raise ExampleError(f"{test_context}.name is invalid: {name}")
        if name in names:
            raise ExampleError(f"{context} contains duplicate test name: {name}")
        names.add(name)

        arguments = None
        if "arguments" in table:
            arguments = _read_argument_list(table, "arguments", test_context)
        timeout_seconds = table.get("timeout_seconds", DEFAULT_TEST_TIMEOUT_SECONDS)
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ExampleError(f"{test_context}.timeout_seconds must be a positive integer")
        expected_exit_code = table.get("expected_exit_code", 0)
        if type(expected_exit_code) is not int:
            raise ExampleError(f"{test_context}.expected_exit_code must be an integer")
        stdout_contains = _read_string_list(table, "stdout_contains", test_context, required=False, allow_empty=True)
        stderr_contains = _read_string_list(table, "stderr_contains", test_context, required=False, allow_empty=True)
        environment = table.get("environment", {})
        if not isinstance(environment, dict) or any(
            not isinstance(key, str) or ENVIRONMENT_NAME_PATTERN.fullmatch(key) is None or not isinstance(item, str)
            for key, item in environment.items()
        ):
            raise ExampleError(f"{test_context}.environment must contain portable variable names and string values")
        if any(key.upper() == "PYTHONDONTWRITEBYTECODE" for key in environment):
            raise ExampleError(f"{test_context}.environment must not override PYTHONDONTWRITEBYTECODE")
        tests.append(
            _TestConfiguration(
                name=name,
                arguments=arguments,
                timeout_seconds=timeout_seconds,
                expected_exit_code=expected_exit_code,
                stdout_contains=stdout_contains,
                stderr_contains=stderr_contains,
                environment=dict(environment),
            )
        )
    return tuple(tests)


def _load_requirements(value: Any, context: str) -> _Requirements:
    """Validate module and system requirements.

    Args:
        value: Requirements table.
        context: User-facing location of the table.

    Returns:
        Validated requirements.
    """
    if not isinstance(value, dict):
        raise ExampleError(f"{context} must be a table")
    _validate_keys(value, {"modules"}, {"system"}, context)
    modules = value["modules"]
    if not isinstance(modules, list) or not modules:
        raise ExampleError(f"{context}.modules must be a non-empty array of tables")
    names: set[str] = set()
    module_requirements: list[_ModuleRequirement] = []
    for index, module in enumerate(modules):
        module_context = f"{context}.modules[{index}]"
        if not isinstance(module, dict):
            raise ExampleError(f"{module_context} must be a table")
        _validate_keys(module, {"name", "surfaces"}, set(), module_context)
        name = _read_string(module, "name", module_context)
        if NAME_PATTERN.fullmatch(name) is None or name in names:
            raise ExampleError(f"{module_context}.name is invalid or duplicated: {name}")
        names.add(name)
        surfaces = _read_string_list(module, "surfaces", module_context)
        unknown_surfaces = set(surfaces) - ALLOWED_MODULE_SURFACES
        if unknown_surfaces:
            raise ExampleError(f"{module_context}.surfaces is unsupported: {', '.join(sorted(unknown_surfaces))}")
        module_requirements.append(_ModuleRequirement(name=name, surfaces=surfaces))

    system = _read_string_list(value, "system", context, required=False, allow_empty=True)
    unknown_system_requirements = set(system) - SYSTEM_REQUIREMENT_NAMES
    if unknown_system_requirements:
        raise ExampleError(
            f"{context}.system contains unknown requirements: {', '.join(sorted(unknown_system_requirements))}"
        )
    if "python_development" in system and "python" not in system:
        raise ExampleError(f"{context}.system python_development requires python")
    return _Requirements(modules=tuple(module_requirements), system=system)


def _validate_requirement_coherence(
    build: _BuildConfiguration,
    run: _RunConfiguration,
    requirements: _Requirements,
    context: str,
) -> None:
    """Validate that adapters and declared requirements agree.

    Args:
        build: Structured build configuration.
        run: Structured run configuration.
        requirements: Module and system requirements.
        context: User-facing location of the manifest.
    """
    surfaces = {surface for module in requirements.modules for surface in module.surfaces}
    system = set(requirements.system)
    if build.adapter == "cmake":
        if "native_sdk" not in surfaces:
            raise ExampleError(f"{context} uses the CMake adapter but declares no native_sdk module surface")
        if "cmake" not in system:
            raise ExampleError(f"{context} uses the CMake adapter but does not require cmake")
        if not {"c", "cpp"} & system:
            raise ExampleError(f"{context} uses the CMake adapter but does not require a C or C++ compiler")

    if run.adapter == "python":
        if "python" not in system:
            raise ExampleError(f"{context} uses the Python run adapter but does not require python")
        if build.adapter == "cmake" and "python_development" not in system:
            raise ExampleError(
                f"{context} uses mixed CMake/Python adapters but does not require Python development headers"
            )


def _load_example(example_root: Path) -> _Example:
    """Load and validate one example manifest.

    Args:
        example_root: Root containing the example manifest.

    Returns:
        Validated example definition.
    """
    manifest_path = example_root / "example.toml"
    if not (example_root / "README.md").is_file():
        raise ExampleError(f"Example root has no README.md: {example_root}")
    table = _read_toml(manifest_path)
    context = str(manifest_path)
    _validate_keys(
        table,
        {"build", "id", "owners", "requirements", "run", "summary", "title"},
        {"categories", "published", "test", "topics"},
        context,
    )
    published = table.get("published", True)
    if type(published) is not bool:
        raise ExampleError(f"{context}.published must be a boolean")
    example_id = _read_string(table, "id", context)
    if EXAMPLE_ID_PATTERN.fullmatch(example_id) is None:
        raise ExampleError(f"{context}.id is invalid: {example_id}")
    title = _read_string(table, "title", context)
    summary = _read_string(table, "summary", context)
    owners = _read_string_list(table, "owners", context)
    if any(EXAMPLE_ID_PATTERN.fullmatch(owner) is None for owner in owners):
        raise ExampleError(f"{context}.owners contains an invalid API owner")
    categories = _read_string_list(table, "categories", context, required=False, allow_empty=True)
    topics = _read_string_list(table, "topics", context, required=False, allow_empty=True)
    for field, values in (("categories", categories), ("topics", topics)):
        if any(NAME_PATTERN.fullmatch(item) is None for item in values):
            raise ExampleError(f"{context}.{field} contains an invalid name")
    build = _load_build_configuration(table["build"], example_root, f"{context}.build")
    run = _load_run_configuration(table["run"], example_root, f"{context}.run")
    if run.adapter == "executable" and (build.adapter != "cmake" or run.target not in build.targets):
        raise ExampleError(f"{context}.run.target must name a declared CMake build target")
    requirements = _load_requirements(table["requirements"], f"{context}.requirements")
    _validate_requirement_coherence(build, run, requirements, context)
    tests = _load_test_configurations(table.get("test"), f"{context}.test")
    return _Example(
        id=example_id,
        title=title,
        summary=summary,
        owners=owners,
        categories=categories,
        topics=topics,
        root=example_root,
        build=build,
        run=run,
        requirements=requirements,
        tests=tests,
    )


def _load_series(
    series_root: Path,
    examples_by_id: dict[str, _Example],
) -> _Series:
    """Load and validate a series manifest and its co-located steps.

    Args:
        series_root: Root containing the series manifest and steps.
        examples_by_id: Published examples indexed by stable ID.

    Returns:
        Validated series definition.
    """
    manifest_path = series_root / "series.toml"
    if not (series_root / "README.md").is_file():
        raise ExampleError(f"Series root has no README.md: {series_root}")
    table = _read_toml(manifest_path)
    context = str(manifest_path)
    _validate_keys(table, {"id", "step", "summary", "title"}, set(), context)
    series_id = _read_string(table, "id", context)
    if EXAMPLE_ID_PATTERN.fullmatch(series_id) is None:
        raise ExampleError(f"{context}.id is invalid: {series_id}")
    title = _read_string(table, "title", context)
    summary = _read_string(table, "summary", context)
    steps = table["step"]
    if not isinstance(steps, list) or not steps:
        raise ExampleError(f"{context}.step must be a non-empty array of tables")

    step_ids: set[str] = set()
    validated_steps: list[_SeriesStep] = []
    for index, step in enumerate(steps):
        step_context = f"{context}.step[{index}]"
        if not isinstance(step, dict):
            raise ExampleError(f"{step_context} must be a table")
        _validate_keys(step, {"example", "level"}, set(), step_context)
        example_id = _read_string(step, "example", step_context)
        level = _read_string(step, "level", step_context)
        if example_id in step_ids:
            raise ExampleError(f"{context} contains duplicate step: {example_id}")
        if example_id not in examples_by_id:
            raise ExampleError(f"{step_context}.example is not published: {example_id}")
        if level not in ALLOWED_LEVELS:
            raise ExampleError(f"{step_context}.level is unsupported: {level}")
        if not examples_by_id[example_id].root.is_relative_to(series_root):
            raise ExampleError(f"{step_context}.example must be physically located under {series_root}")
        step_ids.add(example_id)
        validated_steps.append(_SeriesStep(example_id=example_id, level=level))

    co_located_ids = {example.id for example in examples_by_id.values() if example.root.is_relative_to(series_root)}
    if co_located_ids != step_ids:
        missing = sorted(co_located_ids - step_ids)
        raise ExampleError(f"{context} does not list its co-located examples: {', '.join(missing)}")
    return _Series(id=series_id, title=title, summary=summary, root=series_root, steps=tuple(validated_steps))


def _validate_source_tree(examples_dir: Path) -> None:
    """Reject source symlinks and generated output.

    Args:
        examples_dir: Root of the examples collection.
    """
    for path in examples_dir.rglob("*"):
        if path.is_symlink():
            raise ExampleError(f"Example source must not contain symlinks: {path}")
        if path.is_dir() and path.name in GENERATED_DIRECTORY_NAMES:
            raise ExampleError(f"Example source contains generated output: {path}")
        if path.is_file() and (path.name in GENERATED_FILE_NAMES or path.suffix.lower() in GENERATED_FILE_SUFFIXES):
            raise ExampleError(f"Example source contains generated output: {path}")


def _validate_build_root(build_root: Path, examples_dir: Path) -> None:
    """Require generated build output to remain outside source/examples.

    Args:
        build_root: Requested root for isolated builds.
        examples_dir: Root of the canonical or materialized examples collection.
    """
    if build_root == examples_dir or build_root.is_relative_to(examples_dir):
        raise ExampleError(f"Build root must remain outside source/examples: {build_root}")


def _discover_published_example_roots(examples_dir: Path) -> tuple[Path, ...]:
    """Discover published example roots beneath the examples collection.

    Args:
        examples_dir: Root of the examples collection.

    Returns:
        Published example roots in deterministic path order.
    """
    examples_root = examples_dir.resolve()
    roots: list[Path] = []
    for manifest_path in sorted(examples_dir.rglob("example.toml")):
        table = _read_toml(manifest_path)
        context = str(manifest_path)
        published = table.get("published", True)
        if type(published) is not bool:
            raise ExampleError(f"{context}.published must be a boolean")
        if not published:
            continue
        root = manifest_path.parent.resolve()
        relative_parts = root.relative_to(examples_root).parts
        if not relative_parts or relative_parts[0] not in ALLOWED_EXAMPLE_GROUPS:
            raise ExampleError(f"Published example must start with libraries or series: {root}")
        roots.append(root)
    return tuple(roots)


def _load_collection(examples_dir: Path) -> tuple[tuple[_Example, ...], tuple[_Series, ...]]:
    """Load and validate every published example and series.

    Args:
        examples_dir: Root of the examples collection.

    Returns:
        Published examples and series in deterministic path order.
    """
    _validate_source_tree(examples_dir)
    examples = tuple(_load_example(root) for root in _discover_published_example_roots(examples_dir))
    examples_by_id = {example.id: example for example in examples}
    if len(examples_by_id) != len(examples):
        raise ExampleError("Published examples contain duplicate stable example IDs")

    resolved_series_roots: list[Path] = []
    series_ids: set[str] = set()
    series: list[_Series] = []
    for manifest_path in sorted(examples_dir.rglob("series.toml")):
        path = manifest_path.parent.resolve()
        if not any(example.root.is_relative_to(path) for example in examples):
            continue
        relative_parts = path.relative_to(examples_dir.resolve()).parts
        if len(relative_parts) != 2 or relative_parts[0] != "series":
            raise ExampleError(f"Series root must be located at series/<series-name>: {path}")
        series_entry = _load_series(path, examples_by_id)
        if series_entry.id in series_ids:
            raise ExampleError(f"Published series contain duplicate stable series ID: {series_entry.id}")
        series_ids.add(series_entry.id)
        series.append(series_entry)
        resolved_series_roots.append(path)

    for example in examples:
        if example.root.relative_to(examples_dir.resolve()).parts[0] == "series" and not any(
            example.root.is_relative_to(series_root) for series_root in resolved_series_roots
        ):
            raise ExampleError(f"Published series example is not owned by a published series: {example.root}")
    return examples, tuple(series)


def _load_examples(examples_dir: Path) -> tuple[_Example, ...]:
    """Load and validate every published example.

    Args:
        examples_dir: Root directory of the examples collection.

    Returns:
        The resulting value.
    """
    examples, _ = _load_collection(examples_dir)
    return examples


def _load_library_catalog(path: Path) -> tuple[str, set[str], set[str]]:
    """Load the release version and valid public library identities.

    Args:
        path: Filesystem path to process.

    Returns:
        The resulting value.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise ExampleError(f"Unsupported library documentation catalog schema: {path}")
        release_version = document["release_version"]
        if not isinstance(release_version, str) or not release_version:
            raise TypeError("release_version must be a non-empty string")
        distributions = document["distributions"]
        if not isinstance(distributions, list):
            raise TypeError("distributions must be a list")
        published_distributions = [entry for entry in distributions if entry.get("complete", False)]
        distribution_names = {entry["name"] for entry in published_distributions}
        module_names = {module["name"] for entry in published_distributions for module in entry["modules"]}
    except (AttributeError, KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise ExampleError(f"Cannot read library documentation catalog {path}: {error}") from error
    return release_version, distribution_names, module_names


def _catalog_languages(example: _Example) -> list[str]:
    """Derive documented languages from structured build and run adapters.

    Args:
        example: Example metadata entry.

    Returns:
        The resulting value.
    """
    languages: list[str] = []
    system = set(example.requirements.system)
    for language in ("c", "cpp"):
        if language in system:
            languages.append(language)
    if example.run.adapter == "python":
        languages.append("python")
    return languages


def _build_catalog(examples_dir: Path, library_catalog: Path) -> dict[str, Any]:
    """Build deterministic documentation data from the runner's validated manifests.

    Args:
        examples_dir: Root directory of the examples collection.
        library_catalog: Library catalog metadata.

    Returns:
        The resulting value.
    """
    examples, series = _load_collection(examples_dir)
    release_version, distribution_names, module_names = _load_library_catalog(library_catalog)
    series_membership: dict[str, tuple[str, int, str]] = {}
    serialized_series: list[dict[str, Any]] = []
    for series_entry in series:
        steps = [step.example_id for step in series_entry.steps]
        for position, step in enumerate(series_entry.steps, start=1):
            series_membership[step.example_id] = (series_entry.id, position, step.level)
        serialized_series.append(
            {
                "id": series_entry.id,
                "title": series_entry.title,
                "summary": series_entry.summary,
                "readme_path": (series_entry.root / "README.md").relative_to(examples_dir).as_posix(),
                "steps": steps,
            }
        )

    serialized_examples: list[dict[str, Any]] = []
    for example in sorted(examples, key=lambda item: item.id):
        requirements = [requirement.name for requirement in example.requirements.modules]
        unknown_distributions = sorted(set(requirements) - distribution_names)
        unknown_owners = sorted(set(example.owners) - module_names)
        if unknown_distributions:
            raise ExampleError(
                f"Example {example.id} requires unknown distributions: {', '.join(unknown_distributions)}"
            )
        if unknown_owners:
            raise ExampleError(f"Example {example.id} has unknown public API owners: {', '.join(unknown_owners)}")
        relative_root = example.root.relative_to(examples_dir).as_posix()
        relative_parts = example.root.relative_to(examples_dir).parts
        group = {"libraries": "library", "series": "series", "workflows": "workflow"}[relative_parts[0]]
        membership = series_membership.get(example.id)
        entry_point = (
            example.run.path.relative_to(example.root).as_posix()
            if example.run.path is not None
            else str(example.run.target)
        )
        serialized_examples.append(
            {
                "id": example.id,
                "title": example.title,
                "summary": example.summary,
                "source_path": relative_root,
                "readme_path": f"{relative_root}/README.md",
                "owners": list(example.owners),
                "topics": list(example.topics),
                "categories": list(example.categories),
                "group": group,
                "requirements": {
                    "distributions": requirements,
                    "capabilities": list(example.requirements.system),
                },
                "entry_point": entry_point,
                "languages": _catalog_languages(example),
                "commands": {
                    "build": f"python source/examples/examples.py build {example.id}",
                    "run": f"python source/examples/examples.py run {example.id}",
                    "test": f"python source/examples/examples.py test {example.id}",
                },
                "series_id": membership[0] if membership else None,
                "series_position": membership[1] if membership else None,
                "level": membership[2] if membership else None,
            }
        )
    return {
        "schema_version": 1,
        "release_version": release_version,
        "examples": serialized_examples,
        "series": serialized_series,
    }


def _render_command(command: list[str]) -> str:
    """Render a command for user-facing output.

    Args:
        command: Command and arguments.

    Returns:
        Platform-appropriate command string.
    """
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def _run_checked(command: list[str], working_directory: Path, environment: dict[str, str]) -> None:
    """Run a build command and raise an actionable error when it fails.

    Args:
        command: Command and arguments.
        working_directory: Directory in which to run the command.
        environment: Process environment.
    """
    print(f"Running: {_render_command(command)}")
    result = subprocess.run(command, cwd=working_directory, env=environment, check=False)
    if result.returncode != 0:
        raise ExampleError(f"Command failed with exit code {result.returncode}: {_render_command(command)}")


def _restart_in_environment(executable: Path, environment: dict[str, str]) -> int:
    """Restart this command with a prepared execution environment.

    Run the Windows replacement as a child so the invoking terminal waits for
    the example and receives its exit code. Unix can replace the current
    process directly.

    Args:
        executable: Python executable used for the restarted command.
        environment: Complete environment for the restarted command.

    Returns:
        Exit code from the restarted command on Windows. The Unix path replaces
        the current process and does not return.
    """
    command = [str(executable), str(Path(__file__).resolve()), *sys.argv[1:]]
    if os.name == "nt":
        return subprocess.run(command, env=environment, check=False).returncode
    os.execve(executable, command, environment)


def _build_example(
    example: _Example,
    system_requirements: _SystemRequirements,
    build_root: Path,
    cmake: str,
    config: str,
    generator: str | None,
    generator_platform: str | None,
    generator_toolset: str | None,
    make_program: str | None,
) -> Path:
    """Build one example incrementally and return its build directory.

    Args:
        example: Example to build.
        system_requirements: Release-wide system toolchain requirements.
        build_root: Root for isolated example build directories.
        cmake: CMake executable.
        config: Build configuration.
        generator: Optional CMake generator.
        generator_platform: Optional generator platform.
        generator_toolset: Optional generator toolset.
        make_program: Optional native build program.

    Returns:
        Isolated example build directory.
    """
    build_dir = build_root / example.id
    if example.build.adapter == "none":
        print(f"Build {example.id}: no build required")
        return build_dir

    environment = os.environ.copy()
    configure_command = [
        cmake,
        "-S",
        str(example.root),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={config}",
        f"-DISAACSIM_CMAKE_MINIMUM_VERSION={system_requirements.cmake_minimum_version}",
        f"-DISAACSIM_CMAKE_POLICY_MAXIMUM_VERSION={system_requirements.cmake_policy_maximum_version}",
        f"-DISAACSIM_C_STANDARD={system_requirements.c_standard}",
        f"-DISAACSIM_CXX_STANDARD={system_requirements.cpp_standard}",
        f"-DISAACSIM_PYTHON_VERSION={system_requirements.python_version}",
    ]
    if "python" in example.requirements.system or "python_development" in example.requirements.system:
        configure_command.append(f"-DPython_EXECUTABLE={sys.executable}")
    if generator:
        configure_command.extend(["-G", generator])
    if generator_platform:
        configure_command.extend(["-A", generator_platform])
    if generator_toolset:
        configure_command.extend(["-T", generator_toolset])
    if make_program:
        configure_command.append(f"-DCMAKE_MAKE_PROGRAM={make_program}")
    print(f"Build {example.id}")
    _run_checked(configure_command, example.root, environment)
    _run_checked(
        [
            cmake,
            "--build",
            str(build_dir),
            "--config",
            config,
            "--target",
            *example.build.targets,
            "--parallel",
        ],
        example.root,
        environment,
    )
    return build_dir


def _prepend_environment_path(environment: dict[str, str], name: str, paths: list[Path]) -> None:
    """Prepend existing paths to an environment path variable.

    Args:
        environment: Environment to update.
        name: Path variable name.
        paths: Candidate paths to prepend.
    """
    values = [str(path) for path in paths if path.is_dir()]
    current = environment.get(name)
    if current:
        values.append(current)
    if values:
        environment[name] = os.pathsep.join(values)


def _create_developer_environment(
    developer_build: _DeveloperBuild,
) -> dict[str, str]:
    """Create the isolated environment for developer example execution.

    Args:
        developer_build: Configured library build.

    Returns:
        Environment configured for the build's tools and installed surfaces.
    """
    prefix = developer_build.developer_environment
    environment = os.environ.copy()
    environment[DEVELOPER_ENVIRONMENT_VARIABLE] = str(developer_build.library_build_dir.resolve())
    environment["CMAKE_COMMAND"] = str(developer_build.cmake)
    environment["CMAKE_PREFIX_PATH"] = str(prefix)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # Keep the developer environment isolated from incompatible Isaac Sim packages in the caller's Python path.
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(prefix / developer_build.python_install_dir),
            str(developer_build.python_runtime_dependencies),
        )
    )
    native_runtime_paths = [prefix / "bin"]
    if os.name == "nt":
        native_runtime_paths.extend(
            (
                prefix / "bin" / "plugins",
                developer_build.python.parent,
                developer_build.python_runtime_dependencies / "usd_exchange.libs",
            )
        )
    _prepend_environment_path(environment, "PATH", native_runtime_paths)
    if os.name != "nt":
        _prepend_environment_path(environment, "LD_LIBRARY_PATH", [prefix / "lib"])
    return environment


def _create_runtime_environment(example: _Example, build_dir: Path) -> dict[str, str]:
    """Create the inherited runtime environment for an example.

    Args:
        example: Example being executed.
        build_dir: Example's isolated build directory.

    Returns:
        Runtime process environment.
    """
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    prefix_paths = [Path(value) for value in environment.get("CMAKE_PREFIX_PATH", "").split(os.pathsep) if value]
    native_runtime_paths = [prefix / "bin" for prefix in prefix_paths]
    if os.name == "nt":
        native_runtime_paths.extend(prefix / "bin" / "plugins" for prefix in prefix_paths)
        native_runtime_paths.append(Path(sys.executable).parent)
        for python_path in environment.get("PYTHONPATH", "").split(os.pathsep):
            if python_path:
                native_runtime_paths.append(Path(python_path) / "usd_exchange.libs")
    _prepend_environment_path(environment, "PATH", native_runtime_paths)
    if os.name != "nt":
        _prepend_environment_path(environment, "LD_LIBRARY_PATH", [prefix / "lib" for prefix in prefix_paths])
    if example.run.adapter == "python" and example.build.adapter == "cmake":
        _prepend_environment_path(environment, "PYTHONPATH", [build_dir / "python"])
    return environment


def _create_run_command(example: _Example, build_dir: Path, arguments: tuple[str, ...]) -> list[str]:
    """Create the command for an example run adapter.

    Args:
        example: Example being executed.
        build_dir: Example's isolated build directory.
        arguments: Arguments for the normal entry point.

    Returns:
        Executable command and arguments.
    """
    if example.run.adapter == "python":
        if example.run.path is None:
            raise ExampleError(f"Python example has no script path: {example.id}")
        return [sys.executable, str(example.run.path), *arguments]
    if example.run.target is None:
        raise ExampleError(f"Executable example has no target: {example.id}")
    executable = build_dir / "bin" / example.run.target
    if os.name == "nt":
        executable = executable.with_suffix(".exe")
    if not executable.is_file():
        raise ExampleError(f"Built executable does not exist: {executable}")
    return [str(executable), *arguments]


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate a timed-out process and its descendants.

    Args:
        process: Root process to terminate.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            text=True,
        )
        try:
            process.wait(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
        return

    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + PROCESS_TERMINATION_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)


def _run_captured(
    command: list[str],
    working_directory: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> _CommandResult:
    """Run and capture a command with process-tree timeout handling.

    Args:
        command: Command and arguments.
        working_directory: Directory in which to run the command.
        environment: Process environment.
        timeout_seconds: Maximum execution duration.

    Returns:
        Captured process result.
    """
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=working_directory,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name != "nt",
        creationflags=creation_flags,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            stdout, stderr = "", "Timed-out descendants did not close their output streams."
        return _CommandResult(process.returncode, stdout, stderr, timed_out=True)
    return _CommandResult(process.returncode, stdout, stderr)


def _run_example(example: _Example, build_dir: Path, arguments: tuple[str, ...]) -> int:
    """Run one example interactively.

    Args:
        example: Example to run.
        build_dir: Example's isolated build directory.
        arguments: Additional user arguments.

    Returns:
        Example process exit code.
    """
    command = _create_run_command(example, build_dir, example.run.arguments + arguments)
    environment = _create_runtime_environment(example, build_dir)
    print(f"Run {example.id}: {_render_command(command)}")
    return subprocess.run(command, cwd=example.root, env=environment, check=False).returncode


def _test_example(example: _Example, test: _TestConfiguration, build_dir: Path) -> None:
    """Run and validate one named example test.

    Args:
        example: Example to test.
        test: Named test configuration.
        build_dir: Example's isolated build directory.
    """
    arguments = example.run.arguments if test.arguments is None else test.arguments
    command = _create_run_command(example, build_dir, arguments)
    environment = _create_runtime_environment(example, build_dir)
    environment.update(test.environment)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    print(f"Test {example.id}:{test.name}: {_render_command(command)}")
    result = _run_captured(command, example.root, environment, test.timeout_seconds)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.timed_out:
        raise ExampleError(f"Test {example.id}:{test.name} timed out after {test.timeout_seconds} seconds")
    if result.returncode != test.expected_exit_code:
        raise ExampleError(
            f"Test {example.id}:{test.name} exited with {result.returncode}, expected {test.expected_exit_code}"
        )
    for expected in test.stdout_contains:
        if expected not in result.stdout:
            raise ExampleError(f"Test {example.id}:{test.name} stdout does not contain: {expected}")
    for expected in test.stderr_contains:
        if expected not in result.stderr:
            raise ExampleError(f"Test {example.id}:{test.name} stderr does not contain: {expected}")
    print(f"Test {example.id}:{test.name}: passed")


def _select_examples(examples: tuple[_Example, ...], selectors: list[str]) -> tuple[_Example, ...]:
    """Select examples by stable ID while preserving discovery order.

    Args:
        examples: Complete published examples collection.
        selectors: Stable IDs to select.

    Returns:
        Selected examples in discovery order.
    """
    examples_by_id = {example.id: example for example in examples}
    if not selectors:
        return examples
    if len(selectors) != len(set(selectors)):
        raise ExampleError("Example selectors must not contain duplicates")
    unknown = set(selectors) - examples_by_id.keys()
    if unknown:
        raise ExampleError(f"Unknown example ID: {', '.join(sorted(unknown))}")
    selected_ids = set(selectors)
    return tuple(example for example in examples if example.id in selected_ids)


def _select_run_example(examples: tuple[_Example, ...], selector: str, examples_dir: Path) -> _Example:
    """Select one example by stable ID, root path, or declared Python script.

    Args:
        examples: Complete published examples collection.
        selector: Stable ID or path relative to the examples collection.
        examples_dir: Root of the examples collection.

    Returns:
        Selected example.
    """
    examples_by_id = {example.id: example for example in examples}
    if selector in examples_by_id:
        return examples_by_id[selector]

    selector_path = Path(selector)
    if selector_path.is_absolute():
        raise ExampleError(f"Example path must be relative to source/examples: {selector}")
    examples_root = examples_dir.resolve()
    candidate = (examples_root / selector_path).resolve()
    if not candidate.is_relative_to(examples_root):
        raise ExampleError(f"Example path escapes source/examples: {selector}")
    for example in examples:
        if candidate == example.root or candidate == example.run.path:
            return example
    raise ExampleError(f"Unknown example ID or run path: {selector}")


def _select_tests(
    examples: tuple[_Example, ...],
    selectors: list[str],
) -> tuple[tuple[_Example, _TestConfiguration], ...]:
    """Select all or named test configurations.

    Args:
        examples: Complete published examples collection.
        selectors: Example or named-test selectors.

    Returns:
        Selected example and test pairs.
    """
    examples_by_id = {example.id: example for example in examples}
    if not selectors:
        return tuple((example, test) for example in examples for test in example.tests)

    selected: list[tuple[_Example, _TestConfiguration]] = []
    selected_keys: set[tuple[str, str]] = set()
    for selector in selectors:
        example_id, separator, test_name = selector.partition(":")
        if example_id not in examples_by_id:
            raise ExampleError(f"Unknown example ID: {example_id}")
        example = examples_by_id[example_id]
        tests = example.tests
        if separator:
            tests = tuple(test for test in tests if test.name == test_name)
            if not tests:
                raise ExampleError(f"Unknown test selector: {selector}")
        for test in tests:
            key = (example.id, test.name)
            if key in selected_keys:
                raise ExampleError(f"Duplicate test selector: {example.id}:{test.name}")
            selected_keys.add(key)
            selected.append((example, test))
    return tuple(selected)


def _select_command_examples(
    arguments: argparse.Namespace, examples: tuple[_Example, ...], examples_dir: Path
) -> tuple[_Example, ...]:
    """Select examples whose installed requirements are needed by a command.

    Args:
        arguments: Parsed command-line arguments.
        examples: Complete published examples collection.
        examples_dir: Root of the examples collection.

    Returns:
        Selected examples in deterministic collection order.
    """
    if arguments.command == "build":
        return _select_examples(examples, arguments.examples)
    if arguments.command == "run":
        return (_select_run_example(examples, arguments.example, examples_dir),)

    selected_ids = {example.id for example, _ in _select_tests(examples, arguments.selectors)}
    return tuple(example for example in examples if example.id in selected_ids)


def _add_environment_arguments(parser: argparse.ArgumentParser, destination: str) -> None:
    """Add developer and installed environment overrides to a parser.

    Args:
        parser: Parser that accepts the environment selection.
        destination: Namespace attribute populated by these options.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dev",
        dest=destination,
        action="store_const",
        const="dev",
        help="Require and use the configured source/libraries developer build.",
    )
    group.add_argument(
        "--installed",
        dest=destination,
        action="store_const",
        const="installed",
        help="Use only the currently active environment.",
    )


def _create_argument_parser(default_build_root: Path) -> argparse.ArgumentParser:
    """Create the examples command-line parser.

    Args:
        default_build_root: Default root for isolated build directories.

    Returns:
        Configured command-line parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    _add_environment_arguments(parser, "root_environment_mode")
    parser.add_argument("--build-root", type=Path, default=default_build_root)
    parser.add_argument("--cmake", default=os.environ.get("CMAKE_COMMAND", "cmake"))
    parser.add_argument("--config", default="Release")
    parser.add_argument("--generator")
    parser.add_argument("--generator-platform")
    parser.add_argument("--generator-toolset")
    parser.add_argument("--make-program")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List published examples by stable ID.")

    build_parser = subparsers.add_parser("build", help="Build selected examples, or all examples by default.")
    _add_environment_arguments(build_parser, "command_environment_mode")
    build_parser.add_argument("examples", nargs="*", metavar="EXAMPLE_ID")

    run_parser = subparsers.add_parser("run", help="Build and run one example by ID, root path, or Python script path.")
    _add_environment_arguments(run_parser, "command_environment_mode")
    run_parser.add_argument("example", metavar="EXAMPLE_ID_OR_PATH")
    run_parser.add_argument("arguments", nargs=argparse.REMAINDER, metavar="EXAMPLE_ARGUMENT")

    test_parser = subparsers.add_parser("test", help="Build and test selected examples, or all examples by default.")
    _add_environment_arguments(test_parser, "command_environment_mode")
    test_parser.add_argument(
        "--no-build",
        action="store_true",
        help="Test examples already present under --build-root without invoking the build toolchain.",
    )
    test_parser.add_argument("selectors", nargs="*", metavar="EXAMPLE_ID[:TEST_NAME]")

    catalog_parser = subparsers.add_parser("catalog", help="Write the validated published-example catalog.")
    catalog_parser.add_argument("--format", choices=("json",), default="json")
    catalog_parser.add_argument("--output", type=Path, required=True)
    catalog_parser.add_argument(
        "--library-catalog",
        type=Path,
        default=default_build_root.parents[1]
        / "_cmake_build"
        / "isaacsim-libraries-release"
        / "documentation"
        / "library_catalog.json",
        help="Library documentation catalog used to validate owners and distribution requirements.",
    )
    return parser


def _main() -> int:
    """Execute the examples command-line interface.

    Returns:
        Process exit code.
    """
    examples_dir = Path(__file__).resolve().parent
    repository_root = examples_dir.parents[1]
    parser = _create_argument_parser(repository_root / "_build" / "examples")
    arguments = parser.parse_args()
    resources = contextlib.ExitStack()
    try:
        if arguments.command == "catalog":
            catalog = _build_catalog(examples_dir, arguments.library_catalog)
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return 0
        examples = _load_examples(examples_dir)
        if arguments.command == "list":
            for example in examples:
                print(f"{example.id}\t{example.title}")
            return 0
        selected_modes = {arguments.root_environment_mode, arguments.command_environment_mode} - {None}
        if len(selected_modes) > 1:
            raise ExampleError("--dev and --installed cannot be used together")
        build_entry_point = "build.bat" if os.name == "nt" else "build.sh"
        source_checkout = (repository_root / ".git").exists() and (
            repository_root / "source" / "libraries" / build_entry_point
        ).is_file()
        environment_mode = arguments.command_environment_mode or arguments.root_environment_mode
        environment_mode = environment_mode or ("dev" if source_checkout else "installed")
        developer_build = None
        active_developer_build = None
        environment_was_announced = False
        if environment_mode == "dev":
            if not source_checkout:
                raise ExampleError("--dev requires examples.py to be run from an Isaac Sim source checkout")
            library_build_dir = repository_root / "_cmake_build" / f"isaacsim-libraries-{arguments.config.lower()}"
            resources.enter_context(_developer_environment_lock(library_build_dir, exclusive=False))
            developer_build = _read_developer_build(repository_root, arguments.config)
            expected_developer_build = str(developer_build.library_build_dir.resolve())
            active_developer_build = os.environ.get(DEVELOPER_ENVIRONMENT_VARIABLE)
            if active_developer_build not in {
                expected_developer_build,
                f"{DEVELOPER_BOOTSTRAP_PREFIX}{expected_developer_build}",
            }:
                print(f"Examples environment: developer ({developer_build.library_build_dir})", flush=True)
                environment_was_announced = True
                if Path(sys.executable).resolve() != developer_build.python.resolve():
                    bootstrap_environment = os.environ.copy()
                    bootstrap_environment[DEVELOPER_ENVIRONMENT_VARIABLE] = (
                        f"{DEVELOPER_BOOTSTRAP_PREFIX}{expected_developer_build}"
                    )
                    return _restart_in_environment(developer_build.python, bootstrap_environment)
        system_requirements = _load_system_requirements(
            repository_root / "source" / "libraries" / "system_requirements.toml"
        )
        if environment_mode == "dev":
            if developer_build is None:
                raise ExampleError("Developer environment selection did not resolve a configured build")
            expected_developer_build = str(developer_build.library_build_dir.resolve())
            if active_developer_build != expected_developer_build:
                if not environment_was_announced:
                    print(f"Examples environment: developer ({developer_build.library_build_dir})", flush=True)
                environment = _create_developer_environment(developer_build)
                return _restart_in_environment(developer_build.python, environment)
            arguments.cmake = str(developer_build.cmake)
            arguments.generator = arguments.generator or developer_build.generator
            arguments.generator_platform = arguments.generator_platform or developer_build.generator_platform
            arguments.generator_toolset = arguments.generator_toolset or developer_build.generator_toolset
            arguments.make_program = arguments.make_program or developer_build.make_program
        else:
            print("Examples environment: installed", flush=True)
        _validate_python_version(system_requirements)
        build_root = arguments.build_root.resolve()
        _validate_build_root(build_root, examples_dir)
        common_build_arguments = (
            system_requirements,
            build_root,
            arguments.cmake,
            arguments.config,
            arguments.generator,
            arguments.generator_platform,
            arguments.generator_toolset,
            arguments.make_program,
        )
        if arguments.command == "build":
            for example in _select_examples(examples, arguments.examples):
                _build_example(example, *common_build_arguments)
            return 0
        if arguments.command == "run":
            example = _select_run_example(examples, arguments.example, examples_dir)
            build_dir = _build_example(example, *common_build_arguments)
            passthrough = tuple(arguments.arguments)
            if passthrough[:1] == ("--",):
                passthrough = passthrough[1:]
            return _run_example(example, build_dir, passthrough)

        tests = _select_tests(examples, arguments.selectors)
        selected_example_ids = {example.id for example, _ in tests}
        if arguments.selectors:
            explicitly_selected_examples = {selector.partition(":")[0] for selector in arguments.selectors}
        else:
            explicitly_selected_examples = {example.id for example in examples}
        for example in examples:
            if example.id in explicitly_selected_examples and not example.tests:
                print(f"Test {example.id}: skipped (no test configurations)")
        build_directories: dict[str, Path] = {}
        for example in examples:
            if example.id in selected_example_ids:
                if arguments.no_build:
                    build_directories[example.id] = build_root / example.id
                else:
                    build_directories[example.id] = _build_example(example, *common_build_arguments)
        for example, test in tests:
            _test_example(example, test, build_directories[example.id])
        return 0
    except (ExampleError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        resources.close()


if __name__ == "__main__":
    raise SystemExit(_main())
