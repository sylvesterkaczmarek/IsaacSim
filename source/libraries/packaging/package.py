#!/usr/bin/env python3
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

"""Create native archives from CMake install components."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sysconfig
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

_MANIFEST_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "package_manifest.py"
_MANIFEST_CONTRACT_SPEC = importlib.util.spec_from_file_location("isaacsim_package_manifest", _MANIFEST_CONTRACT_PATH)
if _MANIFEST_CONTRACT_SPEC is None or _MANIFEST_CONTRACT_SPEC.loader is None:
    raise RuntimeError(f"Cannot load package manifest contract: {_MANIFEST_CONTRACT_PATH}")
_MANIFEST_CONTRACT = importlib.util.module_from_spec(_MANIFEST_CONTRACT_SPEC)
_MANIFEST_CONTRACT_SPEC.loader.exec_module(_MANIFEST_CONTRACT)

_ArchiveKind = Literal["runtime", "sdk"]


@dataclass(frozen=True, slots=True)
class _ArchiveArguments:
    """Validated native archive command-line arguments.

    Args:
        kind: Native artifact kind to create.
        build_directory: Existing CMake build directory.
        output_directory: Directory that receives the archive.
        group: Distribution package to install.
        config: Optional multi-configuration build name.
        cmake: CMake executable or command name.

    """

    kind: _ArchiveKind
    #: Native artifact kind to create.
    build_directory: Path
    #: Existing CMake build directory.
    output_directory: Path
    #: Directory that receives the archive.
    group: str
    #: Distribution package to install.
    config: str
    #: Optional multi-configuration build name.
    cmake: str
    #: CMake executable or command name.


def _parse_arguments() -> _ArchiveArguments:
    """Parse the native archive command-line arguments.

    Returns:
        Parsed command-line arguments.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("runtime", "sdk"))
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--cmake", default="cmake")
    parsed = parser.parse_args()
    return _ArchiveArguments(
        kind=cast(_ArchiveKind, parsed.kind),
        build_directory=cast(Path, parsed.build_dir),
        output_directory=cast(Path, parsed.output_dir),
        group=cast(str, parsed.group),
        config=cast(str, parsed.config),
        cmake=cast(str, parsed.cmake),
    )


def _run_install(arguments: _ArchiveArguments, component: str, prefix: Path) -> None:
    """Install one CMake component into a staging prefix.

    Args:
        arguments: Command-line arguments that select the build and CMake executable.
        component: CMake install component to stage.
        prefix: Directory that receives the installed component.

    """
    command = [
        arguments.cmake,
        "--install",
        str(arguments.build_directory.resolve()),
        "--prefix",
        str(prefix),
        "--component",
        component,
    ]
    if arguments.config:
        command.extend(("--config", arguments.config))
    subprocess.run(command, check=True)


def _get_platform_tag() -> str:
    """Get the normalized platform tag for an archive name.

    Returns:
        Platform identifier with punctuation replaced by underscores.

    """
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


def _load_manifest(arguments: _ArchiveArguments, stage: Path) -> dict[str, object]:
    """Load and validate the staged distribution manifest.

    Args:
        arguments: Command-line arguments that identify the distribution.
        stage: Installation staging directory.

    Returns:
        Validated distribution manifest.

    Raises:
        RuntimeError: If the manifest identity or version is invalid.

    """
    path = stage / "share" / "isaacsim" / "packages" / arguments.group / "package.json"
    return cast(dict[str, object], _MANIFEST_CONTRACT.load_package_manifest(path, arguments.group))


def _create_native_archive(arguments: _ArchiveArguments, stage: Path) -> Path:
    """Create a native runtime or SDK archive.

    Args:
        arguments: Command-line arguments that select the archive contents.
        stage: Temporary installation staging directory.

    Returns:
        Path to the created archive.

    """
    _run_install(arguments, f"{arguments.group}-runtime", stage)
    if arguments.kind == "sdk":
        _run_install(arguments, f"{arguments.group}-development", stage)

    version = _load_manifest(arguments, stage)["version"]
    if not isinstance(version, str):
        raise RuntimeError("The validated distribution version is not a string")
    name = f"{arguments.group}-{version}-{_get_platform_tag()}-{arguments.kind}.tar.gz"
    output = arguments.output_directory / name
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(stage.rglob("*")):
            archive.add(path, arcname=path.relative_to(stage), recursive=False)
    return output


def _main() -> int:
    """Run the native archive command.

    Returns:
        Process exit status.

    """
    arguments = _parse_arguments()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    stage = arguments.output_directory / ".staging" / f"{arguments.group}-{arguments.kind}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    try:
        output = _create_native_archive(arguments, stage)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        try:
            (arguments.output_directory / ".staging").rmdir()
        except OSError:
            pass

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
