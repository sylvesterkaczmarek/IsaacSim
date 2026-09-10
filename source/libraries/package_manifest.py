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

"""Validate the shared standalone-library package manifest contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_MODULE_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")


def load_package_manifest(
    path: Path,
    expected_name: str,
    *,
    expected_version: str | None = None,
) -> dict[str, Any]:
    """Load and validate one generated or installed package manifest.

    Args:
        path: Manifest JSON file.
        expected_name: Distribution that must own the manifest.
        expected_version: Exact release version required by the caller, if any.

    Returns:
        Validated package manifest.

    Raises:
        RuntimeError: If the manifest is missing, malformed, or internally inconsistent.

    """
    if not path.is_file():
        raise RuntimeError(f"Package manifest is missing: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read package manifest {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Package manifest is not an object: {path}")
    if loaded.get("name") != expected_name or loaded.get("complete") is not True:
        raise RuntimeError(f"Package manifest does not describe complete distribution {expected_name}: {path}")

    version = loaded.get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError(f"Package manifest does not contain a release version: {path}")
    if expected_version is not None and version != expected_version:
        raise RuntimeError(f"Package manifest version {version!r} does not match {expected_version!r}: {path}")

    modules = loaded.get("modules")
    if (
        not isinstance(modules, list)
        or not all(isinstance(module, str) and _MODULE_NAME_PATTERN.fullmatch(module) is not None for module in modules)
        or len(modules) != len(set(modules))
    ):
        raise RuntimeError(f"Package manifest contains an invalid module inventory: {path}")
    python_imports = loaded.get("python_imports")
    if not isinstance(python_imports, list) or not all(
        isinstance(import_name, str) and import_name for import_name in python_imports
    ):
        raise RuntimeError(f"Package manifest contains an invalid Python import inventory: {path}")

    dependencies = loaded.get("dependencies")
    expected_dependency_specifier = f"=={version}"
    if not isinstance(dependencies, dict) or not all(
        isinstance(dependency_name, str)
        and dependency_name
        and isinstance(dependency_specifier, str)
        and dependency_specifier == expected_dependency_specifier
        for dependency_name, dependency_specifier in dependencies.items()
    ):
        raise RuntimeError(f"Package manifest does not contain exact shared-version dependencies: {path}")
    return loaded
