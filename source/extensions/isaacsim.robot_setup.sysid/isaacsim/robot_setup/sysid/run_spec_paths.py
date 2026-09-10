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

"""Resolve filesystem paths embedded in System Identification run specifications."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname

RUN_SPEC_LOCAL_PATH_FIELDS: tuple[tuple[str, str], ...] = (
    ("stage", "input_path"),
    ("telemetry", "source_path"),
    ("telemetry", "mapping_path"),
    ("telemetry", "chunk_manifest_path"),
    ("outputs", "provenance_path"),
    ("outputs", "validation_animation_path"),
)
#: RunSpec object and field names that contain local filesystem paths.

RUN_SPEC_LOCAL_PATH_KEYS: frozenset[str] = frozenset(field_name for _, field_name in RUN_SPEC_LOCAL_PATH_FIELDS)
#: RunSpec field names that contain local filesystem paths.


def is_run_spec_non_local_path(value: str) -> bool:
    """Return whether a RunSpec path identifies a URL or anonymous stage.

    Args:
        value: Serialized RunSpec path.

    Returns:
        True when the value must not be treated as a local filesystem path.
    """
    scheme, separator, _remainder = value.partition("://")
    return value.startswith("anon:") or bool(separator and scheme.lower() != "file")


def _local_path_from_value(value: str) -> Path:
    """Convert a serialized local path or file URL into a platform path.

    Args:
        value: Serialized local path value.

    Returns:
        Platform-native path representation.
    """
    if not value.lower().startswith("file://"):
        return Path(value)
    parsed = urlsplit(value)
    local_value = url2pathname(parsed.path)
    if parsed.netloc:
        local_value = f"//{parsed.netloc}{local_value}"
    return Path(local_value)


def resolve_run_spec_local_path(value: str, base_directory: Path) -> str:
    """Resolve a filesystem path relative to its RunSpec directory.

    Args:
        value: Serialized path value from the RunSpec.
        base_directory: Directory containing the RunSpec JSON file.

    Returns:
        Absolute local path, or the original value for URLs and non-local paths.
    """
    if not value or is_run_spec_non_local_path(value):
        return value
    path = _local_path_from_value(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    if PureWindowsPath(value).is_absolute():
        # Preserve Windows drive and UNC paths when a RunSpec is inspected on
        # another host rather than incorrectly rebasing them as local paths.
        return value
    return str((base_directory / path).resolve())


def resolve_run_spec_local_paths(spec: Any, base_directory: Path) -> None:
    """Resolve every local filesystem field carried by a RunSpec.

    Args:
        spec: Parsed SysID RunSpec to update in place.
        base_directory: Directory containing the RunSpec JSON file.
    """
    # Keep this explicit contract synchronized with path-backed fields in
    # `SysIdRunSpec`. Names ending in `_path` cannot be inferred because fields
    # such as `robot_prim_path` and `source_env_path` are USD prim paths.
    for owner_name, field_name in RUN_SPEC_LOCAL_PATH_FIELDS:
        owner = getattr(spec, owner_name)
        value = str(getattr(owner, field_name, "") or "")
        if value:
            setattr(owner, field_name, resolve_run_spec_local_path(value, base_directory))
