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

"""Portable access to schemas and bundled configuration resources."""

from __future__ import annotations

from importlib.resources import files

_PACKAGE = "isaacsim.robot_setup.sysid"


def read_schema_resource(name: str) -> str:
    """Read a bundled schema by filename.

    Args:
        name: Schema filename without path components.

    Returns:
        UTF-8 schema contents.
    """
    return _read_resource("schemas", name)


def read_config_resource(name: str) -> str:
    """Read a bundled configuration example or default by filename.

    Args:
        name: Configuration filename without path components.

    Returns:
        UTF-8 configuration contents.
    """
    return _read_resource("config", name)


def _read_resource(group: str, name: str) -> str:
    if not name or name != name.replace("\\", "/").rsplit("/", 1)[-1]:
        raise ValueError("Resource name must be a filename without path components.")
    resource = files(_PACKAGE).joinpath("resources", group, name)
    if not resource.is_file():
        raise FileNotFoundError(f"Unknown SysID {group} resource: {name}")
    return resource.read_text(encoding="utf-8")
