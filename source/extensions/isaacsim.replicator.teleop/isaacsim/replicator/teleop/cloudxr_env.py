# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""External CloudXR runtime environment helpers for live teleop sessions.

The CloudXR runtime is started externally (for example
``python -m isaacteleop.cloudxr --accept-eula`` in a separate terminal).
This module only detects readiness and applies ``cloudxr.env`` to the current
process before opening a live OpenXR session.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def default_cloudxr_install_dir() -> Path:
    """Return the default CloudXR install directory."""
    return Path.home() / ".cloudxr"


def cloudxr_env_filepath(install_dir: Path | None = None) -> Path:
    """Return the path to the CloudXR env file written by the runtime launcher."""
    root = install_dir if install_dir is not None else default_cloudxr_install_dir()
    return root / "run" / "cloudxr.env"


def cloudxr_runtime_ready_sentinel(install_dir: Path | None = None) -> Path:
    """Return the path to the ``runtime_started`` sentinel created by a healthy runtime."""
    root = install_dir if install_dir is not None else default_cloudxr_install_dir()
    return root / "run" / "runtime_started"


def is_cloudxr_runtime_ready(install_dir: Path | None = None) -> bool:
    """Return whether a CloudXR runtime has signaled readiness on disk."""
    return cloudxr_runtime_ready_sentinel(install_dir).is_file()


def parse_cloudxr_env_file(path: Path) -> dict[str, str]:
    """Parse a ``cloudxr.env`` file (``export KEY=value`` lines)."""
    if not path.is_file():
        raise FileNotFoundError(f"CloudXR env file not found: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not _ENV_KEY_RE.fullmatch(key):
            continue
        try:
            parts = shlex.split(value, posix=True)
        except ValueError as exc:
            raise ValueError(f"Malformed CloudXR env value for {key}: {exc}") from exc
        parsed_value = parts[0] if parts else ""
        values[key] = os.path.expanduser(os.path.expandvars(parsed_value))
    return values


def apply_cloudxr_env(values: dict[str, str]) -> None:
    """Apply CloudXR environment variables to the current process."""
    for key, value in values.items():
        os.environ[key] = value


def load_and_apply_cloudxr_env(install_dir: Path | None = None) -> Path:
    """Load ``cloudxr.env`` from disk and apply it to ``os.environ``.

    Returns:
        Path to the env file that was loaded.

    Raises:
        FileNotFoundError: When the env file does not exist yet.
    """
    env_path = cloudxr_env_filepath(install_dir)
    apply_cloudxr_env(parse_cloudxr_env_file(env_path))
    return env_path


def prepare_live_cloudxr_env(install_dir: Path | None = None) -> tuple[bool, str]:
    """Verify CloudXR is ready and apply OpenXR env vars for a live teleop connect.

    Returns:
        Success flag and a user-facing status or error message.
    """
    root = install_dir if install_dir is not None else default_cloudxr_install_dir()
    if not is_cloudxr_runtime_ready(root):
        return False, "CloudXR not running"
    try:
        load_and_apply_cloudxr_env(root)
    except FileNotFoundError:
        return False, "CloudXR env missing"
    except (OSError, ValueError) as exc:
        return False, f"CloudXR env invalid: {exc}"
    return True, ""
