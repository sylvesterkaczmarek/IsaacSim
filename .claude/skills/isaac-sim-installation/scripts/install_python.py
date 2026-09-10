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

"""Plan or execute Isaac Sim Python package installation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path

# Only inherit env vars needed by pip/venv/python. Anything else (tokens,
# cloud creds, unrelated app secrets) is intentionally dropped so the
# subprocess cannot read or leak them.
_SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "ALLUSERSPROFILE",
        "COMMONPROGRAMFILES",
        "COMMONPROGRAMFILES(X86)",
        "COMMONPROGRAMW6432",
        "COMPUTERNAME",
        "DRIVERDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "PROCESSOR_LEVEL",
        "PROCESSOR_REVISION",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "PUBLIC",
        "USERDOMAIN",
        "USERNAME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "VIRTUAL_ENV",
        "PIP_CACHE_DIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)


def _safe_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    # Build the subprocess environment by pulling each allowlisted variable
    # via os.getenv individually. This deliberately avoids enumerating the
    # parent environment table so unrelated host secrets are never touched.
    filtered: dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        value = os.getenv(key)
        if value is not None:
            filtered[key] = value
    if extra:
        filtered.update(extra)
    return filtered


def print_command(command: list[str], env: dict[str, str] | None = None) -> None:
    if platform.system().lower().startswith("win"):
        prefix = ""
        if env:
            prefix = " && ".join(f'set "{key}={value}"' for key, value in env.items()) + " && "
        print(prefix + subprocess.list2cmdline(command))
    else:
        prefix = ""
        if env:
            prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items()) + " "
        print(prefix + shlex.join(command))


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print_command(command, env)
    subprocess.run(command, check=True, env=_safe_env(env))


def venv_python(env_dir: Path) -> Path:
    if platform.system().lower().startswith("win"):
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def default_python() -> str:
    if sys.version_info[:2] == (3, 12):
        return sys.executable
    return "python" if platform.system().lower().startswith("win") else "python3.12"


def verify_nvidia_package_origins(report_path: Path) -> None:
    """Require every resolved isaacsim distribution to come from NVIDIA HTTPS."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checked: list[str] = []
    for item in report.get("install", []):
        name = str(item.get("metadata", {}).get("name", ""))
        normalized_name = re.sub(r"[-_.]+", "-", name).lower()
        if not normalized_name.startswith("isaacsim"):
            continue
        url = str(item.get("download_info", {}).get("url", ""))
        parsed = urllib.parse.urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not (hostname == "nvidia.com" or hostname.endswith(".nvidia.com")):
            raise RuntimeError(f"Refusing package '{name}' from non-NVIDIA source: {url or '<missing URL>'}")
        checked.append(name)
    if not checked:
        raise RuntimeError("Pip resolution report did not contain the required isaacsim package.")
    print(f"NVIDIA package provenance verified: {', '.join(sorted(checked))}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Isaac Sim into a Python 3.12 virtual environment.")
    parser.add_argument("--env-dir", default="~/env_isaacsim", help="Virtual environment directory.")
    parser.add_argument("--python", default=default_python(), help="Python 3.12 executable to use.")
    parser.add_argument(
        "--extras",
        default="all,extscache",
        help="Isaac Sim extras, for example all,extscache or ros2.",
    )
    parser.add_argument(
        "--version",
        help="Exact isaacsim version. Required with --execute.",
    )
    parser.add_argument(
        "--accept-eula",
        action="store_true",
        help="Set OMNI_KIT_ACCEPT_EULA=YES for install validation.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default is dry-run.")
    args = parser.parse_args()
    if args.execute and not args.version:
        parser.error(
            "--version is required with --execute so NVIDIA packages are resolved " "at an explicit release version."
        )

    env_dir = Path(args.env_dir).expanduser()
    python = shutil.which(args.python) or args.python
    env_python = venv_python(env_dir)
    package = f"isaacsim[{args.extras}]"
    if args.version:
        package += f"=={args.version}"

    is_windows = platform.system().lower().startswith("win")
    install_packages = [package]
    if is_windows or "compatibility-check" in {extra.strip() for extra in args.extras.split(",")}:
        # Isaac Sim 6.0.1's Windows compatibility-check extra imports
        # packaging.version but does not declare packaging as a dependency.
        install_packages.insert(0, "packaging")

    resolution_report = env_dir / ".isaacsim-pip-resolution.json"
    pip_install_prefix = [str(env_python), "-m", "pip", "--isolated", "install"]
    commands = [
        [python, "-m", "venv", str(env_dir)],
        [str(env_python), "-m", "pip", "--isolated", "install", "--upgrade", "pip"],
        [
            *pip_install_prefix,
            "--report",
            str(resolution_report),
            *install_packages,
            "--extra-index-url",
            "https://pypi.nvidia.com",
        ],
    ]
    isaacsim_command = env_dir / "Scripts" / "isaacsim.exe"
    compatibility_command = [
        str(isaacsim_command),
        "isaacsim.exp.compatibility_check",
        "--no-window",
    ]
    compatibility_validator_command = [
        sys.executable,
        str(Path(__file__).with_name("validate_install.py")),
        "python-compatibility",
        "--isaacsim",
        str(isaacsim_command),
        "--execute",
    ]
    print("Isaac Sim Python install plan:")
    print("# Installation-only boundary: do not import or launch the Isaac Sim main application.")
    for command in commands:
        print_command(command)
    if is_windows:
        print("# Windows-only post-install Compatibility Checker:")
        print_command(compatibility_command, {"OMNI_KIT_ACCEPT_EULA": "YES"})
        print("# Require the explicit result: System checking result: PASSED")

    if not args.execute:
        print("\nDry-run only. Re-run with --execute after confirming the target environment.")
        return 0

    pip_env = {
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
    }
    run(commands[0])
    run(commands[1], pip_env)
    try:
        run(commands[2], pip_env)
        verify_nvidia_package_origins(resolution_report)
    finally:
        resolution_report.unlink(missing_ok=True)
    if is_windows:
        compatibility_result = subprocess.run(
            compatibility_validator_command,
            check=False,
            env=_safe_env({"OMNI_KIT_ACCEPT_EULA": "YES"}),
        )
        if compatibility_result.returncode != 0:
            raise RuntimeError(
                "Isaac Sim Python packages were installed, but the Windows "
                "Compatibility Checker did not report PASSED. "
                f"Installed environment: {env_dir.resolve()}. "
                f"Python executable: {env_python.absolute()}."
            )
    print("Isaac Sim Python package installed successfully.")
    print(f"Virtual environment: {env_dir.resolve()}")
    print(f"Python executable: {env_python.absolute()}")
    print(f"Installed package request: {package}")
    if "packaging" in install_packages:
        print("Compatibility Checker dependency workaround: packaging")
    if is_windows:
        print(f"Compatibility Checker: PASSED ({isaacsim_command})")
    print("The Isaac Sim main application was not imported or launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
