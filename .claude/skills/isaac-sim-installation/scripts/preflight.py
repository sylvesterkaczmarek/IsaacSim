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

"""Collect Isaac Sim installation preflight information."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], timeout: int = 15) -> dict[str, object]:
    if not shutil.which(command[0]):
        return {"available": False, "command": command, "error": f"{command[0]} not found"}
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "command": command, "error": str(exc)}
    return {
        "available": True,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip()[-4000:],
        "stderr": result.stderr.strip()[-4000:],
    }


def gpu_summary() -> dict[str, object]:
    cuda_version = None
    basic_smi = run(["nvidia-smi"])
    if basic_smi.get("returncode") == 0:
        match = re.search(r"CUDA Version:\s*([0-9.]+)", str(basic_smi.get("stdout", "")))
        if match:
            cuda_version = match.group(1)

    query = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query.get("available") or query.get("returncode") != 0:
        return {
            "available": query.get("available", False),
            "error": query.get("stderr") or query.get("stdout") or query.get("error"),
        }

    gpus = []
    for line in str(query.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, driver, memory_mib = parts
        gpus.append(
            {
                "name": name,
                "driver_version": driver,
                "cuda_version": cuda_version,
                "memory_total_mib": memory_mib,
                "rtx_name_hint": "RTX" in name.upper(),
            }
        )
    return {"available": True, "gpus": gpus}


def memory_gb() -> float | None:
    if os.name == "nt":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / 1024 / 1024 / 1024, 1)
        return None
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 1024 / 1024, 1)
    return None


def disk_gb(path: str) -> float:
    usage = shutil.disk_usage(os.path.expanduser(path))
    return round(usage.free / 1024 / 1024 / 1024, 1)


def python312_prefix() -> list[str]:
    if sys.version_info[:2] == (3, 12):
        return [sys.executable]
    if os.name == "nt":
        python312 = shutil.which("python3.12")
        if python312:
            return [python312]
        launcher = shutil.which("py")
        if launcher:
            return [launcher, "-3.12"]
    return ["python3.12"]


def require_docker_host() -> None:
    system = platform.system().lower()
    release = platform.release().lower()
    if system != "linux" or "microsoft" in release or "wsl" in release:
        raise RuntimeError(
            "Isaac Sim Docker preflight is supported only on native Linux hosts "
            "(x86_64 or aarch64); Windows and WSL are not supported."
        )
    machine = platform.machine().lower()
    if machine not in {"amd64", "x86_64", "aarch64", "arm64"}:
        raise RuntimeError(
            f"Unsupported Isaac Sim Docker host architecture: {machine}. "
            "Only Linux x86_64 and Linux aarch64 are supported."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Isaac Sim install preflight details.")
    parser.add_argument("--docker-gpu-check", action="store_true", help="Run CUDA container GPU validation.")
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    if args.docker_gpu_check:
        try:
            require_docker_host()
        except RuntimeError as exc:
            parser.error(str(exc))

    python312 = python312_prefix()
    report: dict[str, object] = {
        "system": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "memory_gb": memory_gb(),
            "home_free_gb": disk_gb("~"),
        },
        "commands": {
            "nvidia_smi": run(["nvidia-smi"]),
            "python3_12": run([*python312, "--version"]),
            "python3_12_venv": run([*python312, "-m", "venv", "--help"]),
            "python3_12_ensurepip": run([*python312, "-m", "ensurepip", "--version"]),
            "docker": run(["docker", "--version"]),
            "nvidia_ctk": run(["nvidia-ctk", "--version"]),
        },
        "gpu_summary": gpu_summary(),
        "environment": {
            "DISPLAY": os.environ.get("DISPLAY"),
            "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
            "OMNI_KIT_ACCEPT_EULA": os.environ.get("OMNI_KIT_ACCEPT_EULA"),
        },
    }

    if args.docker_gpu_check:
        report["commands"]["docker_gpu"] = run(
            [
                "docker",
                "run",
                "--rm",
                "--runtime=nvidia",
                "--gpus",
                "all",
                "nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04",
                "nvidia-smi",
            ],
            timeout=120,
        )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))
        print("\nSummary:")
        print("- Python package install expects Python 3.12.")
        venv_cmd = report["commands"]["python3_12_venv"]
        ensurepip_cmd = report["commands"]["python3_12_ensurepip"]
        venv_ok = isinstance(venv_cmd, dict) and venv_cmd.get("returncode") == 0
        ensurepip_ok = isinstance(ensurepip_cmd, dict) and ensurepip_cmd.get("returncode") == 0
        if venv_ok and ensurepip_ok:
            print("- python3.12 venv + ensurepip available.")
        else:
            if os.name == "nt":
                print(
                    "- Python 3.12 venv/ensurepip missing; repair or reinstall Python 3.12 "
                    "with pip and the standard library before the Python package flow."
                )
            else:
                print(
                    "- python3.12 venv/ensurepip missing; install the venv module before "
                    "the Python package flow. On Debian/Ubuntu: sudo apt install python3.12-venv"
                )
        gpu_info = report["gpu_summary"]
        if isinstance(gpu_info, dict) and gpu_info.get("gpus"):
            for gpu in gpu_info["gpus"]:
                print(
                    "- GPU detected: "
                    f"{gpu['name']} | driver {gpu['driver_version']} | "
                    f"CUDA {gpu['cuda_version']} | VRAM {gpu['memory_total_mib']} MiB"
                )
                if not gpu["rtx_name_hint"]:
                    print("  Note: GPU name does not contain RTX; run Compatibility Checker for final support status.")
        else:
            print("- GPU/driver query did not succeed; run nvidia-smi and the Compatibility Checker.")
        if os.name == "nt":
            print("- Docker installation is unsupported on Windows and WSL; use Standalone or Python.")
        else:
            toolkit = report["commands"]["nvidia_ctk"]
            if isinstance(toolkit, dict) and toolkit.get("returncode") == 0:
                print("- NVIDIA Container Toolkit detected.")
            else:
                print(
                    "- Container flow needs Docker plus NVIDIA Container Toolkit; "
                    "install/configure it before Docker launch."
                )
            if not args.docker_gpu_check:
                print("- Run preflight with --docker-gpu-check to verify Docker can access the GPU.")
        print("- Keep at least 50GB free storage; more is better for caches.")
        print("- Run the Isaac Sim Compatibility Checker for final GPU/driver pass-fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
