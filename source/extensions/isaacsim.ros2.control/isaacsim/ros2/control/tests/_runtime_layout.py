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

"""Runtime-layout helpers shared by ros2_control packaging tests."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

IS_WINDOWS = os.name == "nt"
LIBRARY_PATH_ENV = "PATH" if IS_WINDOWS else "LD_LIBRARY_PATH"

if not IS_WINDOWS:
    _LIBC = ctypes.CDLL(None)
    _LIBC.getenv.argtypes = [ctypes.c_char_p]
    _LIBC.getenv.restype = ctypes.c_char_p


def process_env(name: str) -> str:
    if IS_WINDOWS:
        encoded = name.encode("utf-8")
        for crt_name in ("ucrtbase", "msvcrt"):
            try:
                crt = ctypes.CDLL(crt_name)
                crt.getenv.argtypes = [ctypes.c_char_p]
                crt.getenv.restype = ctypes.c_char_p
                value = crt.getenv(encoded)
            except (AttributeError, OSError):
                continue
            if value:
                return value.decode("utf-8")
        return os.environ.get(name, "")

    value = _LIBC.getenv(name.encode("utf-8"))
    return value.decode("utf-8") if value else ""


def env_paths(name: str) -> list[str]:
    return [path_key(path) for path in process_env(name).split(os.pathsep) if path]


def path_key(path: Path | str) -> str:
    value = str(path)
    if IS_WINDOWS and value.startswith("\\\\?\\"):
        value = value[4:]
    return os.path.normcase(os.path.normpath(value))


def backend_library_name(ros_distro: str) -> str:
    if IS_WINDOWS:
        return f"isaacsim.ros2.control.{ros_distro}.dll"
    return f"libisaacsim.ros2.control.{ros_distro}.so"


def loaded_library_paths() -> list[str]:
    if IS_WINDOWS:
        return _loaded_windows_module_paths()

    proc_maps = Path("/proc/self/maps")
    if not proc_maps.is_file():
        return []

    paths = []
    for line in proc_maps.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[-1].startswith("/"):
            paths.append(fields[-1])
    return paths


def _loaded_windows_module_paths() -> list[str]:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)

    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    psapi.EnumProcessModules.restype = wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD

    process = kernel32.GetCurrentProcess()
    modules = (wintypes.HMODULE * 4096)()
    bytes_needed = wintypes.DWORD()
    if not psapi.EnumProcessModules(process, modules, ctypes.sizeof(modules), ctypes.byref(bytes_needed)):
        return []

    count = min(bytes_needed.value // ctypes.sizeof(wintypes.HMODULE), len(modules))
    paths = []
    for module in modules[:count]:
        buffer = ctypes.create_unicode_buffer(32768)
        length = psapi.GetModuleFileNameExW(process, module, buffer, len(buffer))
        if length:
            paths.append(buffer.value)
    return paths
