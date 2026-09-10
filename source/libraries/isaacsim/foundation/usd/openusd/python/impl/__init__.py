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

import ctypes
import glob
import os
import pathlib
import sys

__all__ = []

_dll_directory_handles = []


# Pre-load libusd_ms (hash-suffixed after packaging) into the global symbol namespace
# before the C++ bindings .so is imported. Python's dlopen uses RTLD_NOW, so all symbols
# in libisaacsim-foundation-usd-openusd.so must be resolvable at load time - they are
# only present if USD shared libraries are already in the process with RTLD_GLOBAL.
def _preload_usd() -> None:
    import pxr

    if sys.platform == "win32":
        # Importing pxr records this directory in PXR_USD_WINDOWS_DLL_PATH for
        # pxr's own extension bootstrap, but that does not add it to the Windows
        # loader search path for Isaac Sim extensions that link OpenUSD directly.
        usd_library_directory = pathlib.Path(
            os.environ.get(
                "PXR_USD_WINDOWS_DLL_PATH",
                pathlib.Path(pxr.__path__[0]).parent / "usd_exchange.libs",
            )
        )
        if not usd_library_directory.is_dir():
            raise RuntimeError(f"Unable to find OpenUSD DLL directory: {usd_library_directory}")
        _dll_directory_handles.append(os.add_dll_directory(usd_library_directory.resolve()))
        return

    # usd-core libraries
    usd_core_libs = pathlib.Path(pxr.__path__[0]).parent / "usd_core.libs"
    usd_core_libs_not_found = []
    for pattern in ("libtbb*.so*", "libusd_ms*.so"):
        candidates = glob.glob(str(usd_core_libs / pattern))
        if not candidates:
            usd_core_libs_not_found.append(pattern)
            continue
        ctypes.CDLL(candidates[0], mode=ctypes.RTLD_GLOBAL)
    if not usd_core_libs_not_found:
        return

    # usd-exchange libraries
    usd_exchange_libs = pathlib.Path(pxr.__path__[0]).parent / "usd_exchange.libs"
    usd_exchange_libs_not_found = []
    for pattern in ("libtbb*.so*", "libusd*.so"):
        candidates = glob.glob(str(usd_exchange_libs / pattern))
        if not candidates:
            usd_exchange_libs_not_found.append(pattern)
            continue
        for candidate in candidates:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
    if not usd_exchange_libs_not_found:
        return

    error_message = "Unable to find/load OpenUSD shared libraries."
    if usd_core_libs_not_found:
        error_message += f" Looked for {', '.join(usd_core_libs_not_found)} in {usd_core_libs}."
    if usd_exchange_libs_not_found:
        error_message += f" Looked for {', '.join(usd_exchange_libs_not_found)} in {usd_exchange_libs}."
    raise RuntimeError(error_message)


_preload_usd()
