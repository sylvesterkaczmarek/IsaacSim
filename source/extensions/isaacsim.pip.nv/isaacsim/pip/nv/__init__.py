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

import os
import sys

import omni.ext

__all__ = []


def _add_prebundle_to_sys_path(ext_root: str) -> None:
    """Make the bundled nvidia/* CUDA wheels discoverable on sys.path.

    torch (omni.isaac.ml_archive) locates its CUDA runtime libraries by scanning sys.path for the
    bundled nvidia/* wheels. Those wheels were split out of ml_archive into this extension's
    pip_prebundle, so torch's CUDA-dependency fallback loader can only find them if that directory
    is on sys.path. Kit registers the prebundle for its own import machinery but does not add it to
    the global sys.path, and not every launcher sources setup_python_env.sh (e.g. the extension test
    runner invokes kit/kit directly). This extension loads early (order = -1000), before torch is
    ever imported, so inserting the path here makes the CUDA libraries discoverable in every launch
    context.

    Args:
        ext_root: Root directory of the extension installation.
    """
    pip_prebundle = os.path.join(ext_root, "pip_prebundle")
    if os.path.isdir(pip_prebundle) and pip_prebundle not in sys.path:
        sys.path.insert(0, pip_prebundle)


# Best-effort on import. NOTE: do not resolve() symlinks here: in a built layout the package is
# symlinked back into the source tree (which has no pip_prebundle), while the prebundle only exists
# next to the package in the build output. abspath keeps us in the build tree; on_startup below uses
# the extension manager to get the authoritative path.
_module_ext_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_add_prebundle_to_sys_path(_module_ext_root)


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        import omni.kit.app

        ext_root = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)
        if ext_root:
            _add_prebundle_to_sys_path(ext_root)

    def on_shutdown(self) -> None:
        pass
