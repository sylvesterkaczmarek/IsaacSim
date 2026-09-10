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

import os
import sys

import omni.ext


def _add_prebundle_to_sys_path(ext_root: str) -> None:
    """Make the bundled ONNX Runtime package discoverable on ``sys.path``.

    Args:
        ext_root: Root directory of the staged extension.
    """
    pip_prebundle = os.path.join(ext_root, "pip_prebundle")
    if os.path.isdir(pip_prebundle) and pip_prebundle not in sys.path:
        sys.path.insert(0, pip_prebundle)


# Preserve the build-tree path. Resolving the package symlink would point into the source tree,
# while pip_prebundle exists beside the staged extension in the build output.
_module_ext_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_add_prebundle_to_sys_path(_module_ext_root)


class Extension(omni.ext.IExt):
    """Make the staged ONNX prebundle available to Python imports."""

    def on_startup(self, ext_id: str) -> None:
        """Add this extension's staged prebundle to ``sys.path``.

        Args:
            ext_id: Identifier of the extension being started.
        """
        import omni.kit.app

        ext_root = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)
        if ext_root:
            _add_prebundle_to_sys_path(ext_root)

    def on_shutdown(self) -> None:
        """Shut down without removing the process-wide archive path."""
