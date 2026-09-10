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

"""Extension lifecycle for the optional Python Server UI."""

import carb
import omni.ext

from .indicator import PythonServerUi


class Extension(omni.ext.IExt):
    """Own the Python Server status indicator in UI-capable applications."""

    def on_startup(self, _ext_id: str) -> None:
        """Create the indicator when application UI is visible."""
        self._ui = None
        if not bool(carb.settings.get_settings().get("/app/window/hideUi")):
            self._ui = PythonServerUi()

    def on_shutdown(self) -> None:
        """Destroy the indicator without affecting the Python server."""
        if self._ui is not None:
            self._ui.destroy()
            self._ui = None
