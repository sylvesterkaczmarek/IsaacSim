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

"""Extension entry point for SimReady profile validation."""

from __future__ import annotations

import omni.ext
import omni.ui as ui
from omni.kit.menu.utils import MenuHelperExtension

from .window import WINDOW_TITLE, SimReadyValidationWindow


class SimReadyValidationUIExtension(omni.ext.IExt, MenuHelperExtension):
    """Extension providing the SimReady Asset Validation window."""

    def on_startup(self, ext_id: str) -> None:
        """Register the validation window and Window menu item.

        Args:
            ext_id: Extension identifier provided by the extension manager.
        """
        self._window: SimReadyValidationWindow | None = None
        ui.Workspace.set_show_window_fn(WINDOW_TITLE, self.show_window)
        self.menu_startup(WINDOW_TITLE, WINDOW_TITLE, "Window")

    def on_shutdown(self) -> None:
        """Destroy the validation window and remove its menu item."""
        self.menu_shutdown()
        if self._window is not None:
            self._window.destroy()
            self._window = None
        ui.Workspace.set_show_window_fn(WINDOW_TITLE, None)

    def show_window(self, visible: bool) -> None:
        """Show or hide the validation window.

        Args:
            visible: Whether the validation window should be visible.
        """
        if visible:
            if self._window is None:
                self._window = SimReadyValidationWindow()
            else:
                self._window.visible = True
        elif self._window is not None:
            self._window.visible = False
