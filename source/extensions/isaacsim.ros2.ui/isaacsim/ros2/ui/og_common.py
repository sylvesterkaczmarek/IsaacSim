# SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Common OmniGraph shortcut UI helpers."""

from collections.abc import Callable

import isaacsim.core.experimental.utils.prim as prim_utils
import omni.ui as ui
import OmniGraphSchema
from isaacsim.gui.components.callbacks import on_docs_link_clicked, on_open_IDE_clicked
from isaacsim.gui.components.style import get_style
from omni.kit.menu.utils import MenuHelperWindow
from omni.kit.notification_manager import NotificationStatus, post_notification


def add_ok_cancel_buttons(on_ok: Callable[[], None], on_cancel: Callable[[], None]) -> None:
    with ui.HStack():
        ui.Spacer(width=ui.Percent(10))
        ui.Button("OK", height=40, width=ui.Percent(30), clicked_fn=on_ok)
        ui.Spacer(width=ui.Percent(20))
        ui.Button("Cancel", height=40, width=ui.Percent(30), clicked_fn=on_cancel)
        ui.Spacer(width=ui.Percent(10))


def add_script_docs_footer(source_file: str, docs_url: str) -> None:
    with ui.Frame(height=30):
        with ui.VStack():
            with ui.HStack():
                ui.Label("Python Script for Graph Generation", width=ui.Percent(30))
                ui.Button(
                    name="IconButton",
                    width=24,
                    height=24,
                    clicked_fn=lambda: on_open_IDE_clicked("", source_file),
                    style=get_style()["IconButton.Image::OpenConfig"],
                )
            with ui.HStack():
                ui.Label("Documentations", width=0, word_wrap=True)
                ui.Button(
                    name="IconButton",
                    width=24,
                    height=24,
                    clicked_fn=lambda: on_docs_link_clicked(docs_url),
                    style=get_style()["IconButton.Image::OpenLink"],
                )


def validate_new_graph_path(graph_path: str, graph_label: str = "graph") -> bool:
    og_prim = prim_utils.get_prim_at_path(graph_path)
    if og_prim.IsValid() and og_prim.IsA(OmniGraphSchema.OmniGraph):
        post_notification(
            f"{graph_path} already exists. Delete the existing {graph_label} or change the graph path",
            status=NotificationStatus.WARNING,
        )
        return False
    return True


def validate_existing_graph_path(graph_path: str) -> bool:
    og_prim = prim_utils.get_prim_at_path(graph_path)
    if og_prim.IsValid() and og_prim.IsA(OmniGraphSchema.OmniGraph):
        return True
    post_notification(f"{graph_path} is not an existing graph, check the og path", status=NotificationStatus.WARNING)
    return False


class Ros2GraphWindow(MenuHelperWindow):
    """Base class for ROS 2 graph shortcut dialogs."""

    def _finish_on_ok(self, make_graph_fn: Callable[[], None] | None = None) -> None:
        if self._check_params():
            (make_graph_fn or self.make_graph)()
            self.visible = False
        else:
            post_notification("Parameter check failed", status=NotificationStatus.WARNING)

    def _on_cancel(self) -> None:
        self.visible = False

    def _on_use_existing_graph(self, check_state: bool) -> None:
        self._add_to_existing_graph = check_state
