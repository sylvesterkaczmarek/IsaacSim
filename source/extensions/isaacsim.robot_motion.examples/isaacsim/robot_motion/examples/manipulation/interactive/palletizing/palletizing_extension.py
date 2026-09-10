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

"""Robotics Examples browser registration for the palletizing sample."""

from __future__ import annotations

import asyncio
import os

import omni.ext
import omni.ui as ui
from isaacsim.examples.base import BaseSampleUITemplate
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.gui.components import btn_builder
from isaacsim.robot_motion.examples.manipulation.interactive.palletizing.palletizing import Palletizing


class PalletizingExtension(omni.ext.IExt):
    """Extension lifecycle for the palletizing sample."""

    def on_startup(self, ext_id: str) -> None:
        """Register the sample with the Robotics Examples browser.

        Args:
            ext_id: Extension identifier provided by Kit.
        """

        self.example_name = "UR10 Palletizing"
        self.category = "Manipulation"
        ui_handle = PalletizingUI(
            ext_id=ext_id,
            file_path=os.path.abspath(__file__),
            title=self.example_name,
            overview=(
                "Use a UR10 with cuMotion RMPflow to pick bins from a conveyor, flip inverted bins, "
                "and stack them on a pallet."
            ),
            sample=Palletizing(),
        )
        get_browser_instance().register_example(
            name=self.example_name,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

    def on_shutdown(self) -> None:
        """Deregister the sample from the Robotics Examples browser."""

        get_browser_instance().deregister_example(name=self.example_name, category=self.category)


class PalletizingUI(BaseSampleUITemplate):
    """UI for the palletizing sample."""

    def build_extra_frames(self) -> None:
        """Build the palletizing controls."""

        with self.get_extra_frames_handle():
            with ui.CollapsableFrame(title="Task Control", collapsed=False):
                self.task_ui_elements = {
                    "Start Palletizing": btn_builder(
                        label="Start Palletizing",
                        type="button",
                        text="START",
                        tooltip="Start palletizing",
                        on_clicked_fn=self._start,
                    )
                }
                self.task_ui_elements["Start Palletizing"].enabled = False

    def post_load_button_event(self) -> None:
        """Enable the task after loading."""

        self.task_ui_elements["Start Palletizing"].enabled = True

    def post_reset_button_event(self) -> None:
        """Enable the task after resetting."""

        self.task_ui_elements["Start Palletizing"].enabled = True

    def post_clear_button_event(self) -> None:
        """Disable the task after clearing."""

        self.task_ui_elements["Start Palletizing"].enabled = False

    def _start(self) -> None:
        self.task_ui_elements["Start Palletizing"].enabled = False
        asyncio.ensure_future(self.sample.start_palletizing_async())
