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

"""Shared Robotics Examples browser registration for cuMotion examples."""

from __future__ import annotations

import gc
from typing import Any

import carb.eventdispatcher
import omni.timeline
import omni.usd
from isaacsim.examples.browser import get_instance as get_browser_instance


class BrowserExampleExtension:
    """Base extension that presents a cuMotion UI builder in Robotics Examples."""

    example_name = ""
    category = "Motion Generation/cuMotion"
    observer_name = ""
    ui_builder_type: type

    def on_startup(self, ext_id: str) -> None:
        """Register the example and initialize its runtime state.

        Args:
            ext_id: The extension ID.
        """
        self._usd_context = omni.usd.get_context()
        self._stage_event_sub_opened = None
        self._stage_event_sub_closed = None
        self._timeline_event_sub_stop = None
        self.ui_builder: Any | None = None

        get_browser_instance().register_example(
            name=self.example_name,
            ui_hook=self._build_ui,
            category=self.category,
        )

    def on_shutdown(self) -> None:
        """Deregister the example and release its runtime state."""
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)
        self._stage_event_sub_opened = None
        self._stage_event_sub_closed = None
        self._timeline_event_sub_stop = None
        if self.ui_builder is not None:
            self.ui_builder.cleanup()
            self.ui_builder = None
        gc.collect()

    def _build_ui(self) -> None:
        if self.ui_builder is not None:
            self.ui_builder.cleanup()
        self.ui_builder = self.ui_builder_type()
        self.ui_builder.build_ui()
        self._subscribe_to_events()

    def _subscribe_to_events(self) -> None:
        if self._stage_event_sub_opened is not None:
            return
        event_dispatcher = carb.eventdispatcher.get_eventdispatcher()
        self._stage_event_sub_opened = event_dispatcher.observe_event(
            event_name=self._usd_context.stage_event_name(omni.usd.StageEventType.OPENED),
            on_event=self._on_stage_changed,
            observer_name=f"{self.observer_name}._on_stage_opened",
        )
        self._stage_event_sub_closed = event_dispatcher.observe_event(
            event_name=self._usd_context.stage_event_name(omni.usd.StageEventType.CLOSED),
            on_event=self._on_stage_changed,
            observer_name=f"{self.observer_name}._on_stage_closed",
        )
        self._timeline_event_sub_stop = event_dispatcher.observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_STOP,
            on_event=self._on_timeline_stop,
            observer_name=f"{self.observer_name}._on_timeline_stop",
        )

    def _on_timeline_stop(self, event: Any) -> None:
        if self.ui_builder is not None:
            self.ui_builder.on_timeline_event(event)

    def _on_stage_changed(self, event: Any) -> None:
        if self.ui_builder is not None:
            self.ui_builder.on_stage_changed(event)
        gc.collect()
