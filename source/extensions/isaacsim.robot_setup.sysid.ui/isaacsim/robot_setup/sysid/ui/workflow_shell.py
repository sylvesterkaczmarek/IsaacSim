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

"""Persistent workflow navigation, readiness, status, and footer controls."""

from __future__ import annotations

from collections.abc import Callable

import omni.ui as ui

from .styles import (
    NAV_BUTTON_STYLE,
    PAGE_DETAIL_STYLE,
    PAGE_TITLE_STYLE,
    SECONDARY_BUTTON_STYLE,
    SPACE_MD,
    SPACE_SM,
    STATUS_BACKGROUND_COLORS,
)
from .workflow_model import WORKFLOW_STAGE_LABELS, WORKFLOW_STAGES, adjacent_stage, normalize_stage


class WorkflowShell:
    """Manage the persistent chrome around the six System Identification stages.

    Args:
        on_stage_changed: Optional callback invoked after the active stage changes.
    """

    def __init__(self, on_stage_changed: Callable[[str], None] | None = None) -> None:
        self._on_stage_changed = on_stage_changed
        self._active_stage = WORKFLOW_STAGES[0]
        self._stage_titles = {
            stage: f"{index}. {WORKFLOW_STAGE_LABELS[stage]}" for index, stage in enumerate(WORKFLOW_STAGES, start=1)
        }
        self._page_frames: dict[str, ui.Frame] = {}
        self._nav_buttons: dict[str, ui.Button] = {}
        self._title_label: ui.Label | None = None
        self._detail_label: ui.Label | None = None
        self._status_background: ui.Rectangle | None = None
        self._status_label: ui.Label | None = None
        self._back_button: ui.Button | None = None
        self._next_button: ui.Button | None = None
        self._action_frame: ui.Frame | None = None
        self._running = False

    @property
    def active_stage(self) -> str:
        """Get the active workflow stage.

        Returns:
            The resulting value.
        """
        return self._active_stage

    def build_header(self) -> None:
        """Build the workflow title and concise purpose statement."""
        with ui.VStack(spacing=2, height=54):
            with ui.HStack(spacing=8, height=22):
                ui.Label("System Identification", style=PAGE_TITLE_STYLE)
                ui.Spacer(width=ui.Fraction(1))
                with ui.VStack(width=46):
                    ui.Spacer(height=2)
                    with ui.ZStack(height=18):
                        ui.Rectangle(style={"background_color": 0x332FB6F6, "border_radius": 9})
                        ui.Label(
                            "Beta",
                            alignment=ui.Alignment.CENTER,
                            tooltip="This extension is in beta; its workflow, run contracts, and APIs may change.",
                            style={"font_size": 11, "color": 0xFF2FB6F6},
                        )
            ui.Label(
                "Calibrate robot dynamics from measured trajectories.",
                style=PAGE_DETAIL_STYLE,
            )

    def build_navigation(self) -> None:
        """Build the six-stage navigation and active-stage summary."""
        with ui.VStack(spacing=SPACE_SM, height=72):
            with ui.HStack(spacing=4, height=30):
                for index, stage in enumerate(WORKFLOW_STAGES, start=1):
                    button = ui.Button(
                        f"{index}  {WORKFLOW_STAGE_LABELS[stage]}",
                        selected=stage == self._active_stage,
                        style=NAV_BUTTON_STYLE,
                        clicked_fn=lambda current=stage: self.activate(current),
                    )
                    self._nav_buttons[stage] = button
            with ui.HStack(spacing=SPACE_SM, height=24):
                self._title_label = ui.Label("", style=PAGE_TITLE_STYLE, width=150)
                self._detail_label = ui.Label("", style=PAGE_DETAIL_STYLE, word_wrap=True)
        self._refresh_summary()

    def build_status(self) -> None:
        """Build the persistent global status banner."""
        with ui.ZStack(height=38):
            self._status_background = ui.Rectangle(
                style={"background_color": STATUS_BACKGROUND_COLORS["info"], "border_radius": 3}
            )
            with ui.HStack(spacing=SPACE_SM):
                ui.Spacer(width=SPACE_MD)
                self._status_label = ui.Label(
                    "Complete Robot, Data, Parameters, and Check to enable Solve.",
                    word_wrap=True,
                    alignment=ui.Alignment.CENTER_TOP,
                )
                ui.Spacer(width=SPACE_MD)

    def build_footer(self) -> ui.Frame:
        """Build Back/Next controls and the primary-action slot.

        Returns:
            The frame reserved for the active stage's primary action.
        """
        with ui.HStack(spacing=SPACE_SM, height=42):
            self._back_button = ui.Button(
                "Back",
                width=88,
                height=30,
                style=SECONDARY_BUTTON_STYLE,
                clicked_fn=lambda: self.activate(adjacent_stage(self._active_stage, -1)),
            )
            self._next_button = ui.Button(
                "Next",
                width=88,
                height=30,
                style=SECONDARY_BUTTON_STYLE,
                clicked_fn=lambda: self.activate(adjacent_stage(self._active_stage, 1)),
            )
            ui.Spacer(width=ui.Fraction(1))
            self._action_frame = ui.Frame(width=0, visible=self._active_stage == "solve")
        self._refresh_controls()
        return self._action_frame

    def bind_pages(self, pages: dict[str, ui.Frame]) -> None:
        """Bind built page frames and apply current stage visibility.

        Args:
            pages: Page frames keyed by workflow stage identifier.
        """
        self._page_frames = dict(pages)
        self.activate(self._active_stage, notify=False)

    def activate(self, stage: str, *, notify: bool = True) -> None:
        """Show one stage and hide the other stage pages.

        Args:
            stage: Workflow stage to activate.
            notify: Whether to invoke the stage-change callback.
        """
        next_stage = normalize_stage(stage)
        self._active_stage = next_stage
        for key, frame in self._page_frames.items():
            frame.visible = key == next_stage
        for key, button in self._nav_buttons.items():
            button.selected = key == next_stage
        self._refresh_summary()
        self._refresh_controls()
        if notify and self._on_stage_changed is not None:
            self._on_stage_changed(next_stage)

    def set_stage_titles(self, titles: dict[str, str]) -> None:
        """Update readiness summaries without rebuilding navigation.

        Args:
            titles: Summary text keyed by workflow stage identifier.
        """
        self._stage_titles.update(titles)
        self._refresh_summary()

    def set_status(self, text: str, severity: str = "info") -> None:
        """Update the global status banner text and severity color.

        Args:
            text: Status message to display.
            severity: Semantic color identifier for the banner.
        """
        if self._status_label is not None:
            self._status_label.text = text
        if self._status_background is not None:
            color = STATUS_BACKGROUND_COLORS.get(severity, STATUS_BACKGROUND_COLORS["info"])
            self._status_background.style = {"background_color": color, "border_radius": 3}

    def set_running(self, running: bool) -> None:
        """Disable manual workflow navigation while a solve is running.

        Args:
            running: Whether a solve is currently active.
        """
        self._running = bool(running)
        for button in self._nav_buttons.values():
            button.enabled = not self._running
        self._refresh_controls()

    def cleanup(self) -> None:
        """Release references to UI widgets."""
        self._page_frames = {}
        self._nav_buttons = {}
        self._title_label = None
        self._detail_label = None
        self._status_background = None
        self._status_label = None
        self._back_button = None
        self._next_button = None
        self._action_frame = None

    def _refresh_summary(self) -> None:
        title = self._stage_titles.get(self._active_stage, "")
        short_title, separator, detail = title.partition(" — ")
        if self._title_label is not None:
            self._title_label.text = short_title
        if self._detail_label is not None:
            self._detail_label.text = detail if separator else ""

    def _refresh_controls(self) -> None:
        index = WORKFLOW_STAGES.index(self._active_stage)
        if self._back_button is not None:
            self._back_button.enabled = not self._running and index > 0
        if self._next_button is not None:
            self._next_button.enabled = not self._running and index < len(WORKFLOW_STAGES) - 1
        if self._action_frame is not None:
            self._action_frame.visible = self._active_stage == "solve" or self._running


__all__ = ["WorkflowShell"]
