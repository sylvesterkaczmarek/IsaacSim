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

"""Render profile validation result rows."""

from __future__ import annotations

import omni.ui as ui
from omni.asset_validator.core import IssueSeverity

from .controller import RequirementStatus
from .result_model import FeatureItem, FindingItem, RequirementItem, ResultItem

_STATUS_COLORS = {
    RequirementStatus.PASS: 0xFF70C070,
    RequirementStatus.FAIL: 0xFF6060E0,
    RequirementStatus.WARNING: 0xFF50B0E0,
    RequirementStatus.ERROR: 0xFF4040FF,
    RequirementStatus.NOT_IMPLEMENTED: 0xFF909090,
}

_SEVERITY_COLORS = {
    IssueSeverity.ERROR: 0xFF4040FF,
    IssueSeverity.FAILURE: 0xFF6060E0,
    IssueSeverity.WARNING: 0xFF50B0E0,
    IssueSeverity.INFO: 0xFFB0B0B0,
    IssueSeverity.NONE: 0xFF909090,
}


class ResultsTreeDelegate(ui.AbstractItemDelegate):
    """Delegate that emphasizes status and hierarchy in validation results."""

    def build_header(self, column_id: int = 0) -> None:
        """Build a result table header.

        Args:
            column_id: Column being rendered.
        """
        ui.Label(("Requirement / Finding", "Status", "Rule / Location")[column_id])

    def build_branch(
        self,
        model: ui.AbstractItemModel,
        item: ResultItem,
        column_id: int,
        level: int,
        expanded: bool,
    ) -> None:
        """Build the expand and collapse control for a tree row.

        Args:
            model: Tree item model.
            item: Item being rendered.
            column_id: Column being rendered.
            level: Tree indentation level.
            expanded: Whether the item is expanded.
        """
        if column_id != 0:
            return
        with ui.HStack(width=0, height=20):
            ui.Spacer(width=12 * level)
            if model.get_item_children(item):
                with ui.VStack(width=14):
                    ui.Spacer()
                    ui.Triangle(
                        width=8,
                        height=8,
                        alignment=ui.Alignment.CENTER_BOTTOM if expanded else ui.Alignment.RIGHT_CENTER,
                        style={"background_color": 0xFFCCCCCC},
                    )
                    ui.Spacer()
            else:
                ui.Spacer(width=14)
            ui.Spacer(width=4)

    def build_widget(
        self,
        model: ui.AbstractItemModel,
        item: ResultItem,
        column_id: int,
        level: int,
        expanded: bool,
    ) -> None:
        """Build one result tree cell.

        Args:
            model: Tree item model.
            item: Item being rendered.
            column_id: Column being rendered.
            level: Tree indentation level.
            expanded: Whether the item is expanded.
        """
        value = model.get_item_value_model(item, column_id).get_value_as_string()
        style: dict[str, object] = {}
        if isinstance(item, FeatureItem):
            style["font_size"] = 15
            if column_id != 0:
                style["color"] = 0xFFAAAAAA
        elif isinstance(item, RequirementItem):
            if column_id == 1:
                style["color"] = _STATUS_COLORS[item.result.status]
            elif column_id == 2:
                style["color"] = 0xFFAAAAAA
        elif isinstance(item, FindingItem):
            style["color"] = _SEVERITY_COLORS.get(item.issue.severity, 0xFFB0B0B0) if column_id == 1 else 0xFFB0B0B0
        ui.Label(value, height=20, elided_text=True, tooltip=value, style=style)
