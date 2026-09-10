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

"""Tree model for profile validation results."""

from __future__ import annotations

import omni.ui as ui
from omni.asset_validator.core import Issue

from .controller import RequirementResult, RequirementStatus
from .report import location_text, rule_text


class ResultItem(ui.AbstractItem):
    """Base item in the validation result hierarchy.

    Args:
        parent: Parent item in the hierarchy.
    """

    def __init__(self, parent: ResultItem | None = None) -> None:
        super().__init__()
        self.parent = parent
        self.children: list[ResultItem] = []


class FeatureItem(ResultItem):
    """Feature group in the validation result hierarchy.

    Args:
        name: Feature display name.
        results: Requirement results belonging to the feature.
        parent: Parent item in the hierarchy.
    """

    def __init__(self, name: str, results: tuple[RequirementResult, ...], parent: ResultItem | None = None) -> None:
        super().__init__(parent)
        self.name = name
        self.results = results

    @property
    def status_summary(self) -> str:
        """Get a per-status rollup for the feature.

        Returns:
            Counts for each status present in the feature.
        """
        counts: dict[RequirementStatus, int] = {}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return "  ".join(f"{status.value}: {counts[status]}" for status in RequirementStatus if status in counts)

    @property
    def detail_summary(self) -> str:
        """Get requirement and finding totals for the feature.

        Returns:
            Requirement count and finding count text.
        """
        findings = sum(len(result.issues) for result in self.results)
        return f"{len(self.results)} requirement(s), {findings} finding(s)"


class RequirementItem(ResultItem):
    """Requirement row in the validation result hierarchy.

    Args:
        result: Requirement result to display.
        parent: Parent item in the hierarchy.
    """

    def __init__(self, result: RequirementResult, parent: ResultItem | None = None) -> None:
        super().__init__(parent)
        self.result = result


class FindingItem(ResultItem):
    """Individual validator finding in the result hierarchy.

    Args:
        issue: Validator issue to display.
        parent: Parent item in the hierarchy.
    """

    def __init__(self, issue: Issue, parent: ResultItem | None = None) -> None:
        super().__init__(parent)
        self.issue = issue


class ResultsTreeModel(ui.AbstractItemModel):
    """Filterable feature, requirement, and finding tree model."""

    def __init__(self) -> None:
        super().__init__()
        self._roots: list[FeatureItem] = []
        self._results: tuple[RequirementResult, ...] = ()
        self._query = ""
        self._failures_only = False
        self._actionable_only = False

    @property
    def roots(self) -> tuple[FeatureItem, ...]:
        """Get the top-level feature items currently displayed.

        Returns:
            Feature items after filtering.
        """
        return tuple(self._roots)

    def update(self, results: tuple[RequirementResult, ...]) -> None:
        """Replace result data and rebuild the hierarchy.

        Args:
            results: Requirement results to display.
        """
        self._results = results
        self._rebuild()

    def set_filters(self, query: str, failures_only: bool, actionable_only: bool) -> None:
        """Set result filters and rebuild the hierarchy.

        Args:
            query: Case-insensitive search text.
            failures_only: Whether to hide passing requirements.
            actionable_only: Whether to show only requirements with fixes.
        """
        self._query = query.strip().lower()
        self._failures_only = failures_only
        self._actionable_only = actionable_only
        self._rebuild()

    def get_item_children(self, item: ResultItem | None) -> list[ResultItem]:
        """Get child items for a tree node.

        Args:
            item: Parent item, or None for the tree roots.

        Returns:
            Child items for the node.
        """
        return self._roots if item is None else item.children

    def get_item_value_model_count(self, item: ResultItem | None) -> int:
        """Get the number of columns in the tree.

        Args:
            item: Item being queried.

        Returns:
            Column count.
        """
        return 3

    def get_item_value_model(self, item: ResultItem | None, column_id: int) -> ui.AbstractValueModel:
        """Get a text model for a tree cell.

        Args:
            item: Item being rendered.
            column_id: Column being rendered.

        Returns:
            String model holding the cell text.
        """
        if isinstance(item, FeatureItem):
            values = (item.name, item.status_summary, item.detail_summary)
        elif isinstance(item, RequirementItem):
            result = item.result
            label = f"{result.code}: {result.name}" if result.name else result.code
            if result.issues:
                label = f"{label}  ({len(result.issues)} finding(s))"
            status = result.status.value
            if result.actionable:
                status = f"{status} (fixable)"
            values = (label, status, result.rule_name or result.category or "No registered validator")
        elif isinstance(item, FindingItem):
            issue = item.issue
            severity = issue.severity.name.title() if issue.severity is not None else ""
            if issue.suggestions and issue.all_fix_sites:
                severity = f"{severity} (fixable)"
            location = location_text(issue.at) or rule_text(issue) or "Whole asset"
            values = (issue.message or "", severity, location)
        else:
            values = ("Requirement / Finding", "Status", "Rule / Location")
        return ui.SimpleStringModel(values[column_id])

    def _rebuild(self) -> None:
        grouped: dict[str, list[RequirementResult]] = {}
        for result in self._results:
            if self._failures_only and result.status is RequirementStatus.PASS:
                continue
            if self._actionable_only and not result.actionable:
                continue
            haystack = " ".join(
                (
                    result.code,
                    result.name,
                    result.guidance,
                    result.category,
                    result.rule_name,
                    *(issue.message for issue in result.issues),
                )
            ).lower()
            if self._query and self._query not in haystack:
                continue
            grouped.setdefault(result.feature_name, []).append(result)

        roots: list[FeatureItem] = []
        for feature_name, results in grouped.items():
            feature_item = FeatureItem(feature_name, tuple(results))
            for result in results:
                requirement_item = RequirementItem(result, feature_item)
                requirement_item.children = [FindingItem(issue, requirement_item) for issue in result.issues]
                feature_item.children.append(requirement_item)
            roots.append(feature_item)
        self._roots = roots
        self._item_changed(None)
