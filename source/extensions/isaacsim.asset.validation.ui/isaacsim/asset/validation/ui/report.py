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

"""Serialize profile validation results into a JSON report."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omni.asset_validator.core import Issue

from .controller import RequirementResult, RequirementStatus
from .profile_resolver import ResolvedProfile


def location_text(at: Any) -> str:
    """Format an issue location into a readable string.

    Args:
        at: Location reported by a validator finding.

    Returns:
        Comma-separated location text, or an empty string when unlocated.
    """
    if at is None:
        return ""
    if isinstance(at, (list, tuple)):
        return ", ".join(text for text in (location_text(entry) for entry in at) if text)
    path = getattr(at, "path", None)
    if path is not None:
        return str(path)
    get_path = getattr(at, "GetPath", None)
    if callable(get_path):
        return str(get_path())
    identifier = getattr(at, "identifier", None)
    if identifier is not None:
        return str(identifier)
    return str(at)


def rule_text(issue: Issue) -> str:
    """Get the validator class name that emitted a finding.

    Args:
        issue: Validator finding.

    Returns:
        Rule class name, or an empty string when unavailable.
    """
    rule = getattr(issue, "rule", None)
    return getattr(rule, "__name__", "") if rule is not None else ""


def issue_entry(issue: Issue) -> dict[str, Any]:
    """Convert one validator finding into report data.

    Args:
        issue: Validator finding.

    Returns:
        JSON-serializable finding data.
    """
    return {
        "severity": issue.severity.name.title() if issue.severity is not None else "",
        "message": issue.message or "",
        "location": location_text(issue.at),
        "rule": rule_text(issue),
        "code": issue.code or "",
        "suggestion": issue.suggestion.message if issue.suggestions else "",
        "fixable": bool(issue.suggestions and issue.all_fix_sites),
    }


def requirement_entry(result: RequirementResult) -> dict[str, Any]:
    """Convert one requirement result into report data.

    Args:
        result: Requirement result to serialize.

    Returns:
        JSON-serializable requirement data.
    """
    return {
        "feature_id": result.feature_id,
        "feature_name": result.feature_name,
        "optional_feature": result.optional_feature,
        "code": result.code,
        "name": result.name,
        "status": result.status.value,
        "category": result.category,
        "rule": result.rule_name,
        "guidance": result.guidance,
        "documentation_path": result.documentation_path,
        "actionable": result.actionable,
        "findings": [issue_entry(issue) for issue in result.issues],
    }


def build_report(
    results: Sequence[RequirementResult],
    profile: ResolvedProfile | None,
    asset: str,
    selected_features: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a full validation report.

    Args:
        results: Requirement results to serialize.
        profile: Resolved profile used for the run.
        asset: Asset identifier or stage root layer path.
        selected_features: Feature identifiers included in the run.

    Returns:
        JSON-serializable report data.
    """
    counts = {status.value: 0 for status in RequirementStatus}
    for result in results:
        counts[result.status.value] += 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "asset": asset,
        "profile": {
            "id": profile.definition.profile_id if profile else "",
            "version": profile.definition.version if profile else "",
        },
        "selected_features": list(selected_features),
        "summary": {
            **counts,
            "requirements": len(results),
            "findings": sum(len(result.issues) for result in results),
        },
        "requirements": [requirement_entry(result) for result in results],
    }


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Write a validation report to disk as JSON.

    Args:
        path: Destination file path.
        report: Report data produced by :func:`build_report`.

    Returns:
        Path that was written.
    """
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        destination = destination.with_suffix(".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination
