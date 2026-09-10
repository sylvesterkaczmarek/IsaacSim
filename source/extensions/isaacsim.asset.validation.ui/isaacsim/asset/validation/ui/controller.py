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

"""Control profile-scoped validation independently from the user interface."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import omni.kit.app
from omni.asset_validator.core import (
    AssetProgress,
    FixResult,
    Issue,
    IssueFixer,
    IssueSeverity,
    Requirement,
    RequirementsRegistry,
    Results,
    ValidationEngine,
    ValidationRulesRegistry,
)
from pxr import Usd

from .profile_resolver import ResolvedFeature, ResolvedProfile

AssetType = str | Usd.Stage
ChangeCallback = Callable[[], None]


class RequirementStatus(Enum):
    """User-facing status for a selected requirement."""

    PASS = "Pass"
    FAIL = "Fail"
    WARNING = "Warning"
    ERROR = "Error"
    NOT_IMPLEMENTED = "Not implemented"


@dataclass(frozen=True)
class RequirementResult:
    """Validation result for one requirement.

    Args:
        feature_id: Feature that selected the requirement.
        feature_name: User-facing feature name.
        optional_feature: Whether the containing feature is optional.
        code: Requirement code.
        name: User-facing requirement name.
        guidance: Requirement-level guidance.
        documentation_path: Relative requirement documentation path.
        category: Registered Asset Validator category.
        rule_name: Registered validator class name.
        status: Aggregated requirement status.
        issues: Findings emitted for the requirement.
    """

    feature_id: str
    feature_name: str
    optional_feature: bool
    code: str
    name: str
    guidance: str
    documentation_path: str
    category: str
    rule_name: str
    status: RequirementStatus
    issues: tuple[Issue, ...]

    @property
    def actionable(self) -> bool:
        """Check whether any finding supplies a valid fix.

        Returns:
            True when a fix suggestion and edit target are available.
        """
        return any(issue.suggestions and issue.all_fix_sites for issue in self.issues)


class ValidationController:
    """Async controller for one profile validation workflow."""

    def __init__(self) -> None:
        self.asset: AssetType | None = None
        self.profile: ResolvedProfile | None = None
        self.selected_optional_features: set[str] = set()
        self.progress = 0.0
        self.running = False
        self.results: tuple[RequirementResult, ...] = ()
        self.error_message = ""
        self._task: asyncio.Task | None = None
        self._callbacks: list[ChangeCallback] = []

    def subscribe(self, callback: ChangeCallback) -> Callable[[], None]:
        """Subscribe to controller state changes.

        Args:
            callback: Function invoked after state changes.

        Returns:
            Function that removes the subscription.
        """
        self._callbacks.append(callback)

        def unsubscribe() -> None:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

        return unsubscribe

    def set_profile(self, profile: ResolvedProfile) -> None:
        """Set the active profile and reset optional selections.

        Args:
            profile: Resolved profile to validate.
        """
        self.profile = profile
        self.selected_optional_features = set()
        self.results = ()
        self.error_message = ""
        self._notify()

    def set_optional_feature_enabled(self, feature_id: str, enabled: bool) -> None:
        """Enable or disable an optional profile feature.

        Args:
            feature_id: Optional feature identifier.
            enabled: Whether the feature participates in validation.
        """
        if enabled:
            self.selected_optional_features.add(feature_id)
        else:
            self.selected_optional_features.discard(feature_id)
        self.results = ()
        self._notify()

    def set_asset(self, asset: AssetType | None) -> None:
        """Set the stage or URI to validate.

        Args:
            asset: Current stage, asset URI, or None.
        """
        self.asset = asset
        self.results = ()
        self.error_message = ""
        self._notify()

    @property
    def selected_features(self) -> tuple[ResolvedFeature, ...]:
        """Get mandatory and enabled optional features.

        Returns:
            Selected features in dependency order.
        """
        if self.profile is None:
            return ()
        return tuple(
            feature
            for feature in self.profile.features
            if not feature.optional or feature.definition.feature_id in self.selected_optional_features
        )

    @property
    def can_validate(self) -> bool:
        """Check whether validation can start.

        Returns:
            True when an asset and profile are selected and no run is active.
        """
        return self.asset is not None and self.profile is not None and not self.running

    def start_validation(self) -> asyncio.Task:
        """Start profile validation.

        Returns:
            Task representing the validation operation.

        Raises:
            RuntimeError: If the controller is not ready to validate.
        """
        if not self.can_validate:
            raise RuntimeError("Select an asset and SimReady profile before validating")
        self._task = asyncio.ensure_future(self.validate_async())
        return self._task

    async def validate_async(self) -> None:
        """Validate the selected asset against selected profile requirements."""
        if self.asset is None or self.profile is None:
            raise RuntimeError("Select an asset and SimReady profile before validating")

        self.running = True
        self.progress = 0.0
        self.results = ()
        self.error_message = ""
        self._notify()

        registry = RequirementsRegistry()
        requirements = self._selected_requirements()
        engine = ValidationEngine(init_rules=False)
        for requirement in requirements:
            if registry.is_implemented(requirement):
                engine.enable_requirement(requirement)

        validation_results: Results | None = None

        def on_progress(progress: AssetProgress) -> None:
            self.progress = progress.progress
            self._notify()

        def on_validated(results: Results) -> None:
            nonlocal validation_results
            validation_results = results

        try:
            await engine.validate_with_callbacks(
                asset=self.asset,
                asset_progress_fn=on_progress,
                asset_validated_fn=on_validated,
            )
            issues = tuple(validation_results.issues) if validation_results is not None else ()
            self.results = self._build_results(issues)
            self.progress = 1.0
        except asyncio.CancelledError:
            self.error_message = "Validation cancelled."
            raise
        except Exception as exc:
            self.error_message = str(exc)
        finally:
            self.running = False
            self._task = None
            self._notify()

    def cancel(self) -> None:
        """Cancel the active validation task."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def apply_fix_async(self, issue: Issue) -> FixResult:
        """Apply one validator-provided fix to the current stage.

        Args:
            issue: Finding whose selected suggestion should be applied.

        Returns:
            Result returned by the validator fix framework.

        Raises:
            RuntimeError: If the asset is not the current stage or has no valid fix.
        """
        if not isinstance(self.asset, Usd.Stage):
            raise RuntimeError("Open the asset as the current stage before applying a fix")
        if not issue.suggestions or not issue.all_fix_sites:
            raise RuntimeError("This finding does not provide an applicable fix")
        fixer = IssueFixer(self.asset)
        result = fixer.apply(issue, issue.default_fix_site, suggestion=issue.suggestion)
        await omni.kit.app.get_app().next_update_async()
        await self.validate_async()
        return result

    def cleanup(self) -> None:
        """Cancel work and remove all state listeners."""
        self.cancel()
        self._callbacks.clear()

    def _selected_requirements(self) -> tuple[Requirement, ...]:
        requirements: dict[tuple[str, object], Requirement] = {}
        for feature in self.selected_features:
            for requirement in feature.requirements:
                requirements.setdefault((requirement.code, requirement.version), requirement)
        return tuple(requirements.values())

    def _build_results(self, issues: tuple[Issue, ...]) -> tuple[RequirementResult, ...]:
        registry = RequirementsRegistry()
        results: list[RequirementResult] = []
        emitted_codes: set[str] = set()
        for feature in self.selected_features:
            for requirement in feature.requirements:
                if requirement.code in emitted_codes:
                    continue
                emitted_codes.add(requirement.code)
                validator = registry.get_validator(requirement)
                requirement_issues = tuple(
                    issue
                    for issue in issues
                    if issue.requirement is not None and issue.requirement.code == requirement.code
                )
                status = self._compute_status(requirement, requirement_issues)
                results.append(
                    RequirementResult(
                        feature_id=feature.definition.feature_id,
                        feature_name=feature.definition.display_name,
                        optional_feature=feature.optional,
                        code=requirement.code.removeprefix("com.nvidia.simready."),
                        name=requirement.display_name,
                        guidance=requirement.message,
                        documentation_path=requirement.path,
                        category=ValidationRulesRegistry.category(validator) if validator else "",
                        rule_name=validator.__name__ if validator else "",
                        status=status,
                        issues=requirement_issues,
                    )
                )
            for code in feature.missing_requirements:
                if code in emitted_codes:
                    continue
                emitted_codes.add(code)
                results.append(
                    RequirementResult(
                        feature_id=feature.definition.feature_id,
                        feature_name=feature.definition.display_name,
                        optional_feature=feature.optional,
                        code=code,
                        name=code,
                        guidance="No registered requirement metadata or validator is available.",
                        documentation_path="",
                        category="",
                        rule_name="",
                        status=RequirementStatus.NOT_IMPLEMENTED,
                        issues=(),
                    )
                )
        return tuple(results)

    @staticmethod
    def _compute_status(requirement: Requirement, issues: tuple[Issue, ...]) -> RequirementStatus:
        registry = RequirementsRegistry()
        if not registry.is_implemented(requirement):
            return RequirementStatus.NOT_IMPLEMENTED
        severities = {issue.severity for issue in issues}
        if IssueSeverity.ERROR in severities:
            return RequirementStatus.ERROR
        if IssueSeverity.FAILURE in severities:
            return RequirementStatus.FAIL
        if IssueSeverity.WARNING in severities:
            return RequirementStatus.WARNING
        return RequirementStatus.PASS

    def _notify(self) -> None:
        for callback in tuple(self._callbacks):
            callback()
