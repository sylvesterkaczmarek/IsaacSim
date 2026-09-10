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

"""Tests for the profile validation controller."""

import omni.kit.test
from isaacsim.asset.validation.ui.controller import RequirementStatus, ValidationController
from isaacsim.asset.validation.ui.profile_resolver import (
    FeatureDefinition,
    ProfileDefinition,
    ResolvedFeature,
    ResolvedProfile,
)
from omni.asset_validator.core import Issue, IssueSeverity, RequirementsRegistry
from pxr import Usd


def _make_profile(requirement_code: str) -> ResolvedProfile:
    registry = RequirementsRegistry()
    requirement = registry.find_requirement(requirement_code) or registry.find_requirement(
        f"com.nvidia.simready.{requirement_code}"
    )
    if requirement is None:
        raise RuntimeError(f"Requirement {requirement_code} was not registered")
    feature = FeatureDefinition(
        feature_id="TEST_FEATURE",
        version="1.0.0",
        display_name="Test Feature",
        path="",
        requirements=(requirement_code,),
        dependencies=(),
    )
    return ResolvedProfile(
        definition=ProfileDefinition(
            profile_id="Test-Profile",
            version="1.0.0",
            features=(),
        ),
        features=(
            ResolvedFeature(
                definition=feature,
                optional=False,
                requirements=(requirement,),
                missing_requirements=(),
            ),
        ),
    )


class TestValidationController(omni.kit.test.AsyncTestCase):
    """Verify status synthesis and guarded mutations."""

    async def setUp(self) -> None:
        """Create a controller for each test."""
        self.controller = ValidationController()

    async def tearDown(self) -> None:
        """Release controller tasks and listeners."""
        self.controller.cleanup()

    async def test_synthesizes_failure_and_pass_results(self) -> None:
        """Synthesize pass rows and preserve emitted failures."""
        profile = _make_profile("RC.001")
        requirement = profile.requirements[0]
        self.controller.set_profile(profile)

        failure = Issue(
            message="Folder contains an unexpected file.", requirement=requirement, severity=IssueSeverity.FAILURE
        )
        failed = self.controller._build_results((failure,))
        passed = self.controller._build_results(())

        self.assertEqual(RequirementStatus.FAIL, failed[0].status)
        self.assertEqual(RequirementStatus.PASS, passed[0].status)
        self.assertEqual("CleanFolder", failed[0].rule_name)
        self.assertEqual("RobotCore", failed[0].category)

    async def test_file_asset_cannot_apply_stage_fix(self) -> None:
        """Reject fix requests for assets not open as the current stage."""
        self.controller.set_asset("/tmp/example.usd")

        with self.assertRaisesRegex(RuntimeError, "current stage"):
            await self.controller.apply_fix_async(Issue(message="No fix", severity=IssueSeverity.FAILURE))

    async def test_validates_an_in_memory_stage_by_requirement(self) -> None:
        """Run one registered requirement against an in-memory stage."""
        self.controller.set_profile(_make_profile("UN.006"))
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        self.controller.set_asset(stage)

        await self.controller.validate_async()

        self.assertFalse(self.controller.running)
        self.assertEqual(1.0, self.controller.progress)
        self.assertEqual(1, len(self.controller.results))
        self.assertEqual("UN.006", self.controller.results[0].code)
