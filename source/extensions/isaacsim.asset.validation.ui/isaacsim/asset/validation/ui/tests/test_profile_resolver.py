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

"""Tests for bundled SimReady profile resolution."""

import omni.kit.test
from isaacsim.asset.validation.ui.profile_resolver import ProfileResolver


class TestProfileResolver(omni.kit.test.AsyncTestCase):
    """Verify profile discovery and dependency expansion."""

    async def setUp(self) -> None:
        """Create a resolver for each test."""
        self.resolver = ProfileResolver()

    async def test_discovers_profiles_and_sorts_versions(self) -> None:
        """Discover bundled profiles and return newest versions first."""
        self.assertIn("Robot-Body-Isaac", self.resolver.profile_ids)
        self.assertEqual(("1.1.0", "1.0.0"), self.resolver.versions("Robot-Body-Isaac"))

    async def test_resolves_cross_tier_profile_requirements(self) -> None:
        """Resolve core and Isaac feature requirements in one profile."""
        profile = self.resolver.resolve("Robot-Body-Isaac", "1.1.0")
        feature_ids = [feature.definition.feature_id for feature in profile.features]
        requirement_codes = {
            requirement.code.removeprefix("com.nvidia.simready.") for requirement in profile.requirements
        }

        self.assertIn("FET001_BASE_NEUTRAL", feature_ids)
        self.assertIn("FET021_ROBOT_CORE_ISAAC", feature_ids)
        self.assertIn("FET022_DRIVEN_JOINTS_PHYSX", feature_ids)
        self.assertIn("RC.001", requirement_codes)
        self.assertIn("DJ.008", requirement_codes)

    async def test_preserves_optional_profile_features(self) -> None:
        """Keep explicitly optional profile features optional."""
        profile = self.resolver.resolve("Prop-Robotics-Isaac", "1.0.0")
        optional = {feature.definition.feature_id for feature in profile.features if feature.optional}

        self.assertIn("FET004_BASE_PHYSX", optional)

    async def test_required_reference_overrides_optional_dependency(self) -> None:
        """Treat a feature as required when any profile reference requires it."""
        profile = self.resolver.resolve("Prop-Robotics-Isaac", "1.0.1")
        selected = {feature.definition.feature_id: feature.optional for feature in profile.features}

        self.assertFalse(selected["FET004_BASE_PHYSX"])
