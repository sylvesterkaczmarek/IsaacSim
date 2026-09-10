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

"""UI smoke tests for profile validation."""

import omni.kit.ui_test as ui_test
from isaacsim.asset.validation.ui.window import (
    WINDOW_TITLE,
    SimReadyValidationWindow,
    _filtered_profile_ids,
    _pretty_profile_name,
    _profile_name_overrides,
    _requirement_docs_url,
)
from isaacsim.test.utils import MenuUITestCase


class TestSimReadyValidationUI(MenuUITestCase):
    """Verify that the validation workflow controls are available."""

    async def setUp(self) -> None:
        """Open the profile validation window."""
        await super().setUp()
        self.window = SimReadyValidationWindow()
        await ui_test.human_delay(50)

    async def tearDown(self) -> None:
        """Destroy the profile validation window."""
        self.window.destroy()
        await super().tearDown()

    async def test_builds_profile_validation_controls(self) -> None:
        """Build asset, profile, command, and result controls."""
        query = f"{WINDOW_TITLE}//Frame/**"

        self.assertIsNotNone(ui_test.find(query + "/ComboBox[*].identifier=='simready_profile'"))
        self.assertIsNotNone(ui_test.find(query + "/Button[*].identifier=='simready_validate'"))
        self.assertIsNotNone(ui_test.find(query + "/TreeView[*].identifier=='simready_validation_results'"))
        self.assertIsNotNone(ui_test.find(query + "/Button[*].identifier=='simready_export_report'"))
        self.assertIn("Robot-Body-Isaac", self.window._resolver.profile_ids)
        self.assertEqual(
            {"Robot-Body-Isaac", "Prop-Robotics-Isaac"},
            set(self.window._profile_ids),
        )

    async def test_formats_and_filters_profile_names(self) -> None:
        """Display readable labels while preserving configured profile IDs."""
        overrides = _profile_name_overrides({"Robot-Body-Isaac": "Robot Assets", "Prop-Robotics-Isaac": "Props"})
        self.assertEqual("Prop Robotics (Isaac)", _pretty_profile_name("Prop-Robotics-Isaac"))
        self.assertEqual("Robot Assets", _pretty_profile_name("Robot-Body-Isaac", overrides))
        self.assertEqual("Props", _pretty_profile_name("Prop-Robotics-Isaac", overrides))
        self.assertEqual(
            ("Robot-Body-Isaac",),
            _filtered_profile_ids(
                ("Robot-Body-Isaac", "Prop-Robotics-Isaac"),
                ["Prop-Robotics-Isaac", "Missing-Profile"],
            ),
        )
        self.assertEqual(
            ("Robot-Body-Isaac", "Prop-Robotics-Isaac"),
            _filtered_profile_ids(("Robot-Body-Isaac", "Prop-Robotics-Isaac"), []),
        )

    async def test_requirement_docs_url_includes_capabilities_root(self) -> None:
        """Insert the published capabilities root into requirement documentation URLs."""
        base = "https://docs.omniverse.nvidia.com/kit/docs/asset-requirements/latest"

        self.assertEqual(
            f"{base}/capabilities/core/atomic_asset/requirements/supported-file-types.html",
            _requirement_docs_url("core/atomic_asset/requirements/supported-file-types.html", f"{base}/"),
        )
        self.assertEqual(
            f"{base}/capabilities/core/units/requirements/up-axis-z.html",
            _requirement_docs_url("capabilities/core/units/requirements/up-axis-z.html", base),
        )
        self.assertEqual(base, _requirement_docs_url("", base))
