# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the generated standalone-library documentation catalog."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path


class DocumentationCatalogTests(unittest.TestCase):
    """Validate ordering and language metadata emitted during CMake finalization."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the catalog produced by the CMake tree under test."""
        catalog_path = os.environ.get("ISAACSIM_DOCUMENTATION_CATALOG")
        if catalog_path is None:
            raise unittest.SkipTest("ISAACSIM_DOCUMENTATION_CATALOG is not set")
        cls.catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))

    def test_catalog_collections_have_deterministic_order(self) -> None:
        """Keep every catalog collection sorted and duplicate-free."""
        distributions = self.catalog["distributions"]
        self.assertEqual(
            [item["name"] for item in distributions],
            sorted(item["name"] for item in distributions),
        )
        for distribution in distributions:
            self.assertEqual(distribution["dependencies"], sorted(set(distribution["dependencies"])))
            modules = distribution["modules"]
            self.assertEqual(
                [item["name"] for item in modules],
                sorted(item["name"] for item in modules),
            )
            for module in modules:
                for headers in module["public_headers"].values():
                    self.assertEqual(headers, sorted(set(headers)))
                self.assertEqual(module["python_imports"], sorted(set(module["python_imports"])))

    def test_api_languages_match_authoritative_inputs(self) -> None:
        """Derive exposed languages only from public headers and Python imports."""
        for distribution in self.catalog["distributions"]:
            for module in distribution["modules"]:
                expected = []
                if module["public_headers"]["c"]:
                    expected.append("c")
                if module["public_headers"]["cpp"]:
                    expected.append("cpp")
                if module["python_imports"]:
                    expected.append("python")
                self.assertEqual(module["api_languages"], expected)

    def test_every_distribution_has_an_authored_landing_page(self) -> None:
        """Require package-level documentation for every publishable library."""
        for distribution in self.catalog["distributions"]:
            if distribution["complete"]:
                self.assertTrue(distribution["documentation_root"], distribution["name"])

    def test_robot_schema_exposes_headers_and_python_imports(self) -> None:
        """Keep USD schema module APIs in the documentation catalog."""
        distributions = {item["name"]: item for item in self.catalog["distributions"]}
        robot_schema = distributions["isaacsim_robot_schema"]
        modules = {item["name"]: item for item in robot_schema["modules"]}
        module = modules["usd.schema.isaac"]
        self.assertEqual(module["documentation_root"], "")
        self.assertEqual(
            module["public_headers"]["cpp"],
            [
                "source/libraries/isaacsim_usd_schemas/robot/include/isaacsim/robot/schema/robot_schema.hpp",
                "source/libraries/isaacsim_usd_schemas/robot/include/isaacsim/robot/schema/sensor_tokens.hpp",
                "source/libraries/isaacsim_usd_schemas/robot/include/isaacsim/robot/schema/utils.hpp",
            ],
        )
        self.assertEqual(
            module["python_imports"],
            [
                "omni.isaac.IsaacSensorSchema",
                "omni.isaac.RangeSensorSchema",
                "usd.schema.isaac",
            ],
        )
        self.assertEqual(module["api_languages"], ["cpp", "python"])


if __name__ == "__main__":
    unittest.main()
