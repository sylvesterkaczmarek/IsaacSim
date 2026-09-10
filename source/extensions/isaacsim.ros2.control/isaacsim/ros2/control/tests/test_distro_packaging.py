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

"""Packaging and active-distro load checks for ros2_control backends."""

from __future__ import annotations

import os
from pathlib import Path
from xml.etree import ElementTree

import omni.kit.app
import omni.kit.test
from isaacsim.ros2.control.bindings import _isaacsim_ros2_control as _backend
from isaacsim.ros2.control.tests._runtime_layout import (
    IS_WINDOWS,
    LIBRARY_PATH_ENV,
    backend_library_name,
    env_paths,
    loaded_library_paths,
    path_key,
)

_EXTENSION_NAME = "isaacsim.ros2.control"
_PACKAGE_NAME = "isaacsim_ros2_control"
_PLUGIN_CLASS_NAME = "isaacsim_ros2_control/IsaacSimSystem"
_PLUGIN_CLASS_TYPE = "isaacsim::ros2::control::backend::IsaacSimSystem"
_PLUGIN_BASE_CLASS_TYPE = "hardware_interface::SystemInterface"
_SUPPORTED_DISTROS = ("humble", "jazzy")
_PLUGIN_XML_RELATIVE_PATH = Path("share") / _PACKAGE_NAME / "isaac_sim_system_plugin.xml"


def _get_extension_path() -> Path:
    manager = omni.kit.app.get_app().get_extension_manager()
    ext_id = manager.get_enabled_extension_id(_EXTENSION_NAME)
    if ext_id:
        return Path(manager.get_extension_path(ext_id))

    module_path = manager.get_extension_path_by_module(__name__)
    if module_path:
        return Path(module_path)

    return Path(__file__).resolve().parents[4]


class TestRos2ControlDistroLoading(omni.kit.test.AsyncTestCase):
    """Verify distro payloads are isolated and the active distro backend loads."""

    async def setUp(self) -> None:
        self._extension_path = _get_extension_path()

    def _assert_distro_payload(self, ros_distro: str) -> None:
        distro_prefix = self._extension_path / ros_distro
        plugin_xml_path = distro_prefix / _PLUGIN_XML_RELATIVE_PATH
        self.assertTrue(
            plugin_xml_path.is_file(),
            f"{ros_distro}: missing pluginlib XML at {plugin_xml_path}",
        )

        plugin_xml = ElementTree.parse(plugin_xml_path).getroot()
        self.assertEqual(plugin_xml.tag, "library", f"{ros_distro}: plugin XML root must be <library>")
        self.assertEqual(
            plugin_xml.get("path"),
            f"isaacsim.ros2.control.{ros_distro}",
            f"{ros_distro}: plugin XML points at the wrong backend library",
        )
        plugin_class = None
        for candidate in plugin_xml.findall("class"):
            if candidate.get("name") == _PLUGIN_CLASS_NAME:
                plugin_class = candidate
                break
        self.assertIsNotNone(plugin_class, f"{ros_distro}: plugin XML does not declare {_PLUGIN_CLASS_NAME}")
        self.assertEqual(
            plugin_class.get("type"),
            _PLUGIN_CLASS_TYPE,
            f"{ros_distro}: plugin XML declares the wrong implementation type",
        )
        self.assertEqual(
            plugin_class.get("base_class_type"),
            _PLUGIN_BASE_CLASS_TYPE,
            f"{ros_distro}: plugin XML declares the wrong base class type",
        )

        package_xml_path = plugin_xml_path.parent / "package.xml"
        self.assertTrue(
            package_xml_path.is_file(),
            f"{ros_distro}: missing package.xml beside pluginlib XML at {package_xml_path}",
        )
        package_xml = ElementTree.parse(package_xml_path).getroot()
        package_name = package_xml.findtext("name")
        self.assertEqual(package_name, _PACKAGE_NAME, f"{ros_distro}: package.xml has the wrong package name")

        package_marker = distro_prefix / "share" / "ament_index" / "resource_index" / "packages" / _PACKAGE_NAME
        self.assertTrue(
            package_marker.is_file(),
            f"{ros_distro}: missing ament package marker at {package_marker}",
        )

        plugin_marker = (
            distro_prefix
            / "share"
            / "ament_index"
            / "resource_index"
            / "hardware_interface__pluginlib__plugin"
            / _PACKAGE_NAME
        )
        self.assertTrue(
            plugin_marker.is_file(),
            f"{ros_distro}: missing hardware_interface plugin marker at {plugin_marker}",
        )
        self.assertEqual(
            plugin_marker.read_text(encoding="utf-8").strip(),
            _PLUGIN_XML_RELATIVE_PATH.as_posix(),
            f"{ros_distro}: plugin marker points at the wrong XML descriptor",
        )

        backend_filename = backend_library_name(ros_distro)
        bin_backend = self._extension_path / "bin" / backend_filename
        self.assertFalse(
            os.path.lexists(bin_backend),
            f"{ros_distro}: duplicate backend library should not be staged at {bin_backend}",
        )

        distro_backend = distro_prefix / "lib" / backend_filename
        self.assertTrue(
            distro_backend.is_file(),
            f"{ros_distro}: missing per-distro backend library entry at {distro_backend}",
        )

    async def test_humble_and_jazzy_pluginlib_payloads_are_isolated(self) -> None:
        for ros_distro in _SUPPORTED_DISTROS:
            with self.subTest(ros_distro=ros_distro):
                self._assert_distro_payload(ros_distro)

    async def test_active_supported_ros_distro_backend_is_ready(self) -> None:
        ros_distro = os.environ.get("ROS_DISTRO", "").lower()
        self.assertIn(
            ros_distro,
            _SUPPORTED_DISTROS,
            f"ROS_DISTRO must be one of {_SUPPORTED_DISTROS}, got {ros_distro or '<unset>'}",
        )

        self._assert_distro_payload(ros_distro)
        self.assertTrue(_backend.is_ready(), f"{ros_distro}: ros2_control backend did not initialize")

        distro_prefix = self._extension_path / ros_distro
        ament_prefixes = env_paths("AMENT_PREFIX_PATH")
        self.assertIn(
            path_key(distro_prefix),
            ament_prefixes,
            f"{ros_distro}: AMENT_PREFIX_PATH does not include the active distro prefix",
        )

        backend_lib_dir = path_key(distro_prefix / "lib")
        bin_dir = path_key(self._extension_path / "bin")
        library_prefixes = env_paths(LIBRARY_PATH_ENV)
        self.assertIn(
            backend_lib_dir,
            library_prefixes,
            f"{ros_distro}: {LIBRARY_PATH_ENV} does not include the active distro backend lib dir",
        )
        self.assertNotIn(
            bin_dir,
            library_prefixes,
            f"{ros_distro}: {LIBRARY_PATH_ENV} should not include the extension bin dir",
        )

        backend_name = backend_library_name(ros_distro)
        loaded_backends = [path for path in loaded_library_paths() if "isaacsim.ros2.control." in Path(path).name]
        self.assertTrue(
            any(Path(path).name == backend_name for path in loaded_backends),
            f"{ros_distro}: process did not load {backend_name}; loaded backends: {loaded_backends}",
        )
        self.assertIn(
            path_key(distro_prefix / "lib" / backend_name),
            [path_key(path) for path in loaded_backends],
            f"{ros_distro}: backend loaded from the wrong path; loaded backends: {loaded_backends}",
        )
        unexpected_backends = {
            backend_library_name(supported_distro)
            for supported_distro in _SUPPORTED_DISTROS
            if supported_distro != ros_distro
        }
        self.assertFalse(
            [path for path in loaded_backends if Path(path).name in unexpected_backends],
            f"{ros_distro}: process loaded a non-active distro backend; loaded backends: {loaded_backends}",
        )

    async def test_backend_use_sim_time_is_keyword_only(self) -> None:
        with self.assertRaises(TypeError):
            _backend.setup_cm("/World/R", "<robot/>", "/cfg.yaml", "", True, False)
