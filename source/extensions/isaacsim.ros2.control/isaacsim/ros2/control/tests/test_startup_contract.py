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

"""Startup/runtime-layout contract tests for isaacsim.ros2.control.

These tests intentionally sit between the mocked Python glue tests and the
full physics/controller_manager integration tests. They catch packaging and
loader regressions, such as missing ROS runtime libraries or a broken
AMENT_PREFIX_PATH, before a controller is constructed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import omni.kit.app
import omni.kit.test
from isaacsim.ros2.control.tests._runtime_layout import (
    IS_WINDOWS,
    LIBRARY_PATH_ENV,
    backend_library_name,
    env_paths,
    path_key,
    process_env,
)

_CORE_EXT = "isaacsim.ros2.core"
_CONTROL_EXT = "isaacsim.ros2.control"
_CONTROL_PACKAGE = "isaacsim_ros2_control"


def _ros_distro() -> str:
    return os.environ.get("ROS_DISTRO") or process_env("ROS_DISTRO") or "jazzy"


def _extension_path(ext_name: str) -> Path:
    manager = omni.kit.app.get_app().get_extension_manager()
    ext_id = manager.get_enabled_extension_id(ext_name)
    if not ext_id:
        raise AssertionError(f"{ext_name!r} is not enabled")
    path = manager.get_extension_path(ext_id)
    if not path:
        raise AssertionError(f"no extension path returned for {ext_name!r} ({ext_id})")
    return Path(path)


async def _wait_until(predicate, max_frames: int = 600) -> bool:
    app = omni.kit.app.get_app()
    for _ in range(max_frames):
        if predicate():
            return True
        await app.next_update_async()
    return predicate()


async def _ensure_backend_ready():
    from isaacsim.ros2.control.bindings import _isaacsim_ros2_control as backend

    ready = await _wait_until(lambda: backend.is_ready())
    if not ready:
        raise AssertionError("ros2.control backend never became ready")
    return backend


class TestStartupContract(omni.kit.test.AsyncTestCase):
    async def test_backend_binding_is_ready_and_exposes_backend_apis(self):
        backend = await _ensure_backend_ready()

        backend.set_profiling_enabled(False)
        backend.reset_profiling()
        self.assertIsInstance(json.loads(backend.get_profiling_json()), dict)
        backend.teardown_all()

    async def test_runtime_path_environment_prioritizes_control_and_core_prefixes(self):
        await _ensure_backend_ready()

        distro = _ros_distro()
        control_root = _extension_path(_CONTROL_EXT)
        core_root = _extension_path(_CORE_EXT)
        control_prefix = path_key(control_root / distro)
        core_prefix = path_key(core_root / distro)

        ament_paths = env_paths("AMENT_PREFIX_PATH")
        self.assertIn(control_prefix, ament_paths)
        self.assertIn(core_prefix, ament_paths)
        self.assertLess(ament_paths.index(control_prefix), ament_paths.index(core_prefix))

        library_paths = env_paths(LIBRARY_PATH_ENV)
        control_lib = path_key(Path(control_prefix) / "lib")
        core_lib = path_key(Path(core_prefix) / "lib")
        self.assertIn(control_lib, library_paths)
        self.assertIn(core_lib, library_paths)
        self.assertLess(library_paths.index(control_lib), library_paths.index(core_lib))
        self.assertNotIn(path_key(control_root / "bin"), library_paths)

    async def test_runtime_layout_contains_loader_prerequisites(self):
        distro = _ros_distro()
        control_prefix = _extension_path(_CONTROL_EXT) / distro
        control_bin = _extension_path(_CONTROL_EXT) / "bin"
        core_prefix = _extension_path(_CORE_EXT) / distro

        core_libraries = (
            [
                "fmt.dll",
                "controller_manager.dll",
                "hardware_interface.dll",
                "controller_interface.dll",
                "class_loader.dll",
                "console_bridge.dll",
                "rosidl_typesupport_cpp.dll",
                "rosgraph_msgs__rosidl_typesupport_cpp.dll",
                "std_msgs__rosidl_typesupport_cpp.dll",
            ]
            if IS_WINDOWS
            else [
                "libfmt.so.8",
                "libcontroller_manager.so",
                "libhardware_interface.so",
                "libcontroller_interface.so",
                "libclass_loader.so",
            ]
        )
        if IS_WINDOWS and distro == "jazzy":
            core_libraries.append("pal_statistics.dll")
        backend_library = backend_library_name(distro)
        self.assertFalse(
            os.path.lexists(control_bin / backend_library),
            f"duplicate backend library should not be staged in {control_bin}",
        )

        required = [
            *(core_prefix / "lib" / library for library in core_libraries),
            core_prefix / "share" / "ament_index" / "resource_index" / "packages" / "controller_manager",
            core_prefix / "share" / "ament_index" / "resource_index" / "packages" / "hardware_interface",
            core_prefix / "share" / "ament_index" / "resource_index" / "packages" / "joint_state_broadcaster",
            core_prefix / "share" / "ament_index" / "resource_index" / "packages" / "joint_trajectory_controller",
            core_prefix / "share" / "ament_index" / "resource_index" / "packages" / "rclcpp",
            control_prefix / "lib" / backend_library,
            control_prefix / "share" / _CONTROL_PACKAGE / "package.xml",
            control_prefix / "share" / _CONTROL_PACKAGE / "isaac_sim_system_plugin.xml",
            control_prefix / "share" / "ament_index" / "resource_index" / "packages" / _CONTROL_PACKAGE,
            control_prefix
            / "share"
            / "ament_index"
            / "resource_index"
            / "hardware_interface__pluginlib__plugin"
            / _CONTROL_PACKAGE,
        ]
        missing = [str(path) for path in required if not path.exists()]
        self.assertFalse(missing, "missing ros2.control runtime files:\n" + "\n".join(missing))

        plugin_xml = (control_prefix / "share" / _CONTROL_PACKAGE / "isaac_sim_system_plugin.xml").read_text(
            encoding="utf-8"
        )
        self.assertIn(f'<library path="isaacsim.ros2.control.{distro}">', plugin_xml)
        self.assertIn('name="isaacsim_ros2_control/IsaacSimSystem"', plugin_xml)

        plugin_resource = (
            control_prefix
            / "share"
            / "ament_index"
            / "resource_index"
            / "hardware_interface__pluginlib__plugin"
            / _CONTROL_PACKAGE
        ).read_text(encoding="utf-8")
        self.assertEqual(plugin_resource.strip(), "share/isaacsim_ros2_control/isaac_sim_system_plugin.xml")

    async def test_ament_index_resolves_core_and_control_resources(self):
        await _ensure_backend_ready()

        distro = _ros_distro()
        control_prefix = (_extension_path(_CONTROL_EXT) / distro).resolve()
        core_prefix = (_extension_path(_CORE_EXT) / distro).resolve()

        from ament_index_python.packages import get_package_prefix
        from ament_index_python.resources import get_resource

        # The native plugin uses setenv(), which updates the process env used by
        # C++ pluginlib but not Python's os.environ cache. Mirror it before
        # calling ament_index_python, and keep it mirrored for later tests.
        os.environ["AMENT_PREFIX_PATH"] = process_env("AMENT_PREFIX_PATH")
        self.assertEqual(Path(get_package_prefix("controller_manager")).resolve(), core_prefix)
        self.assertEqual(Path(get_package_prefix("joint_trajectory_controller")).resolve(), core_prefix)
        self.assertEqual(Path(get_package_prefix(_CONTROL_PACKAGE)).resolve(), control_prefix)
        content, prefix = get_resource("hardware_interface__pluginlib__plugin", _CONTROL_PACKAGE)
        self.assertEqual(Path(prefix).resolve(), control_prefix)
        self.assertEqual(content.strip(), "share/isaacsim_ros2_control/isaac_sim_system_plugin.xml")
