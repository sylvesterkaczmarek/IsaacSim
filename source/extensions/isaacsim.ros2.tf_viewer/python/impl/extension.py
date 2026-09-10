# SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""TF Viewer extension for visualizing ROS 2 TF frames."""

import os
import sys
import threading
import time

import carb
import omni.ext
import omni.kit.app
import omni.kit.viewport.utility

from . import ui_builder, viewport_scene


class Extension(omni.ext.IExt):
    """Extension for the ROS 2 TF Viewer."""

    def on_startup(self, ext_id: str) -> None:
        """Initialize the extension.

        Args:
            ext_id: Extension identifier provided by Kit.
        """
        self._extension_manager = omni.kit.app.get_app().get_extension_manager()
        ext_path = self._extension_manager.get_extension_path(ext_id)

        # get extension settings
        settings = carb.settings.get_settings()
        self._cpp = settings.get("/exts/isaacsim.ros2.tf_viewer/cpp")
        self._include_root_frame = settings.get("/exts/isaacsim.ros2.tf_viewer/include_root_frame")

        # load the transform listener module (C++ plugin or Python implementation)
        if self._cpp:
            # set environment PATH
            if sys.platform == "win32":
                if os.environ.get("PATH"):
                    os.environ["PATH"] = os.environ.get("PATH") + ";" + ext_path + "/bin"
                else:
                    os.environ["PATH"] = ext_path + "/bin"
            # load carb plugin
            carb.get_framework().load_plugins(
                loaded_file_wildcards=["isaacsim.ros2.tf_viewer.plugin"],
                search_paths=[os.path.abspath(os.path.join(ext_path, "bin"))],
            )
            from .. import _transform_listener as transform_listener
        else:
            from . import transform_listener_ros2 as transform_listener

        self._module = transform_listener

        # get viewport scene
        self._viewport_window = omni.kit.viewport.utility.get_active_viewport_window()
        self._viewport_scene = viewport_scene.ViewportScene(self._viewport_window, ext_id)

        # ui components
        self._ui_builder = ui_builder.UIBuilder(
            menu_path="Window",
            window_title="TF Viewer",
            viewport_scene=self._viewport_scene,
            on_visibility_changed_callback=self._on_visibility_changed,
            on_reset_callback=self._on_reset,
        )

        self._running = False
        self._interface = None

        # data
        self._frames = {"World", "world", "map"}

    def on_shutdown(self) -> None:
        """Clean up resources when the extension shuts down."""
        self._running = False
        self._ui_builder.shutdown()
        # destroy viewport scene
        if self._viewport_scene:
            self._viewport_scene.manipulator.clear()
            self._viewport_scene.destroy()
            self._viewport_scene = None

    def _on_visibility_changed(self, visible: bool) -> None:
        bridge_enabled = self._extension_manager.is_extension_enabled("isaacsim.ros2.bridge")
        if not bridge_enabled:
            carb.log_warn("The 'isaacsim.ros2.bridge' extension is not enabled")

        distro = os.environ.get("ROS_DISTRO", "").lower()
        module = self._module if bridge_enabled else None
        if visible:
            carb.log_info(f"Acquire interface (ROS2 | {distro} | {'cpp' if self._cpp else 'python'})")
            # acquire interface
            if module:
                self._interface = module.acquire_transform_listener_interface()
                self._interface.initialize(distro)
            # start thread
            threading.Thread(target=self._update_transforms).start()
        else:
            carb.log_info(f"Release interface (ROS2 | {distro} | {'cpp' if self._cpp else 'python'})")
            self._running = False
            # release interface
            if module:
                self._interface.finalize()
                module.release_transform_listener_interface(self._interface)
            self._interface = None
            # clear scene
            if self._viewport_scene:
                self._viewport_scene.manipulator.clear()
            carb.log_info("Transform listener released")

    def _on_reset(self) -> None:
        if self._interface:
            self._interface.reset()

    def _update_transforms(self) -> None:
        self._running = True
        while self._running:
            if self._cpp:
                self._interface.spin()
            # get transforms
            root_frame = self._ui_builder.root_frame
            frames, transforms, relations = self._interface.get_transforms(root_frame)
            # add root frame if not listed
            if self._include_root_frame and root_frame not in transforms:
                transforms[root_frame] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            # update frames and ui
            self._frames.update(frames)
            self._ui_builder.update(self._frames)
            # draw scene
            self._viewport_scene.manipulator.update_transforms(transforms, relations)
            time.sleep(1 / self._ui_builder.update_frequency)
        # clear scene
        if self._viewport_scene:
            self._viewport_scene.manipulator.clear()
