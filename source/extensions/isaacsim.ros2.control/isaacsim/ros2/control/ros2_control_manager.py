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

"""Public Python API for the in-process ros2_control ControllerManager."""

from __future__ import annotations

import carb
import omni.usd

from .bindings import _isaacsim_ros2_control as _backend
from .urdf_synth import build_full_urdf


class Ros2ControlManager:
    """In-process ros2_control ControllerManager for a USD articulation."""

    @staticmethod
    def setup(
        prim_path: str,
        controller_config: str,
        urdf_path: str | None = None,
        namespace: str = "",
        publish_robot_description: bool = True,
        *,
        use_sim_time: bool = True,
    ) -> int:
        """Bring up a ControllerManager for the articulation at ``prim_path``.

        ``urdf_path`` is optional; if None, falls back to
        ``customLayerData["isaac:sourceUrdf"]`` on the root layer (set by the
        URDF importer). Used only as a sensor / hardware-param overlay; the
        kinematic URDF is always live-exported from USD.

        When ``use_sim_time`` is true, provide ``/clock`` with a ROS 2 bridge
        clock node or another ROS clock source.
        """
        if not _backend.is_ready():
            raise RuntimeError(
                "isaacsim.ros2.control plugin not ready: backend .so failed to load "
                "or the shared rcl_context from isaacsim.ros2.core is not valid"
            )

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Ros2ControlManager.setup: no active USD stage")

        if urdf_path is None:
            meta = stage.GetRootLayer().customLayerData
            if isinstance(meta, dict):
                urdf_path = meta.get("isaac:sourceUrdf")
        if urdf_path:
            carb.log_info(f"Using sensor overlay URDF at {urdf_path}")

        urdf_xml = build_full_urdf(stage, prim_path, sensor_overlay_urdf_path=urdf_path)
        rc = _backend.setup_cm(
            articulation_path=prim_path,
            urdf_xml=urdf_xml,
            controller_yaml_path=controller_config,
            ns_name=namespace,
            publish_robot_description=publish_robot_description,
            use_sim_time=use_sim_time,
        )
        if rc == 0:
            carb.log_info(f"ros2_control: ControllerManager configured for {prim_path} (ns={namespace!r})")
            return 0
        if rc == -2:
            raise RuntimeError(
                f"setup_cm: a ControllerManager is already registered for {prim_path!r}. "
                "Tear it down first via Ros2ControlManager.teardown(prim_path)."
            )
        if rc == -3:
            raise RuntimeError(
                f"setup_cm: ControllerManager init failed for {prim_path!r}. "
                "Check the controller YAML and the synthesized URDF."
            )
        raise RuntimeError(f"setup_cm: failure code {rc}")

    @staticmethod
    def teardown(prim_path: str) -> None:
        """Tear down the ControllerManager for the articulation at ``prim_path``."""
        _backend.teardown_cm(prim_path)

    @staticmethod
    def teardown_all() -> None:
        """Tear down all ControllerManagers."""
        _backend.teardown_all()
