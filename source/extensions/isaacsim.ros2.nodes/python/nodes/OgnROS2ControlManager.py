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

"""OmniGraph node that brings up the ControllerManager on the first tick after Play.

The hot read/update/write loop runs from physics post-step callbacks, not OG ticks;
a timeline Stop schedules reconfiguration on the next compute.
"""

from __future__ import annotations

import carb
import omni.graph.core as og
import omni.timeline
from isaacsim.ros2.control import Ros2ControlManager


class OgnROS2ControlManagerInternalState:
    """Per-node state: timeline subscription and failure latch."""

    def __init__(self) -> None:
        self._timeline_sub = None
        self._needs_reconfigure = False
        # Latched after a setup() failure so compute() stops re-running setup (and
        # re-exporting the URDF) every tick; cleared on the next Stop->Play.
        self._failed = False

    def ensure_timeline_subscription(self) -> None:
        """Subscribe to timeline Stop events once (idempotent)."""
        if self._timeline_sub is not None:
            return
        timeline = omni.timeline.get_timeline_interface()
        if timeline is None:
            carb.log_warn("ROS2ControlManager: omni.timeline unavailable; per-Play teardown disabled")
            return
        stream = timeline.get_timeline_event_stream()
        if stream is None:
            carb.log_warn("ROS2ControlManager: timeline event stream unavailable; per-Play teardown disabled")
            return
        self._timeline_sub = stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            self._on_stop,
        )

    def _on_stop(self, _event: carb.events.IEvent) -> None:
        self._needs_reconfigure = True


class OgnROS2ControlManager:
    """OmniGraph node that configures a ControllerManager on the first tick after Play."""

    @staticmethod
    def internal_state() -> OgnROS2ControlManagerInternalState:
        """Create the per-node internal state."""
        return OgnROS2ControlManagerInternalState()

    @staticmethod
    def compute(db: og.Database) -> bool:
        """Configure the ControllerManager on the first tick after Play; no-op afterward."""
        state = db.per_instance_state
        state.ensure_timeline_subscription()

        if state._needs_reconfigure:
            db.state.configured = False
            state._failed = False
            state._needs_reconfigure = False

        if db.state.configured:
            return True
        if state._failed:
            return False  # latched: do not re-attempt until the next Stop->Play

        target_prims = db.inputs.targetPrim
        if not target_prims:
            db.log_error("ROS2ControlManager: targetPrim is required")
            return False
        if len(target_prims) != 1:
            db.log_error(f"ROS2ControlManager: exactly one targetPrim is required, got {len(target_prims)}")
            return False
        controller_config = db.inputs.controllerConfig
        if not controller_config:
            db.log_error("ROS2ControlManager: controllerConfig path is required")
            return False

        prim_path = str(target_prims[0])
        try:
            Ros2ControlManager.setup(
                prim_path=prim_path,
                controller_config=controller_config,
                urdf_path=db.inputs.urdfPath or None,
                namespace=db.inputs.namespace,
                publish_robot_description=db.inputs.publishRobotDescription,
                use_sim_time=db.inputs.useSimTime,
            )
        except Exception as exc:
            carb.log_error(f"ROS2ControlManager setup failed for {prim_path!r}: {exc}")
            state._failed = True
            return False

        db.state.configured = True
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True
