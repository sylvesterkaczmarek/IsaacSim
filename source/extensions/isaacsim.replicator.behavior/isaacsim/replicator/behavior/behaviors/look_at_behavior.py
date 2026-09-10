# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Behavior script that orients prims to look at a target location or prim."""

from __future__ import annotations

from typing import Any

import carb
import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.xform as xform_utils
import numpy as np
import omni.replicator.core as rep
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS
from isaacsim.replicator.behavior.utils.behavior_utils import (
    check_if_exposed_variables_should_be_removed,
    create_exposed_variables,
    get_exposed_variable,
    remove_exposed_variables,
)
from omni.behavior.scripting.core import BehaviorScript
from pxr import Gf, Sdf, UsdGeom


class LookAtBehavior(BehaviorScript):
    """Behavior script that orients prims to look at a target location or prim.

    The behavior can be applied to multiple prims at once, and the target can be a location or a prim.
    The behavior can be updated every frame or at a specified interval.
    """

    BEHAVIOR_NS = "lookAtBehavior"

    VARIABLES_TO_EXPOSE = [
        {
            "attr_name": "targetLocation",
            "attr_type": Sdf.ValueTypeNames.Vector3d,
            "default_value": Gf.Vec3d(0.0, 0.0, 0.0),
            "doc": "The 3D vector specifying the location to look at.",
        },
        {
            "attr_name": "targetPrimPath",
            "attr_type": Sdf.ValueTypeNames.String,
            "default_value": "",
            "doc": "The path of the target prim to look at. If specified, it has priority over the target location.",
        },
        {
            "attr_name": "upAxis",
            "attr_type": Sdf.ValueTypeNames.Vector3d,
            "default_value": Gf.Vec3d(0.0, 0.0, 1.0),
            "doc": (
                "The look-at up axis. Use the world up axis (e.g., Gf.Vec3d(0.0, 0.0, 1.0) for +Z) to keep\n"
                "the prim (usually camera) aligned with the world's vertical (e.g., to keep the horizon level).\n"
                "Or use the camera's local up axis (e.g., Gf.Vec3d(0.0, 1.0, 0.0) for +Y) if the camera moves."
            ),
        },
        {
            "attr_name": "includeChildren",
            "attr_type": Sdf.ValueTypeNames.Bool,
            "default_value": True,
            "doc": "Include prim children in the look-at randomization.",
        },
        {
            "attr_name": "interval",
            "attr_type": Sdf.ValueTypeNames.UInt,
            "default_value": 0,
            "doc": "Interval for updating the behavior. Value 0 means every frame.",
        },
    ]

    def on_init(self) -> None:
        """Called when the script is assigned to a prim."""
        self._target_location = Gf.Vec3d(0.0, 0.0, 0.0)
        self._target_prim = None
        self._up_axis = Gf.Vec3d(0.0, 0.0, 1.0)
        self._update_counter = 0
        self._interval = 0
        self._valid_prims = []
        self._initial_rotations = {}

        # Expose the variables as USD attributes
        create_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)

    def on_destroy(self) -> None:
        """Called when the script is unassigned from a prim."""
        self._reset()
        # Exposed variables should be removed if the script is no longer assigned to the prim
        if check_if_exposed_variables_should_be_removed(self.prim, __file__):
            remove_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)

    def on_play(self) -> None:
        """Called when `play` is pressed."""
        self._setup()
        # Make sure the initial behavior is applied if the interval is larger than 0
        if self._interval > 0:
            self._apply_behavior()

    def on_stop(self) -> None:
        """Called when `stop` is pressed."""
        self._reset()

    def on_update(self, current_time: float, delta_time: float) -> None:
        """Called on per frame update events that occur when `playing`.

        Args:
            current_time: The current simulation time.
            delta_time: The time elapsed since the last update.
        """
        if delta_time <= 0:
            return
        if self._interval <= 0:
            self._apply_behavior()
        else:
            self._update_counter += 1
            if self._update_counter >= self._interval:
                self._apply_behavior()
                self._update_counter = 0

    def _setup(self) -> None:
        # Fetch the exposed attributes (re-read on every setup so runtime edits take effect on the next apply)
        self._target_location = self._get_exposed_variable("targetLocation")
        target_prim_path = self._get_exposed_variable("targetPrimPath")
        self._include_children = self._get_exposed_variable("includeChildren")
        self._interval = self._get_exposed_variable("interval")
        self._up_axis = self._get_exposed_variable("upAxis")

        # Skip the one-shot setup if already initialized (e.g. a play/pause/play loop). Re-caching here
        # would store the current look-at rotation as the "initial" and break restoration on stop.
        if self._valid_prims:
            return

        # Get the prims to apply the behavior to
        if self.prim and self.prim.IsValid():
            if self._include_children:
                self._valid_prims = prim_utils.get_all_matching_child_prims(
                    self.prim,
                    predicate=lambda prim, _: prim.IsValid() and prim.IsA(UsdGeom.Xformable),
                    include_self=True,
                )
            elif self.prim.IsA(UsdGeom.Xformable):
                self._valid_prims = [self.prim]
            else:
                self._valid_prims = []
        else:
            self._valid_prims = []
        if not self._valid_prims:
            carb.log_warn(f"[{self.prim_path}] No valid prims found.")

        # Save the initial local orientations of the prims.
        for prim in self._valid_prims:
            _, orientation = xform_utils.get_local_pose(prim, device="cpu")
            self._initial_rotations[prim] = orientation.numpy().tolist()

        # Check if targetPrimPath is specified and retrieve the target prim
        if target_prim_path:
            if not self.stage:
                carb.log_warn(f"[{self.prim_path}] Stage is not valid to access target prim '{target_prim_path}'.")
                self._target_prim = None
            else:  # Stage is valid
                with stage_utils.use_stage(self.stage):
                    fetched_prim = prim_utils.get_prim_at_path(target_prim_path)
                if fetched_prim and fetched_prim.IsValid() and fetched_prim.IsA(UsdGeom.Xformable):
                    self._target_prim = fetched_prim
                else:
                    self._target_prim = None
                    carb.log_warn(
                        f"[{self.prim_path}] Target prim '{target_prim_path}' not found, not valid, or not Xformable."
                    )

    def _reset(self) -> None:
        # Set prims back to their initial rotations
        for prim, orientation in self._initial_rotations.items():
            if prim_utils.is_prim_valid(prim):
                quaternion = Gf.Quatd(orientation[0], Gf.Vec3d(*orientation[1:]))
                with stage_utils.use_stage(self.stage):
                    rep.functional.modify.rotation(prim, quaternion, write_to_usd=True)
        # Clear cached values
        self._valid_prims.clear()
        self._initial_rotations.clear()
        self._target_prim = None
        self._interval = 0
        self._update_counter = 0

    def _apply_behavior(self) -> None:
        target = (
            self._target_prim
            if self._target_prim is not None and prim_utils.is_prim_valid(self._target_prim)
            else self._target_location
        )
        if isinstance(target, Gf.Vec3d):
            target_position = np.asarray(target)
        else:
            target_position, _ = xform_utils.get_world_pose(target, device="cpu")
            target_position = target_position.numpy()
        for prim in self._valid_prims:
            if prim_utils.is_prim_valid(prim):
                eye_position, _ = xform_utils.get_world_pose(prim, device="cpu")
                if np.linalg.norm(target_position - eye_position.numpy()) < 1e-6:
                    continue
                with stage_utils.use_stage(self.stage):
                    rep.functional.modify.look_at(
                        prim,
                        target,
                        look_at_up_axis=self._up_axis,
                        write_to_usd=True,
                    )

    def _get_exposed_variable(self, attr_name: str) -> Any:
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)
