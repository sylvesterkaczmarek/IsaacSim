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

"""Behavior script that randomizes prim rotations within specified euler angle bounds."""

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
    apply_behavior_seed,
    check_if_exposed_variables_should_be_removed,
    create_exposed_variables,
    get_exposed_variable,
    order_range_vectors,
    remove_exposed_variables,
)
from omni.behavior.scripting.core import BehaviorScript
from pxr import Gf, Sdf, UsdGeom


class RotationRandomizer(BehaviorScript):
    """Behavior script that randomizes the rotation of prims within specified euler angle bounds.

    Rotations are handled using various xformOps, including 'xformOp:rotateXYZ', 'xformOp:orient', etc.
    depending on the existing xformOps of the prim. If no rotation xformOp exists 'xformOp:orient' is used.
    The behavior can be applied to multiple prims at once.
    """

    BEHAVIOR_NS = "rotationRandomizer"
    VARIABLES_TO_EXPOSE = [
        {
            "attr_name": "range:minRotation",
            "attr_type": Sdf.ValueTypeNames.Vector3d,
            "default_value": Gf.Vec3d(0.0, 0.0, 0.0),
            "doc": "The minimum rotation (in degrees) for the randomization. Inverted axes are swapped.",
        },
        {
            "attr_name": "range:maxRotation",
            "attr_type": Sdf.ValueTypeNames.Vector3d,
            "default_value": Gf.Vec3d(360.0, 360.0, 360.0),
            "doc": "The maximum rotation (in degrees) for the randomization. Inverted axes are swapped.",
        },
        {
            "attr_name": "includeChildren",
            "attr_type": Sdf.ValueTypeNames.Bool,
            "default_value": True,
            "doc": "Include valid prim children to the behavior.",
        },
        {
            "attr_name": "interval",
            "attr_type": Sdf.ValueTypeNames.UInt,
            "default_value": 0,
            "doc": "Interval for updating the behavior. Value 0 means every frame.",
        },
        {
            "attr_name": "seed",
            "attr_type": Sdf.ValueTypeNames.Int,
            "default_value": -1,
            "doc": "Random seed for reproducible randomization. Use -1 for non-deterministic behavior. Changes apply on the next play or resume.",
        },
    ]

    def on_init(self) -> None:
        """Called when the script is assigned to a prim."""
        self._rng = None
        self._last_seed = None
        self._rng_injected = False
        self._min_rotation = Gf.Vec3d(0.0, 0.0, 0.0)
        self._max_rotation = Gf.Vec3d(360.0, 360.0, 360.0)
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
        self._min_rotation = self._get_exposed_variable("range:minRotation")
        self._max_rotation = self._get_exposed_variable("range:maxRotation")
        self._min_rotation, self._max_rotation = order_range_vectors(
            self._min_rotation, self._max_rotation, labels=("x", "y", "z"), owner=self.prim_path
        )
        include_children = self._get_exposed_variable("includeChildren")
        self._interval = self._get_exposed_variable("interval")
        seed = self._get_exposed_variable("seed")
        apply_behavior_seed(self, seed)

        # Skip the one-shot setup if already initialized (e.g. a play/pause/play loop). Re-caching here
        # would store the current randomized rotation as the "initial" and break restoration on stop.
        if self._valid_prims:
            return

        # Get the prims to apply the behavior to
        if self.prim and self.prim.IsValid():
            if include_children:
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

    def _reset(self) -> None:
        # Set prims back to their initial rotations
        for prim, orientation in self._initial_rotations.items():
            if prim_utils.is_prim_valid(prim):
                quaternion = Gf.Quatd(orientation[0], Gf.Vec3d(*orientation[1:]))
                with stage_utils.use_stage(self.stage):
                    try:
                        rep.functional.modify.rotation(prim, quaternion, write_to_usd=True)
                    except RuntimeError as error:
                        if "Used null prim" not in str(error):
                            raise
        # Clear cached values
        self._valid_prims.clear()
        self._initial_rotations.clear()
        self._interval = 0
        self._update_counter = 0
        self._rng = None
        self._last_seed = None
        self._rng_injected = False

    def _apply_behavior(self) -> None:
        # Randomize the rotation for each valid prim
        for prim in self._valid_prims:
            if prim_utils.is_prim_valid(prim):
                self._randomize_rotation(prim)

    def _randomize_rotation(self, prim: Any) -> None:
        rotation = (
            self._rng.uniform(self._min_rotation[0], self._max_rotation[0]),
            self._rng.uniform(self._min_rotation[1], self._max_rotation[1]),
            self._rng.uniform(self._min_rotation[2], self._max_rotation[2]),
        )
        with stage_utils.use_stage(self.stage):
            try:
                rep.functional.modify.rotation(
                    prim,
                    rotation,
                    value_rotation_order="XYZ",
                    write_to_usd=True,
                )
            except RuntimeError as error:
                if "Used null prim" not in str(error):
                    raise

    def _get_exposed_variable(self, attr_name: str) -> Any:
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)

    def set_rng(self, rng: np.random.Generator | None = None) -> None:
        """Set the random number generator, overriding the USD seed attribute.

        The injected generator is kept until the USD seed changes or the behavior resets.

        Args:
            rng: Numpy random generator. If None, creates a new default generator.
        """
        self._rng = rng if rng is not None else np.random.default_rng()
        self._rng_injected = True
