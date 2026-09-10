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

"""Behavior script that randomizes prim locations within specified bounds."""

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


class LocationRandomizer(BehaviorScript):
    """Behavior script that randomizes the location of prims within specified bounds.

    The behavior can be applied to multiple prims at once.
    """

    BEHAVIOR_NS = "locationRandomizer"
    VARIABLES_TO_EXPOSE = [
        {
            "attr_name": "range:minPosition",
            "attr_type": Sdf.ValueTypeNames.Vector3d,
            "default_value": Gf.Vec3d(-1.0, -1.0, -1.0),
            "doc": "Minimum bounds of the random offset applied each update. Inverted axes are swapped.",
        },
        {
            "attr_name": "range:maxPosition",
            "attr_type": Sdf.ValueTypeNames.Vector3d,
            "default_value": Gf.Vec3d(1.0, 1.0, 1.0),
            "doc": "Maximum bounds of the random offset applied each update. Inverted axes are swapped.",
        },
        {
            "attr_name": "frame:useRelativeFrame",
            "attr_type": Sdf.ValueTypeNames.Bool,
            "default_value": True,
            "doc": (
                "If true, preserve the prim's initial offset (from the target prim if set, otherwise from its own "
                "starting position) and add the random offset on top. If false, the random offset is applied as an "
                "absolute position relative to the target prim if set, or to the world origin otherwise."
            ),
        },
        {
            "attr_name": "frame:targetPrimPath",
            "attr_type": Sdf.ValueTypeNames.String,
            "default_value": "",
            "doc": (
                "Optional path to a reference prim. When set, randomization is anchored to this prim's world "
                "location; leave empty to randomize independently of any other prim."
            ),
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
        self._min_position = Gf.Vec3d(-1.0, -1.0, -1.0)
        self._max_position = Gf.Vec3d(1.0, 1.0, 1.0)
        self._use_relative_frame = False
        self._target_prim = None
        self._update_counter = 0
        self._interval = 0
        self._valid_prims = []
        self._initial_locations = {}
        self._target_offsets = {}

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
        self._min_position = self._get_exposed_variable("range:minPosition")
        self._max_position = self._get_exposed_variable("range:maxPosition")
        self._min_position, self._max_position = order_range_vectors(
            self._min_position, self._max_position, labels=("x", "y", "z"), owner=self.prim_path
        )
        self._use_relative_frame = self._get_exposed_variable("frame:useRelativeFrame")
        target_prim_path = self._get_exposed_variable("frame:targetPrimPath")
        include_children = self._get_exposed_variable("includeChildren")
        self._interval = self._get_exposed_variable("interval")
        seed = self._get_exposed_variable("seed")
        apply_behavior_seed(self, seed)

        # Skip the one-shot setup if already initialized (e.g. a play/pause/play loop). Re-caching here
        # would store the current randomized location as the "initial" and break restoration on stop.
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

        # Check if the randomization should be relative to a target prim
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

        # Save the initial locations (and relative offsets) of the prims
        for prim in self._valid_prims:
            if "xformOp:translate" in prim_utils.get_prim_attribute_names(prim):
                self._initial_locations[prim] = Gf.Vec3d(prim_utils.get_prim_attribute_value(prim, "xformOp:translate"))
            else:
                translation, _ = xform_utils.get_local_pose(prim, device="cpu")
                self._initial_locations[prim] = Gf.Vec3d(*translation.numpy().tolist())
            if self._target_prim:
                self._target_offsets[prim] = self._initial_locations[prim] - self._get_world_location(self._target_prim)

    def _reset(self) -> None:
        # Set prims back to their initial locations
        for prim, location in self._initial_locations.items():
            if prim_utils.is_prim_valid(prim):
                with stage_utils.use_stage(self.stage):
                    rep.functional.modify.position(prim, location, write_to_usd=True)
        # Clear cached values
        self._valid_prims.clear()
        self._initial_locations.clear()
        self._target_offsets.clear()
        self._target_prim = None
        self._interval = 0
        self._update_counter = 0
        self._rng = None
        self._last_seed = None
        self._rng_injected = False

    def _apply_behavior(self) -> None:
        # Run the randomization for each valid prim
        for prim in self._valid_prims:
            if prim_utils.is_prim_valid(prim):
                self._randomize_location(prim)

    def _get_exposed_variable(self, attr_name: str) -> Any:
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)

    def _get_world_location(self, prim: Any) -> Gf.Vec3d:
        translation, _ = xform_utils.get_world_pose(prim, device="cpu")
        return Gf.Vec3d(*translation.numpy().tolist())

    def _randomize_location(self, prim: Any) -> None:
        # Generate a random offset within the bounds
        random_offset = Gf.Vec3d(
            self._rng.uniform(self._min_position[0], self._max_position[0]),
            self._rng.uniform(self._min_position[1], self._max_position[1]),
            self._rng.uniform(self._min_position[2], self._max_position[2]),
        )

        # Initialize location
        loc = random_offset

        # Handle the target prim if specified
        if self._target_prim:
            target_loc = self._get_world_location(self._target_prim)

            if self._use_relative_frame:
                # Maintain the offset from the target prim
                loc = target_loc + self._target_offsets[prim] + random_offset
            else:
                # Move the prim to the randomized location relative to the target prim
                loc = target_loc + random_offset
        else:
            if self._use_relative_frame:
                # Add the initial location if using the relative frame
                loc += self._initial_locations[prim]

        # Set the randomized location to the prim
        with stage_utils.use_stage(self.stage):
            try:
                rep.functional.modify.position(prim, loc, write_to_usd=True)
            except RuntimeError as error:
                if "Used null prim" not in str(error):
                    raise

    def set_rng(self, rng: np.random.Generator | None = None) -> None:
        """Set the random number generator, overriding the USD seed attribute.

        The injected generator is kept until the USD seed changes or the behavior resets.

        Args:
            rng: Numpy random generator. If None, creates a new default generator.
        """
        self._rng = rng if rng is not None else np.random.default_rng()
        self._rng_injected = True
