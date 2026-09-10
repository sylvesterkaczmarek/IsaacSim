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

"""Behavior script that randomly applies texture materials to visual prims."""

from __future__ import annotations

import string
from typing import Any

import carb
import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.materials import OmniPbrMaterial
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS, SCOPE_NAME
from isaacsim.replicator.behavior.utils.behavior_utils import (
    apply_behavior_seed,
    check_if_exposed_variables_should_be_removed,
    create_exposed_variables,
    csv_has_relative_asset_url,
    get_exposed_variable,
    order_scalar_range,
    remove_empty_scopes,
    remove_exposed_variables,
    resolve_csv_asset_urls,
)
from isaacsim.storage.native import get_assets_root_path
from omni.behavior.scripting.core import BehaviorScript
from pxr import Gf, Sdf, UsdGeom, UsdShade


class TextureRandomizer(BehaviorScript):
    """Behavior script that creates texture materials from the given list and randomly applies them to visual prim(s)."""

    BEHAVIOR_NS = "textureRandomizer"
    VARIABLES_TO_EXPOSE = [
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
            "attr_name": "textures:assets",
            "attr_type": Sdf.ValueTypeNames.AssetArray,
            "default_value": [],
            "doc": "Asset list of textures to randomize.",
        },
        {
            "attr_name": "textures:csv",
            "attr_type": Sdf.ValueTypeNames.String,
            "default_value": (
                "/Isaac/Materials/Textures/Patterns/nv_bamboo_desktop.jpg,"
                "/Isaac/Materials/Textures/Patterns/nv_brick_grey.jpg"
            ),
            "doc": "CSV list of texture URLs to randomize.",
        },
        {
            "attr_name": "projectUvwProbability",
            "attr_type": Sdf.ValueTypeNames.Float,
            "default_value": 0.9,
            "doc": "Probability that 'project_uvw' is set to True.",
        },
        {
            "attr_name": "textureScaleRange",
            "attr_type": Sdf.ValueTypeNames.Float2,
            "default_value": Gf.Vec2f(0.1, 1.0),
            "doc": "Texture scale range as (min, max). Inverted bounds are swapped.",
        },
        {
            "attr_name": "textureRotateRange",
            "attr_type": Sdf.ValueTypeNames.Float2,
            "default_value": Gf.Vec2f(0.0, 45.0),
            "doc": "Texture rotation range in degrees as (min, max). Inverted bounds are swapped.",
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
        self._update_counter = 0
        self._interval = 0
        self._texture_urls = []
        self._valid_prims = []
        self._initial_materials = {}
        self._texture_materials = []

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
        include_children = self._get_exposed_variable("includeChildren")
        self._interval = self._get_exposed_variable("interval")
        textures_assets = self._get_exposed_variable("textures:assets")
        textures_csv = self._get_exposed_variable("textures:csv")
        self._project_uvw_probability = self._get_exposed_variable("projectUvwProbability")
        self._texture_scale_range = order_scalar_range(
            self._get_exposed_variable("textureScaleRange"), owner=self.prim_path, label="textureScaleRange"
        )
        self._texture_rotate_range = order_scalar_range(
            self._get_exposed_variable("textureRotateRange"), owner=self.prim_path, label="textureRotateRange"
        )
        seed = self._get_exposed_variable("seed")
        apply_behavior_seed(self, seed)

        # Skip the one-shot setup if already initialized (e.g. a play/pause/play loop used by the SDG
        # capture pipeline). Re-caching here would store the current randomizer material as the
        # "original" and break material restoration on stop. ``_reset`` clears ``_valid_prims`` so
        # the next play session naturally re-runs the full setup.
        if self._valid_prims:
            return

        # Get the valid prims
        if self.prim and self.prim.IsValid():
            if include_children:
                self._valid_prims = prim_utils.get_all_matching_child_prims(
                    self.prim,
                    predicate=lambda prim, _: prim.IsValid() and prim.IsA(UsdGeom.Gprim),
                    include_self=True,
                )
            elif self.prim.IsA(UsdGeom.Gprim):
                self._valid_prims = [self.prim]
            else:
                self._valid_prims = []
        else:
            self._valid_prims = []
        if not self._valid_prims:
            carb.log_warn(f"[{self.prim_path}] No valid prims found.")

        # Cache original materials to restore after the randomization has stopped
        for prim in self._valid_prims:
            self._initial_materials[prim] = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]

        # Create materials to randomize for each valid prim
        self._create_materials()

        # Store the texture urls
        self._texture_urls = []

        # Add the texture paths from the assets list
        for texture_asset in textures_assets or []:
            self._texture_urls.append(texture_asset.path)

        assets_root_path = None
        if csv_has_relative_asset_url(textures_csv or ""):
            try:
                assets_root_path = get_assets_root_path()
            except Exception as error:
                carb.log_warn(f"[{self.prim_path}] Could not resolve assets root path: {error}")
                assets_root_path = None

        self._texture_urls.extend(resolve_csv_asset_urls(textures_csv or "", assets_root_path, owner=self.prim_path))

    def _reset(self) -> None:
        # Bind the original materials back to the prims
        self._restore_original_materials()

        # Delete the materials created for randomization
        self._remove_texture_materials()

        # Clear any empty scopes under the prim behavior scope
        if self.stage:
            with stage_utils.use_stage(self.stage):
                scope_root_prim = prim_utils.get_prim_at_path(f"{SCOPE_NAME}")
            remove_empty_scopes(scope_root_prim, self.stage)
        else:
            carb.log_warn(f"[{self.prim_path}] Stage is not valid to remove empty scopes.")

        self._valid_prims.clear()
        self._update_counter = 0
        self._rng = None
        self._last_seed = None
        self._rng_injected = False

    def _apply_behavior(self) -> None:
        # Skip the tick if no textures are configured to avoid numpy.random.Generator.choice raising on an empty list
        if not self._texture_urls:
            carb.log_warn(f"[{self.prim_path}] No texture URLs configured; skipping randomization tick.")
            return

        # Randomize the textures and parameters for each material
        for material in self._texture_materials:
            diffuse_texture = self._rng.choice(self._texture_urls)
            material.set_input_values("diffuse_texture", [diffuse_texture])
            project_uvw = self._rng.choice(
                [True, False],
                p=[self._project_uvw_probability, 1 - self._project_uvw_probability],
            )
            material.set_input_values("project_uvw", [bool(project_uvw)])
            texture_scale = self._rng.uniform(self._texture_scale_range[0], self._texture_scale_range[1])
            material.set_input_values("texture_scale", [texture_scale, texture_scale])
            texture_rotate = self._rng.uniform(self._texture_rotate_range[0], self._texture_rotate_range[1])
            material.set_input_values("texture_rotate", [texture_rotate])

    def _create_materials(self) -> None:
        if not self.stage:
            carb.log_warn(f"[{self.prim_path}] Stage is not valid to create materials.")
            return

        # Create a unique looks path in the given behavior scope
        with stage_utils.use_stage(self.stage):
            looks_path = stage_utils.generate_next_free_path(
                f"{SCOPE_NAME}/{self.BEHAVIOR_NS}/Looks", prepend_default_prim=False
            )
        for prim in self._valid_prims:
            # Create a unique path for the material (WAR for ISIM-4054)
            rand_postfix = "".join(self._rng.choice(list(string.ascii_letters + string.digits), size=4))
            with stage_utils.use_stage(self.stage):
                mtl_path = stage_utils.generate_next_free_path(
                    f"{looks_path}/OmniPBR_{rand_postfix}", prepend_default_prim=False
                )

            # Create the material and bind it to the prim
            with stage_utils.use_stage(self.stage):
                material = OmniPbrMaterial(mtl_path)
                XformPrim(prim.GetPath().pathString).apply_visual_materials(material)

            # Cache the material for randomization
            self._texture_materials.append(material)

    def _get_exposed_variable(self, attr_name: str) -> Any:
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)

    def _restore_original_materials(self) -> None:
        if not self.stage:
            carb.log_warn(f"[{self.prim_path}] Stage is not valid to restore original materials.")
            self._initial_materials.clear()
            return
        for prim in self._valid_prims:
            orig_mat = self._initial_materials.get(prim)
            if orig_mat:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    orig_mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants
                )
            else:
                UsdShade.MaterialBindingAPI(prim).UnbindAllBindings()
        self._initial_materials.clear()

    def _remove_texture_materials(self) -> None:
        if not self.stage:
            carb.log_warn(f"[{self.prim_path}] Stage is not valid to remove texture materials.")
            self._texture_materials.clear()
            return
        for material in self._texture_materials:
            if material.valid:
                with stage_utils.use_stage(self.stage):
                    stage_utils.delete_prim(material.materials[0].GetPrim())
        self._texture_materials.clear()

    def set_rng(self, rng: np.random.Generator | None = None) -> None:
        """Set the random number generator, overriding the USD seed attribute.

        The injected generator is kept until the USD seed changes or the behavior resets.

        Args:
            rng: Numpy random generator. If None, creates a new default generator.
        """
        self._rng = rng if rng is not None else np.random.default_rng()
        self._rng_injected = True
