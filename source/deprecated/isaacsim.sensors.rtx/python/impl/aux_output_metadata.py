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

"""Helpers for keeping RTX sensor aux output metadata in sync."""

from __future__ import annotations

from typing import Any

import carb
from pxr import Sdf, Usd

GMO_CHANNELS_ATTR = "_replicator:rendervar:GenericModelOutput:channels"
LIDAR_AUX_OUTPUT_ATTR = "omni:sensor:Core:auxOutputType"
RADAR_AUX_OUTPUT_ATTR = "omni:sensor:WpmDmat:auxOutputType"

_GMO_ATTR_PREFIX = "_replicator:rendervar:GenericModelOutput"
_VALID_AUX_OUTPUT_TYPES = frozenset({"NONE", "BASIC", "EXTRA", "FULL"})


def sync_gmo_channel_metadata(prim: Usd.Prim, aux_output_attr_name: str, aux_output_type: Any | None = None) -> bool:
    """Mirror a sensor aux output attribute to Replicator's GMO channels metadata.

    Replicator reads ``GMO_CHANNELS_ATTR`` while creating the render product's
    GenericModelOutput RenderVar. The OmniSensor attribute remains the user-facing
    setting, so keep the Replicator metadata aligned with it before render product
    creation.

    Args:
        prim: Sensor prim to synchronize.
        aux_output_attr_name: User-facing aux output attribute name on ``prim``.
        aux_output_type: Optional creation-time aux output value. Used when
            the referenced sensor asset does not already author ``aux_output_attr_name``.

    Returns:
        True when metadata was authored or already matched, otherwise False.
    """
    if not prim or not prim.IsValid():
        return False

    aux_output_attr = prim.GetAttribute(aux_output_attr_name)
    if aux_output_type is None and aux_output_attr and aux_output_attr.IsValid():
        aux_output_type = aux_output_attr.Get()

    legacy_attr_name = f"{_GMO_ATTR_PREFIX}:{aux_output_attr_name}"
    legacy_attr = prim.GetAttribute(legacy_attr_name)
    if legacy_attr and legacy_attr.IsValid():
        legacy_value = legacy_attr.Get()
        prim.RemoveProperty(legacy_attr_name)
        if legacy_value is not None:
            aux_output_type = legacy_value

    if aux_output_type is None:
        return False

    aux_output_type = str(aux_output_type)
    if aux_output_type not in _VALID_AUX_OUTPUT_TYPES:
        carb.log_warn(
            f"Invalid RTX sensor aux output type '{aux_output_type}' on {prim.GetPath()}; "
            f"expected one of {sorted(_VALID_AUX_OUTPUT_TYPES)}."
        )
        return False

    channels = [aux_output_type]
    channels_attr = prim.GetAttribute(GMO_CHANNELS_ATTR)
    if not channels_attr or not channels_attr.IsValid():
        channels_attr = prim.CreateAttribute(GMO_CHANNELS_ATTR, Sdf.ValueTypeNames.StringArray, True)

    current = channels_attr.Get()
    if list(current or []) != channels:
        channels_attr.Set(channels)
    return True
