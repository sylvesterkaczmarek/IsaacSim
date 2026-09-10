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

"""Collision pipeline configuration for Newton physics in Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class HydroelasticConfig:
    """Runtime configuration for hydroelastic contact.

    Per-shape opt-in (enable, stiffness, SDF resolution) is authored in USD via
    NewtonSDFCollisionAPI and parsed automatically.
    """

    enabled: bool = True
    """Whether to run the hydroelastic contact path when hydroelastic shapes are present."""

    reduce_contacts: bool = True
    """Merge patch triangles into fewer representative contacts."""

    buffer_fraction: float = 1.0
    """GPU buffer size relative to the worst-case estimate. Range: (0, 1]."""

    buffer_mult_broad: int = 1
    """Multiplier for the broad-phase block pair buffer, applied after ``buffer_fraction``.

    Increase when a broad phase overflow warning is reported.
    """

    buffer_mult_iso: int = 1
    """Multiplier for the iso-surface extraction buffers, applied after ``buffer_fraction``.

    Increase when an iso subblock or iso voxel overflow warning is reported.
    """

    buffer_mult_contact: int = 1
    """Multiplier for the face contact buffer, applied after ``buffer_fraction``.

    Increase when a face contact overflow warning is reported.
    """

    normal_matching: bool = True
    """Align reduced contact normals with the aggregate force direction."""

    anchor_contact: bool = False
    """Add an extra contact at the center of pressure per normal bin."""

    margin_contact_area: float = 1.0e-2
    """Area used for non-penetrating margin contacts [m²]."""

    output_contact_surface: bool = False
    """Export contact patch triangles for visualization."""

    mc_edge_clamp_min: float = 0.02
    """Marching-cubes edge clamp parameter."""


@dataclass
class CollisionConfig:
    """Configuration for the Newton collision pipeline."""

    broad_phase: Literal["nxn", "sap", "explicit"] = "explicit"
    """Broad-phase pair generation mode."""

    rigid_contact_max: int | None = None
    """Maximum number of rigid contacts to allocate, or None to derive it from the model.

    Must be at least the per-world contact capacity of the solver, such as
    :attr:`MuJoCoSolverConfig.nconmax`.
    """

    hydroelastic: HydroelasticConfig = field(default_factory=HydroelasticConfig)
    """Hydroelastic contact runtime settings."""
