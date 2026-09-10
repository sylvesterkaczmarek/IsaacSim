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

"""Sample target prim positions uniformly inside a spherical shell."""

from typing import Any

import numpy as np
import omni.graph.core as og
import omni.replicator.core as rep
import omni.usd
from pxr import Sdf, UsdGeom


def sample_points_between_spheres(radius1: float, radius2: float, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample random 3D points uniformly between two concentric spheres.

    Args:
        radius1: First shell boundary, expected to be nonnegative after node validation.
        radius2: Second shell boundary, expected to be positive after node validation. The helper orders the
            validated boundaries before sampling.
        count: Nonnegative number of points to sample.
        rng: Random number generator used for sampling.

    Returns:
        Cartesian point coordinates with shape ``(count, 3)``.
    """
    if radius1 > radius2:
        radius1, radius2 = radius2, radius1
    phi = rng.uniform(0, 2 * np.pi, count)
    costheta = rng.uniform(-1, 1, count)
    theta = np.arccos(costheta)
    r = rng.uniform(radius1**3, radius2**3, count) ** (1 / 3)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)


class OgnSampleBetweenSpheresInternalState:
    """Store a Replicator-aware random number generator across node evaluations."""

    def __init__(self) -> None:
        self.rng = rep.rng.ReplicatorRNG()


class OgnSampleBetweenSpheres:
    """Replicator OmniGraph node that writes random positions between two radii."""

    @staticmethod
    def internal_state() -> OgnSampleBetweenSpheresInternalState:
        """Create the node's persistent internal state.

        Returns:
            Persistent state for the node.
        """
        return OgnSampleBetweenSpheresInternalState()

    @staticmethod
    def release(node: og.Node) -> None:
        """Release subscriptions owned by the node's random number generator.

        Args:
            node: OmniGraph node whose subscriptions are released.
        """
        rep.rng.release(node.get_prim_path())

    @staticmethod
    def compute(db: Any) -> bool:
        """Move each input prim to a uniformly sampled point between two concentric spheres.

        The node accepts the target prim paths from ``inputs:prims`` and two shell radii from
        ``inputs:radius1`` and ``inputs:radius2``. Empty prim input disables ``outputs:execOut`` and
        returns ``False`` without logging. Otherwise, the node first requires ``radius1`` to be
        nonnegative and ``radius2`` to be positive, then orders valid radii for sampling. Positions
        are written to ``xformOp:translate`` on each target prim. Invalid prims or invalid radii log
        an error, disable ``outputs:execOut``, and return ``False``.

        Args:
            db: OmniGraph database object containing node inputs and outputs.

        Returns:
            True when all target prims are sampled and updated, False otherwise.
        """
        prim_paths = db.inputs.prims
        if len(prim_paths) == 0:
            db.outputs.execOut = og.ExecutionAttributeState.DISABLED
            return False

        radius1 = db.inputs.radius1
        radius2 = db.inputs.radius2
        if radius1 < 0 or radius2 <= 0:
            db.log_error(f"Radius must be positive and larger radius larger than 0, got {radius1} and {radius2}")
            db.outputs.execOut = og.ExecutionAttributeState.DISABLED
            return False

        stage = omni.usd.get_context().get_stage()
        prims = [stage.GetPrimAtPath(str(path)) for path in prim_paths]

        try:
            for prim in prims:
                if not prim.IsValid():
                    raise ValueError(f"Invalid prim path: {prim.GetPath()}")
                if not UsdGeom.Xformable(prim):
                    raise ValueError(
                        f"Expected prim at {prim.GetPath()} to be an Xformable prim but got type "
                        f"{prim.GetTypeName()}"
                    )
                if not prim.HasAttribute("xformOp:translate"):
                    UsdGeom.Xformable(prim).AddTranslateOp()
        except Exception as error:
            db.log_error(str(error))
            db.outputs.execOut = og.ExecutionAttributeState.DISABLED
            return False

        state = db.shared_state
        if state.rng.seed != db.inputs.seed:
            node_id = (
                db.node.get_attribute("inputs:nodeId").get() if db.node.get_attribute_exists("inputs:nodeId") else 0
            )
            state.rng.initialize(db.inputs.seed, db.node, node_id)

        positions = sample_points_between_spheres(radius1, radius2, len(prims), state.rng.generator)
        with Sdf.ChangeBlock():
            for prim, position in zip(prims, positions):
                prim.GetAttribute("xformOp:translate").Set(tuple(position))

        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True
