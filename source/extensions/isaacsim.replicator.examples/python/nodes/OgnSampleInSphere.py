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

"""Sample target prim positions uniformly inside a sphere."""

from typing import Any

import numpy as np
import omni.graph.core as og
import omni.replicator.core as rep
import omni.usd
from pxr import Sdf, UsdGeom


def sample_points_in_sphere(radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample random 3D points uniformly within a sphere volume.

    Args:
        radius: Positive sphere radius. The node validates this precondition before calling the helper.
        count: Nonnegative number of points to sample.
        rng: Random number generator used for sampling.

    Returns:
        Cartesian point coordinates with shape ``(count, 3)``.
    """
    phi = rng.uniform(0, 2 * np.pi, count)
    costheta = rng.uniform(-1, 1, count)
    theta = np.arccos(costheta)
    r = radius * (rng.random(count) ** (1 / 3))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)


class OgnSampleInSphereInternalState:
    """Store a Replicator-aware random number generator across node evaluations."""

    def __init__(self) -> None:
        self.rng = rep.rng.ReplicatorRNG()


class OgnSampleInSphere:
    """Replicator OmniGraph node that writes random positions within one radius."""

    @staticmethod
    def internal_state() -> OgnSampleInSphereInternalState:
        """Create the node's persistent internal state.

        Returns:
            Persistent state for the node.
        """
        return OgnSampleInSphereInternalState()

    @staticmethod
    def release(node: og.Node) -> None:
        """Release subscriptions owned by the node's random number generator.

        Args:
            node: OmniGraph node whose subscriptions are released.
        """
        rep.rng.release(node.get_prim_path())

    @staticmethod
    def compute(db: Any) -> bool:
        """Move each input prim to a uniformly sampled point inside a sphere.

        The node reads target prim paths from ``inputs:prims`` and the radius from ``inputs:radius``.
        It samples direction uniformly and scales radius by the cube root of a random value so points
        are distributed through volume rather than clustered near the center. Positions are written
        to ``xformOp:translate`` on each target prim. Empty prim inputs, invalid prim paths, or a
        non-positive radius disable ``outputs:execOut`` and return ``False``.

        Args:
            db: OmniGraph database object containing node inputs and outputs.

        Returns:
            True when all target prims are sampled and updated, False otherwise.
        """
        prim_paths = db.inputs.prims
        if len(prim_paths) == 0:
            db.outputs.execOut = og.ExecutionAttributeState.DISABLED
            return False

        radius = db.inputs.radius
        if radius <= 0:
            db.log_error(f"Radius must be positive, got {radius}")
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

        positions = sample_points_in_sphere(radius, len(prims), state.rng.generator)
        with Sdf.ChangeBlock():
            for prim, position in zip(prims, positions):
                prim.GetAttribute("xformOp:translate").Set(tuple(position))

        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True
