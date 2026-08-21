# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for HolonomicController runtime configuration refresh."""

import numpy as np
import omni.graph.core as og
import omni.graph.core.tests as ogts


class TestHolonomicControllerInputRefresh(ogts.OmniGraphTestCase):
    """Verify changed wheel geometry is applied without resetting the graph."""

    async def test_wheel_radius_change_rebuilds_controller(self) -> None:
        graph, [controller], _, _ = og.Controller.edit(
            {"graph_path": "/ActionGraph"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("HolonomicController", "isaacsim.robot.wheeled_robots.HolonomicController"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("HolonomicController.inputs:wheelRadius", [0.04, 0.04, 0.04]),
                    (
                        "HolonomicController.inputs:wheelPositions",
                        [
                            [-0.0980432, 0.000636773, -0.050501],
                            [0.0493475, -0.084525, -0.050501],
                            [0.0495291, 0.0856937, -0.050501],
                        ],
                    ),
                    (
                        "HolonomicController.inputs:wheelOrientations",
                        [[0, 0, 0, 1], [0.866, 0, 0, -0.5], [0.866, 0, 0, 0.5]],
                    ),
                    ("HolonomicController.inputs:mecanumAngles", [90, 90, 90]),
                    ("HolonomicController.inputs:inputVelocity", [1.0, 1.0, 0.1]),
                ],
            },
        )

        await og.Controller.evaluate(graph)
        first = np.asarray(og.Controller(og.Controller.attribute("outputs:jointVelocityCommand", controller)).get())

        og.Controller(og.Controller.attribute("inputs:wheelRadius", controller)).set([0.08, 0.08, 0.08])
        await og.Controller.evaluate(graph)
        second = np.asarray(og.Controller(og.Controller.attribute("outputs:jointVelocityCommand", controller)).get())

        self.assertTrue(np.all(np.isfinite(first)))
        self.assertTrue(np.all(np.isfinite(second)))
        self.assertTrue(np.allclose(second, first / 2.0, rtol=1e-4, atol=1e-4))
