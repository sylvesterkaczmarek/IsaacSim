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

"""Test the Robot Poser kinematics and IK modules."""

from __future__ import annotations

import numpy as np
import omni.kit.test
from isaacsim.robot.poser import (
    IKSolver,
    IKSolverLM,
    IKSolverRegistry,
    Joint,
    KinematicChain,
    Transform,
    ik_lm,
    pose_error,
)


def _create_revolute_chain() -> KinematicChain:
    """Create a one-joint chain without requiring a USD stage.

    Returns:
        Chain with a revolute Z joint and a unit-length tip offset.
    """
    joint = Joint(
        w=np.array([0.0, 0.0, 1.0]),
        v=np.zeros(3),
        home=Transform(),
        tip=Transform(t=[1.0, 0.0, 0.0]),
        prim_path="/Robot/joint",
        lower=-np.pi,
        upper=np.pi,
        is_revolute=True,
    )
    chain = object.__new__(KinematicChain)
    chain._joints = [joint]
    chain._debug = False
    return chain


class TestKinematicsOwnership(omni.kit.test.AsyncTestCase):
    """Tests for the Robot Poser-owned kinematics API."""

    async def test_public_symbols_resolve_from_robot_poser(self) -> None:
        """Verify the public kinematics symbols are implemented by Robot Poser."""
        for symbol in (IKSolver, IKSolverLM, IKSolverRegistry, Joint, KinematicChain, Transform):
            self.assertTrue(
                symbol.__module__.startswith("isaacsim.robot.poser"),
                f"{symbol.__name__} is implemented by {symbol.__module__}",
            )

    async def test_compute_fk_and_jacobian(self) -> None:
        """Verify FK and Jacobian computation on a simple revolute chain."""
        chain = _create_revolute_chain()
        q = np.array([np.pi / 2.0])

        transform, per_joint = chain.compute_fk(q)
        fused_transform, jacobian = chain.compute_fk_and_jacobian(q)

        np.testing.assert_allclose(transform.t, [0.0, 1.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(fused_transform.t, transform.t, atol=1e-8)
        np.testing.assert_allclose(fused_transform.q, transform.q, atol=1e-8)
        np.testing.assert_allclose(jacobian[:3, 0], [0.0, 0.0, 1.0], atol=1e-8)
        self.assertEqual(len(per_joint), 1)
        self.assertEqual(jacobian.shape, (6, 1))

    async def test_pose_error_is_zero_for_matching_transforms(self) -> None:
        """Verify pose error is zero when desired and actual transforms match."""
        transform = Transform(t=[1.0, 2.0, 3.0], q=[1.0, 0.0, 0.0, 0.0])

        np.testing.assert_allclose(pose_error(transform, transform), np.zeros(6), atol=1e-12)

    async def test_lm_solver_converges_on_revolute_chain(self) -> None:
        """Verify the relocated LM solver converges on a reachable target."""
        chain = _create_revolute_chain()
        expected_q = np.array([0.4])
        target, _ = chain.compute_fk(expected_q)

        solved_q = ik_lm(chain, np.zeros(1), target, iters=100, tol=1e-10)
        solved_transform, _ = chain.compute_fk(solved_q)

        np.testing.assert_allclose(solved_q, expected_q, atol=1e-4)
        self.assertLess(float(np.linalg.norm(pose_error(target, solved_transform))), 1e-4)

    async def test_solver_registry_supports_custom_solver(self) -> None:
        """Verify Robot Poser can register and construct a custom IK solver."""

        class TestSolver(IKSolver):
            """Minimal solver used to validate registry ownership."""

            def solve(
                self,
                chain: KinematicChain,
                target: Transform,
                q0: np.ndarray | None = None,
                **kwargs: object,
            ) -> np.ndarray:
                """Return the provided seed.

                Args:
                    chain: Kinematic chain supplied by the caller.
                    target: Target transform supplied by the caller.
                    q0: Initial joint configuration.
                    **kwargs: Additional solver parameters.

                Returns:
                    Copy of the seed, or zeros when the seed is omitted.
                """
                del target, kwargs
                return np.zeros(len(chain.joints)) if q0 is None else np.array(q0, copy=True)

        solver_name = "test-robot-poser-owner"
        IKSolverRegistry.register(solver_name, TestSolver)
        try:
            solver = IKSolverRegistry.get(solver_name)
            result = solver.solve(_create_revolute_chain(), Transform(), np.array([0.25]))
        finally:
            IKSolverRegistry._solvers.pop(solver_name, None)

        self.assertIsInstance(solver, TestSolver)
        np.testing.assert_allclose(result, [0.25])


if __name__ == "__main__":
    omni.kit.test.main()
