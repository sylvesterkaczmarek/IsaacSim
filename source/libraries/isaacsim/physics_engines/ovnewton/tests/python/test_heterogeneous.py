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

"""Validate heterogeneous articulation batches with the Newton engine."""

from __future__ import annotations

import os
import sys
from typing import Protocol

import _physics_setup  # noqa: F401
import numpy as np
import pytest

_TENSORS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TENSORS_DIR not in sys.path:
    sys.path.append(_TENSORS_DIR)

import warp_utils as wp_utils  # noqa: E402
from _legacy_runner import (  # noqa: E402
    cpu_device,
    gpu_device,
    gpu_only,
    run_scenario,
    unimplemented_placeholder,
)
from common.heterogeneous import (  # noqa: E402
    HeterogeneousBaseDynamicsCommon,
    HeterogeneousSceneArticulationsCommon,
)

_GAP_HETEROGENEOUS_CABINET = (
    "Newton builds an env-inconsistent state for the cabinet articulation in the heterogeneous "
    "cabinet+franka scene: the cross-env link-transform divergence blows up across env replicas "
    "on both CPU and GPU (the franka in the same scene stays ~1e-6). It is a Newton engine "
    "determinism bug for heterogeneous batches (also intermittently aborts in parse_usd on GPU); "
    "link-transforms surfaces it faithfully, it is not a tensor-read issue -- follow-up"
)


class _NewtonArray(Protocol):
    shape: tuple[int, ...]

    def numpy(self) -> np.ndarray: ...

    def assign(self, values: np.ndarray) -> None: ...


class _NewtonModel(Protocol):
    articulation_start: _NewtonArray
    joint_qd_start: _NewtonArray


class _NewtonState(Protocol):
    joint_q: object
    joint_qd: _NewtonArray


class _NewtonStage(Protocol):
    model: _NewtonModel
    state_0: _NewtonState


class _NewtonBackend(Protocol):
    articulation_indices: _NewtonArray


class _NewtonArticulationView(Protocol):
    _newton_stage: _NewtonStage
    _backend: _NewtonBackend
    count: int

    def get_data(self, implementation: str) -> _NewtonArray: ...

    def get_metadata(self, name: str) -> int: ...


class HeterogeneousBaseJacobianSelfConsistency(HeterogeneousBaseDynamicsCommon):
    """Validate Newton Jacobians against their corresponding link velocities.

    The scenario assigns deterministic generalized velocities, evaluates
    forward kinematics, and verifies ``J @ joint_qd == body_qd`` for each Ant
    and cart-pole articulation. It also verifies that padding rows beyond an
    articulation's links remain zero.
    """

    def _assert_jacobian_reproduces_link_velocity(self, view: _NewtonArticulationView) -> None:
        model = view._newton_stage.model
        joint_qd = view._newton_stage.state_0.joint_qd.numpy()
        articulation_start = model.articulation_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        articulation_indices = view._backend.articulation_indices.numpy()

        j = view.get_data("jacobians").numpy()
        num_links = view.get_metadata("num-links")
        link_vel = view.get_data("link-velocities").numpy().reshape(view.count, num_links, 6)
        num_rows = num_links * 6
        for env in range(view.count):
            # The exact generalized velocity this articulation's jacobian was
            # built to map: the state.joint_qd slice for its joints, already in
            # the jacobian's column order, padded to the model-max width.
            art = int(articulation_indices[env])
            lo = int(joint_qd_start[int(articulation_start[art])])
            hi = int(joint_qd_start[int(articulation_start[art + 1])])
            gen_qd = np.zeros(j.shape[2], dtype=np.float32)
            gen_qd[: hi - lo] = joint_qd[lo:hi]
            pred = j[env] @ gen_qd
            assert wp_utils.wp_allclose(
                pred[:num_rows], link_vel[env].reshape(-1), rtol=1e-2, atol=1e-2
            ), f"env {env}: J @ joint_qd must reproduce the link body velocity"
            assert np.all(
                np.abs(pred[num_rows:]) < 1e-2
            ), f"env {env}: jacobian rows past this articulation's links must be zero"

    def on_physics_step(self, sim: object, stepno: int, dt: float) -> None:
        """Inject generalized velocities and validate each articulation Jacobian.

        Args:
            sim: Backend simulation view managed by the scenario runner.
            stepno: Zero-based simulation step number.
            dt: Simulated time interval in seconds.

        """
        if stepno != 3:
            return
        import newton

        state = self.ants._newton_stage.state_0
        model = self.ants._newton_stage.model
        # Consistent state for every env: inject a deterministic non-zero joint
        # velocity, then eval_fk so body_qd matches joint_qd (both robot types).
        joint_qd = np.linspace(0.1, 0.7, state.joint_qd.shape[0]).astype(np.float32)
        state.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        self._assert_jacobian_reproduces_link_velocity(self.ants)
        self._assert_jacobian_reproduces_link_velocity(self.cartpoles)
        self.finish()


class TestHeterogeneousBaseArticulationsDynamics:
    """Validate heterogeneous base articulations dynamics with Newton."""

    def test_heterogeneous_base_articulation_dynamics_newton_cc(self) -> None:
        """Verify heterogeneous base articulation dynamics on the Newton CPU pipeline."""
        run_scenario(self, HeterogeneousBaseJacobianSelfConsistency, "newton", cpu_device())

    @gpu_only
    def test_heterogeneous_base_articulation_dynamics_newton_gg(self) -> None:
        """Verify heterogeneous base articulation dynamics on the Newton GPU pipeline."""
        run_scenario(self, HeterogeneousBaseJacobianSelfConsistency, "newton", gpu_device())


class TestHeterogeneousSceneArticulations:
    """Retain skipped cabinet-and-Franka batch checks for Newton.

    Both device variants are skipped because cabinet state diverges across
    replicated environments.
    """

    @pytest.mark.skip(reason=_GAP_HETEROGENEOUS_CABINET)
    def test_heterogeneous_scene_articulation_newton_cc(self) -> None:
        """Verify heterogeneous scene articulation on the Newton CPU pipeline."""
        run_scenario(self, HeterogeneousSceneArticulationsCommon, "newton", cpu_device())

    @pytest.mark.skip(reason=_GAP_HETEROGENEOUS_CABINET)
    @gpu_only
    def test_heterogeneous_scene_articulation_newton_gg(self) -> None:
        """Verify heterogeneous scene articulation on the Newton GPU pipeline."""
        run_scenario(self, HeterogeneousSceneArticulationsCommon, "newton", gpu_device())


@unimplemented_placeholder
class TestPortShells:
    """Record unimplemented heterogeneous articulation scenarios for Newton."""

    def test_heterogeneous_base_articulation_link_transforms_newton_cc(self) -> None:
        """Record the unimplemented heterogeneous base articulation link transforms case for Newton."""

    def test_heterogeneous_base_articulation_link_transforms_newton_gg(self) -> None:
        """Record the unimplemented heterogeneous base articulation link transforms case for Newton."""
