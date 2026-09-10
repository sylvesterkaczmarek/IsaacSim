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

"""Tests for the autograd-preserving theta decode used by differentiable bridges."""

from __future__ import annotations

import numpy as np
import omni.kit.test
import torch
from isaacsim.robot_setup.sysid.inertia_param import (
    LOG_CHOLESKY_DIM,
    inertia_matrix_to_log_cholesky,
    log_cholesky_jacobian,
    log_cholesky_to_inertia_matrix,
    log_cholesky_to_inertia_matrix_torch,
)
from isaacsim.robot_setup.sysid.parameter_apply import decode_theta_to_apply_state
from isaacsim.robot_setup.sysid.parameter_apply_torch import (
    ThetaDecodeBaselines,
    decode_theta_batch_to_tensors,
    validate_differentiable_entries,
)
from isaacsim.robot_setup.sysid.parameter_types import (
    GLOBAL_DOF_INDEX,
    GLOBAL_LINK_INDEX,
    SysIdParameterEntry,
    SysIdParameterType,
    build_extended_parameter_specs,
)

_NUM_DOF = 3
_NUM_LINKS = 2


def _entry(
    ptype: SysIdParameterType,
    dof_index: int = GLOBAL_DOF_INDEX,
    link_index: int = GLOBAL_LINK_INDEX,
    component_index: int = 0,
) -> SysIdParameterEntry:
    return SysIdParameterEntry(
        param_type=ptype,
        dof_index=dof_index,
        link_index=link_index,
        component_index=component_index,
    )


def _mixed_entries() -> list[SysIdParameterEntry]:
    entries = []
    for dof_index in range(_NUM_DOF):
        entries.append(_entry(SysIdParameterType.JOINT_FRICTION, dof_index=dof_index))
        entries.append(_entry(SysIdParameterType.JOINT_STIFFNESS, dof_index=dof_index))
        entries.append(_entry(SysIdParameterType.JOINT_DAMPING, dof_index=dof_index))
    entries.append(_entry(SysIdParameterType.LINK_MASS))
    entries.append(_entry(SysIdParameterType.LINK_MASS, link_index=1))
    entries.append(_entry(SysIdParameterType.LINK_COM_OFFSET_X, link_index=0))
    entries.append(_entry(SysIdParameterType.LINK_COM_OFFSET_Z, link_index=1))
    for component_index in range(LOG_CHOLESKY_DIM):
        entries.append(
            _entry(
                SysIdParameterType.LINK_INERTIA_LOG_CHOLESKY,
                link_index=1,
                component_index=component_index,
            )
        )
    return entries


def _baselines(dtype: torch.dtype = torch.float32) -> ThetaDecodeBaselines:
    inertia = torch.stack(
        (
            torch.diag(torch.tensor([0.02, 0.03, 0.04], dtype=torch.float64)),
            torch.diag(torch.tensor([0.05, 0.06, 0.07], dtype=torch.float64)),
        )
    )
    inertia_lc = torch.stack(
        tuple(
            torch.as_tensor(inertia_matrix_to_log_cholesky(inertia[i].numpy()), dtype=torch.float64)
            for i in range(_NUM_LINKS)
        )
    )
    return ThetaDecodeBaselines(
        stiffness=torch.tensor([10.0, 20.0, 30.0], dtype=dtype),
        damping=torch.tensor([1.0, 2.0, 3.0], dtype=dtype),
        friction=torch.tensor([0.1, 0.2, 0.3], dtype=dtype),
        link_mass=torch.tensor([1.5, 2.5], dtype=dtype),
        link_com=torch.tensor([[0.01, 0.0, 0.02], [0.0, 0.03, 0.0]], dtype=dtype),
        link_inertia=inertia.to(dtype),
        link_inertia_lc=inertia_lc.to(dtype),
    )


class TestLogCholeskyTorch(omni.kit.test.AsyncTestCase):
    """Verify the Torch Log-Cholesky implementation."""

    async def test_matches_numpy_decode(self) -> None:
        """Match the NumPy decoder for a single inertia tensor."""
        theta = np.array([0.1, -0.3, 0.2, 0.05, -0.1, 0.4], dtype=np.float64)
        expected = log_cholesky_to_inertia_matrix(theta)
        actual = log_cholesky_to_inertia_matrix_torch(torch.as_tensor(theta)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    async def test_batched_shapes(self) -> None:
        """Preserve arbitrary leading batch dimensions."""
        theta = torch.randn(4, 5, LOG_CHOLESKY_DIM, dtype=torch.float64)
        result = log_cholesky_to_inertia_matrix_torch(theta)
        self.assertEqual(tuple(result.shape), (4, 5, 3, 3))

    async def test_gradcheck(self) -> None:
        """Pass Torch's numerical autograd check."""
        theta = torch.randn(LOG_CHOLESKY_DIM, dtype=torch.float64, requires_grad=True) * 0.3
        theta = theta.detach().requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(log_cholesky_to_inertia_matrix_torch, (theta,)))

    async def test_jacobian_matches_analytic(self) -> None:
        """Match the analytical inertia Jacobian."""
        theta = torch.tensor([0.1, -0.3, 0.2, 0.05, -0.1, 0.4], dtype=torch.float64, requires_grad=True)
        jac_torch = torch.autograd.functional.jacobian(
            lambda t: log_cholesky_to_inertia_matrix_torch(t).reshape(-1), theta
        )
        jac_analytic = log_cholesky_jacobian(theta.detach().numpy())
        np.testing.assert_allclose(jac_torch.numpy(), jac_analytic, rtol=1e-10, atol=1e-12)

    async def test_rejects_bad_trailing_dim(self) -> None:
        """Reject tensors without six Log-Cholesky parameters."""
        with self.assertRaises(ValueError):
            log_cholesky_to_inertia_matrix_torch(torch.zeros(5))


class TestDecodeThetaBatchToTensors(omni.kit.test.AsyncTestCase):
    """Verify batched, autograd-preserving parameter decoding."""

    async def test_sparse_optional_vectors_mark_unselected_dofs_with_nan(self) -> None:
        """Require consumers to preserve baselines at unselected optional-vector DOFs."""
        entries = [
            _entry(SysIdParameterType.JOINT_ARMATURE, dof_index=0),
            _entry(SysIdParameterType.JOINT_INTEGRAL_GAIN, dof_index=1),
            _entry(SysIdParameterType.ACTUATOR_COMMAND_DELAY_SECONDS, dof_index=2),
        ]

        state = decode_theta_to_apply_state(
            _NUM_DOF,
            _NUM_LINKS,
            entries,
            np.asarray([0.25, 0.5, 0.02], dtype=np.float32),
        )

        np.testing.assert_equal(state.joint_armature, np.asarray([0.25, np.nan, np.nan], dtype=np.float32))
        np.testing.assert_equal(state.joint_integral_gain, np.asarray([np.nan, 0.5, np.nan], dtype=np.float32))
        np.testing.assert_equal(
            state.actuator_command_delay_seconds,
            np.asarray([np.nan, np.nan, 0.02], dtype=np.float32),
        )

    async def test_matches_numpy_decode(self) -> None:
        """Match the scalar NumPy decoder across candidate rows."""
        entries = _mixed_entries()
        baselines = _baselines()
        generator = torch.Generator().manual_seed(7)
        theta = torch.rand(3, len(entries), generator=generator) * 1.5 + 0.1
        decoded = decode_theta_batch_to_tensors(theta, entries, baselines)

        baseline_lc = {i: baselines.link_inertia_lc[i].double().numpy() for i in range(_NUM_LINKS)}
        for row in range(theta.shape[0]):
            state = decode_theta_to_apply_state(
                num_dof=_NUM_DOF,
                num_links=_NUM_LINKS,
                param_entries=entries,
                theta_row=theta[row],
                baseline_link_inertia_lc=baseline_lc,
            )
            np.testing.assert_allclose(
                decoded.friction[row].numpy(),
                baselines.friction.numpy() * state.friction_scale,
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                decoded.ke[row].numpy(),
                baselines.stiffness.numpy() * state.stiffness_scale,
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                decoded.kd[row].numpy(),
                baselines.damping.numpy() * state.damping_scale,
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                decoded.mass[row].numpy(),
                baselines.link_mass.numpy() * state.link_mass_scale,
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                decoded.com[row].numpy(),
                baselines.link_com.numpy() + state.link_com_delta,
                rtol=1e-5,
                atol=1e-7,
            )
            # Link 1 is touched by inertia entries; link 0 keeps the baseline tensor verbatim.
            np.testing.assert_allclose(
                decoded.inertia[row, 1].numpy().reshape(-1),
                state.link_inertia_flat[1],
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                decoded.inertia[row, 0].numpy(),
                baselines.link_inertia[0].numpy(),
                rtol=0.0,
            )

    async def test_graph_connectivity(self) -> None:
        """Keep every selected parameter connected to autograd."""
        entries = _mixed_entries()
        baselines = _baselines()
        theta = torch.full((2, len(entries)), 0.9, requires_grad=True)
        decoded = decode_theta_batch_to_tensors(theta, entries, baselines)
        total = (
            decoded.ke.sum()
            + decoded.kd.sum()
            + decoded.friction.sum()
            + decoded.mass.sum()
            + decoded.com.sum()
            + decoded.inertia.sum()
        )
        self.assertIsNotNone(total.grad_fn)
        total.backward()
        self.assertIsNotNone(theta.grad)
        self.assertTrue(torch.all(torch.isfinite(theta.grad)))
        # Every theta column influences at least one decoded output.
        self.assertTrue(torch.all(theta.grad.abs().sum(dim=0) > 0.0))

    async def test_inertia_graph_connectivity(self) -> None:
        """Keep every inertia parameter connected to the decoded tensor."""
        entries = [
            _entry(SysIdParameterType.LINK_INERTIA_LOG_CHOLESKY, link_index=1, component_index=component)
            for component in range(LOG_CHOLESKY_DIM)
        ]
        theta = torch.full((2, LOG_CHOLESKY_DIM), 0.9, requires_grad=True)

        inertia = decode_theta_batch_to_tensors(theta, entries, _baselines()).inertia

        self.assertIsNotNone(inertia.grad_fn)
        inertia.sum().backward()
        self.assertIsNotNone(theta.grad)
        self.assertTrue(torch.all(torch.isfinite(theta.grad)))
        self.assertTrue(torch.all(theta.grad.abs().sum(dim=0) > 0.0))

    async def test_global_mass_scale_broadcasts(self) -> None:
        """Broadcast a global mass scale to every link."""
        entries = [_entry(SysIdParameterType.LINK_MASS)]
        baselines = _baselines()
        theta = torch.tensor([[2.0]])
        decoded = decode_theta_batch_to_tensors(theta, entries, baselines)
        np.testing.assert_allclose(decoded.mass[0].numpy(), baselines.link_mass.numpy() * 2.0, rtol=1e-6)

    async def test_global_inertia_component_updates_every_link(self) -> None:
        """Apply a global Log-Cholesky component to every link in both decoders."""
        entries = [_entry(SysIdParameterType.LINK_INERTIA_LOG_CHOLESKY, component_index=0)]
        baselines = _baselines()
        theta = torch.tensor([[0.7]])

        decoded = decode_theta_batch_to_tensors(theta, entries, baselines)
        baseline_lc = {index: baselines.link_inertia_lc[index].double().numpy() for index in range(_NUM_LINKS)}
        scalar = decode_theta_to_apply_state(
            _NUM_DOF,
            _NUM_LINKS,
            entries,
            theta[0],
            baseline_link_inertia_lc=baseline_lc,
        )

        self.assertEqual(set(scalar.link_inertia_flat), {0, 1})
        torch.testing.assert_close(decoded.inertia[0, 0, 0, 0], decoded.inertia[0, 1, 0, 0])
        self.assertNotEqual(float(decoded.inertia[0, 0, 0, 0]), float(baselines.link_inertia[0, 0, 0]))

    async def test_per_link_mass_registry_omits_shadowed_global_mass(self) -> None:
        """Omit the global mass row when per-link mass rows are requested."""
        specs = build_extended_parameter_specs(
            _NUM_DOF,
            _NUM_LINKS,
            include_basic=True,
            include_per_link_mass=True,
        )

        mass_entries = [spec.entry for spec in specs if spec.param_type == SysIdParameterType.LINK_MASS]

        self.assertEqual([entry.link_index for entry in mass_entries], [0, 1])
        self.assertNotIn(GLOBAL_LINK_INDEX, [entry.link_index for entry in mass_entries])

    async def test_negative_scales_clamped(self) -> None:
        """Clamp negative multiplicative scales to zero."""
        entries = [_entry(SysIdParameterType.JOINT_FRICTION, dof_index=0)]
        baselines = _baselines()
        theta = torch.tensor([[-3.0]])
        decoded = decode_theta_batch_to_tensors(theta, entries, baselines)
        self.assertEqual(float(decoded.friction[0, 0]), 0.0)

    async def test_rejects_unsupported_entries(self) -> None:
        """Reject parameters unsupported by differentiable decoding."""
        entries = [_entry(SysIdParameterType.ACTUATOR_COMMAND_DELAY_SECONDS)]
        self.assertEqual(
            validate_differentiable_entries(entries),
            ["actuator_command_delay_seconds"],
        )
        with self.assertRaises(ValueError):
            decode_theta_batch_to_tensors(torch.zeros(1, 1), entries, _baselines())

    async def test_rejects_shape_mismatch(self) -> None:
        """Reject malformed candidate batches."""
        entries = [_entry(SysIdParameterType.JOINT_FRICTION, dof_index=0)]
        with self.assertRaises(ValueError):
            decode_theta_batch_to_tensors(torch.zeros(3), entries, _baselines())
        with self.assertRaises(ValueError):
            decode_theta_batch_to_tensors(torch.zeros(1, 2), entries, _baselines())

    async def test_rejects_out_of_range_parameter_indices(self) -> None:
        """Fail closed when a parameter column cannot address a simulation value."""
        cases = (
            _entry(SysIdParameterType.JOINT_FRICTION, dof_index=_NUM_DOF),
            _entry(SysIdParameterType.LINK_MASS, link_index=_NUM_LINKS),
            _entry(SysIdParameterType.LINK_COM_OFFSET_X, link_index=_NUM_LINKS),
            _entry(SysIdParameterType.LINK_INERTIA_LOG_CHOLESKY, link_index=_NUM_LINKS),
            _entry(
                SysIdParameterType.LINK_INERTIA_LOG_CHOLESKY,
                component_index=LOG_CHOLESKY_DIM,
            ),
        )
        for entry in cases:
            with self.subTest(entry=entry):
                with self.assertRaisesRegex(ValueError, "index"):
                    decode_theta_to_apply_state(
                        _NUM_DOF,
                        _NUM_LINKS,
                        [entry],
                        np.asarray([1.0]),
                    )
                with self.assertRaisesRegex(ValueError, "index"):
                    decode_theta_batch_to_tensors(torch.ones(1, 1), [entry], _baselines())
