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

"""Tests for the deprecated Newton tensor implementation boundary."""

import importlib

import numpy as np
import omni.kit.test
import warp as wp


class TestDeprecatedNewtonTensors(omni.kit.test.AsyncTestCase):
    """Tests for the legacy internal tensor package."""

    async def test_cache_body_com_uses_view_indices(self) -> None:
        """Cache body COM data at view indices instead of model indices."""
        kernels = importlib.import_module("isaacsim.physics.newton.impl.tensors.kernels")
        expected = np.arange(14, dtype=np.float32).reshape(2, 7)
        tensor = wp.array(expected, dtype=wp.float32, device="cpu")
        tensor_indices = wp.array([0, 1], dtype=wp.int32, device="cpu")
        model_body_indices = wp.array([5, 6], dtype=wp.int32, device="cpu")
        cache = wp.zeros((7, 7), dtype=wp.float32, device="cpu")

        wp.launch(
            kernels.cache_body_com,
            dim=2,
            inputs=[tensor, tensor_indices, None, model_body_indices],
            outputs=[cache],
            device="cpu",
        )

        cached = cache.numpy()
        np.testing.assert_array_equal(cached[:2], expected)
        np.testing.assert_array_equal(cached[2:], np.zeros((5, 7), dtype=np.float32))

    async def test_legacy_tensor_package_exports_nothing(self) -> None:
        """Warn on legacy imports without exporting tensor APIs."""
        module = importlib.import_module("isaacsim.physics.newton.impl.tensors")

        with self.assertWarnsRegex(DeprecationWarning, "use the isaacsim.physics.newton.tensors extension"):
            importlib.reload(module)

        self.assertEqual(module.__all__, [])
        self.assertFalse(hasattr(module, "create_simulation_view"))
