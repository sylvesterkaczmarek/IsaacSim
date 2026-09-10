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

"""Verifies IsaacGetViewportRenderProduct handles unresolved viewport names."""

import omni.kit.test
from isaacsim.core.nodes.ogn.python.nodes import OgnIsaacGetViewportRenderProduct as viewport_render_product_node


class TestGetViewportRenderProductWrapper(omni.kit.test.AsyncTestCase):
    """Verify compute behavior without building a full OmniGraph."""

    async def test_missing_viewport_returns_false_without_attribute_error(self) -> None:
        """Test that an unresolved viewport name gracefully skips execution."""

        class FakeInputs:
            viewport = "NoSuchViewport_zzz"

        class FakeOutputs:
            execOut = None
            renderProductPath = ""

        class FakeDb:
            def __init__(self) -> None:
                self.inputs = FakeInputs()
                self.outputs = FakeOutputs()
                self.per_instance_state = viewport_render_product_node.OgnIsaacGetViewportRenderProductInternalState()

        def get_missing_viewport(window_name: str) -> None:
            return None

        original_get_viewport_from_window_name = viewport_render_product_node.get_viewport_from_window_name
        viewport_render_product_node.get_viewport_from_window_name = get_missing_viewport
        try:
            db = FakeDb()

            self.assertFalse(viewport_render_product_node.OgnIsaacGetViewportRenderProduct.compute(db))

            self.assertIsNone(db.outputs.execOut)
            self.assertEqual(db.outputs.renderProductPath, "")
            self.assertIsNone(db.per_instance_state.viewport)
            self.assertFalse(db.per_instance_state.initialized)
        finally:
            viewport_render_product_node.get_viewport_from_window_name = original_get_viewport_from_window_name
