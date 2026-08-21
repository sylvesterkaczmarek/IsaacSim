# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for CreateViewport when no viewport window is available."""

from types import SimpleNamespace
from unittest.mock import patch

import omni.kit.test
from isaacsim.core.nodes.ogn.python.nodes.OgnIsaacCreateViewport import (
    OgnIsaacCreateViewport,
    OgnIsaacCreateViewportInternalState,
)


class TestCreateViewportMissingWindow(omni.kit.test.AsyncTestCase):
    """Verify viewportId zero fails cleanly when there is no active viewport."""

    async def test_missing_active_viewport_returns_false(self) -> None:
        state = OgnIsaacCreateViewportInternalState()
        db = SimpleNamespace(
            per_instance_state=state,
            inputs=SimpleNamespace(name="", viewportId=0),
            outputs=SimpleNamespace(viewport=None, execOut=None),
        )

        with patch(
            "isaacsim.core.nodes.ogn.python.nodes.OgnIsaacCreateViewport.get_active_viewport_window",
            return_value=None,
        ):
            self.assertFalse(OgnIsaacCreateViewport.compute(db))

        self.assertIsNone(state.window)
        self.assertIsNone(db.outputs.viewport)
        self.assertIsNone(db.outputs.execOut)
