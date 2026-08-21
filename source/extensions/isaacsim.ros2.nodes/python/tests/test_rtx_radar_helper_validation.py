# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ROS2 RTX radar helper lifecycle validation."""

from unittest.mock import MagicMock, patch

import omni.kit.test
from isaacsim.ros2.nodes.OgnROS2RtxRadarHelper import OgnROS2RtxRadarHelper


class TestROS2RtxRadarHelperValidation(omni.kit.test.TestCase):
    """Validate early lifecycle and render-product guards."""

    def test_disabling_initialized_helper_resets_state(self) -> None:
        """Disabling an active helper must detach/reset its writer state."""
        db = MagicMock()
        db.inputs.enabled = False
        db.per_instance_state.initialized = True

        self.assertTrue(OgnROS2RtxRadarHelper.compute(db))

        db.per_instance_state.custom_reset.assert_called_once_with()

    @patch("isaacsim.ros2.nodes.OgnROS2RtxRadarHelper.ViewportManager.get_camera")
    @patch("isaacsim.ros2.nodes.OgnROS2RtxRadarHelper.omni.usd.get_context")
    def test_missing_render_product_retries_before_camera_lookup(
        self, mock_get_context: MagicMock, mock_get_camera: MagicMock
    ) -> None:
        """An invalid USD prim should be rejected before resolving the radar camera."""
        db = MagicMock()
        db.inputs.enabled = True
        db.inputs.renderProductPath = "/Render/MissingProduct"
        db.per_instance_state.initialized = False

        invalid_prim = MagicMock()
        invalid_prim.IsValid.return_value = False
        mock_get_context.return_value.get_stage.return_value.GetPrimAtPath.return_value = invalid_prim

        self.assertFalse(OgnROS2RtxRadarHelper.compute(db))

        mock_get_camera.assert_not_called()
