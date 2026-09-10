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

"""Kit-free public API tests for ``isaacsim.robot_motion.cumotion``."""

from __future__ import annotations

import subprocess
import sys

import pytest

_FROZEN_PUBLIC_API = [
    "load_cumotion_robot",
    "load_cumotion_supported_robot",
    "CumotionDebugVisualizer",
    "CumotionRobot",
    "CumotionWorldInterface",
    "RmpFlowController",
    "GraphBasedMotionPlanner",
    "TrajectoryGenerator",
    "TrajectoryOptimizer",
    "CumotionTrajectory",
]


def test_bundled_cumotion_runtime() -> None:
    """Verify the robot-motion distribution provides the pinned native cuMotion runtime."""
    import cumotion

    assert cumotion.__version__ == "1.1.0a2"


def test_public_import_path_and_frozen_all() -> None:
    """Verify the public import path and frozen API names."""
    import isaacsim.robot_motion.cumotion as cumotion

    assert set(cumotion.__all__) == set(_FROZEN_PUBLIC_API)
    for symbol in _FROZEN_PUBLIC_API:
        assert hasattr(cumotion, symbol), f"missing public symbol: {symbol}"


def test_import_is_kit_free() -> None:
    """Verify a fresh import does not load Carbonite or Omniverse Kit modules."""
    program = (
        "import sys\n"
        "import isaacsim.robot_motion.cumotion  # noqa: F401\n"
        "leaked = sorted(m for m in sys.modules if m == 'carb' or m.split('.')[0] == 'omni')\n"
        "assert not leaked, 'Kit modules leaked into a Kit-free import: %r' % leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        "Kit-free import check failed.\n" f"stdout:\n{result.stdout}\n" f"stderr:\n{result.stderr}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
