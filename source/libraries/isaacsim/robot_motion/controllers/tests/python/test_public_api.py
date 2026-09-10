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

"""Kit-free smoke tests for the ``isaacsim.robot_motion.controllers`` library.

These tests verify that the relocated controller package preserves its public API and imports without loading
Carbonite or Omniverse Kit modules. Behavioral and simulation tests remain in the existing Kit extension.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_FROZEN_PUBLIC_API = [
    "AckermannController",
    "DifferentialDriveController",
    "HolonomicController",
]


def test_public_import_path_and_frozen_all() -> None:
    """Verify the public import path and frozen API names."""
    import isaacsim.robot_motion.controllers as controllers

    assert set(controllers.__all__) == set(_FROZEN_PUBLIC_API)
    for symbol in _FROZEN_PUBLIC_API:
        assert hasattr(controllers, symbol), f"missing public symbol: {symbol}"


def test_import_is_kit_free() -> None:
    """Verify a fresh controller import does not load Kit modules."""
    program = (
        "import sys\n"
        "import isaacsim.robot_motion.controllers  # noqa: F401\n"
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
