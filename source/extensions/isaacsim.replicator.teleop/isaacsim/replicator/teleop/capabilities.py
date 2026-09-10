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

"""Runtime capability discovery for optional teleop backends."""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass


def _module_available(module_name: str) -> bool:
    """Return whether a module can be resolved without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


@dataclass(frozen=True)
class TeleopCapabilities:
    """Optional capabilities available to the current Isaac Sim process.

    Debug input and scripted motion are implemented entirely by the Replicator
    Teleop extension. Live OpenXR input and MCAP replay require the separately
    distributed Isaac Teleop package. PINK IK requires its optional backend.
    """

    debug_input: bool
    scripted_motion: bool
    live_input: bool
    mcap_replay: bool
    pink_ik: bool
    native_input_unavailable_reason: str = ""
    pink_ik_unavailable_reason: str = ""


def get_teleop_capabilities() -> TeleopCapabilities:
    """Discover optional teleop backends without loading their native modules.

    The result is intentionally not cached because Kit extensions can be
    enabled after this module is imported.

    Returns:
        Capabilities currently available to the process.
    """
    native_input = _module_available("isaacteleop")
    pink_ik = _module_available("isaacsim.robot_motion.pink")
    platform_name = platform.system() or "this platform"
    native_reason = ""
    if not native_input:
        native_reason = f"Isaac Teleop native input is unavailable on {platform_name}; use Debug Mode instead"
    pink_reason = ""
    if not pink_ik:
        pink_reason = f"PINK IK is unavailable on {platform_name}; use a built-in IK solver instead"
    return TeleopCapabilities(
        debug_input=True,
        scripted_motion=True,
        live_input=native_input,
        mcap_replay=native_input,
        pink_ik=pink_ik,
        native_input_unavailable_reason=native_reason,
        pink_ik_unavailable_reason=pink_reason,
    )


__all__ = ["TeleopCapabilities", "get_teleop_capabilities"]
