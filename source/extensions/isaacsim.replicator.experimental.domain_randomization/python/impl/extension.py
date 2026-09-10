# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Define the extension lifecycle management for the Isaac Sim Replicator experimental domain randomization implementation."""

from typing import Any

import carb.eventdispatcher
import omni.ext
import omni.usd
from isaacsim.replicator.experimental.domain_randomization.scripts import physics_view


class Extension(omni.ext.IExt):
    """Object that tracks the lifetime of the Python part of the extension loading."""

    def on_startup(self) -> None:
        """Set up initial conditions for the Python part of the extension."""
        usd_context = omni.usd.get_context()
        self._stage_event_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=usd_context.stage_event_name(omni.usd.StageEventType.CLOSING),
            on_event=self._on_stage_closing,
            observer_name="isaacsim.replicator.experimental.domain_randomization._on_stage_closing",
        )

    def on_shutdown(self) -> None:
        """Shutting down this part of the extension prepares it for hot reload."""
        self._stage_event_sub = None
        physics_view.cleanup()

    def _on_stage_closing(self, _event: Any) -> None:
        """Clear physics-view registries when the stage is closing."""
        physics_view.cleanup()
