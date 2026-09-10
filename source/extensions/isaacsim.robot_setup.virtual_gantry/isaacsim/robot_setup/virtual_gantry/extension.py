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

"""Kit extension: the per-step rope runtime.

Headless-safe: this extension has no UI dependencies. The Create menu,
property panel and keyboard hotkeys live in the companion
``isaacsim.robot_setup.virtual_gantry.ui`` extension, which reaches the
manager published here via :func:`.._runtime.get_manager`.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import carb.eventdispatcher
import omni.ext
import omni.physics.core
import omni.timeline
import omni.usd

from ._runtime import VirtualGantryManager, _set_active_manager

_LOGGER = logging.getLogger(__name__)


class VirtualGantryExtension(omni.ext.IExt):
    """Drives one rope per ``IsaacVirtualGantry`` prim on the stage.

    The runtime subscribes to the physics pre-step on timeline Play (and
    clears it on Stop) and drives a
    :class:`~isaacsim.robot_setup.virtual_gantry._runtime.VirtualGantryManager`.
    Subscribing on Play (rather than once at startup) guarantees the callback
    binds to the live physics interface regardless of extension load order.
    """

    def on_startup(self, ext_id: str) -> None:
        """Start the manager and subscribe the physics-step runtime.

        Args:
            ext_id: Identifier of the extension being started.
        """
        self._ext_id = ext_id
        self._ext_name = omni.ext.get_extension_name(ext_id)
        self._manager = VirtualGantryManager()
        self._physics_iface = None
        self._physics_sub = None
        self._play_sub = None
        self._stop_sub = None
        self._stage_sub = None

        # Publish for the .ui extension's hotkeys.
        _set_active_manager(self._manager)

        # Runtime: (re)subscribe to the physics pre-step on Play, clear on Stop.
        self._physics_iface = omni.physics.core.get_physics_simulation_interface()
        dispatcher = carb.eventdispatcher.get_eventdispatcher()
        self._play_sub = dispatcher.observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_PLAY,
            on_event=lambda *_: self._subscribe_physics(),
            observer_name="isaacsim.robot_setup.virtual_gantry.on_play",
        )
        self._stop_sub = dispatcher.observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_STOP,
            on_event=lambda *_: self._on_stop(),
            observer_name="isaacsim.robot_setup.virtual_gantry.on_stop",
        )
        usd_ctx = omni.usd.get_context()
        self._stage_sub = dispatcher.observe_event(
            event_name=usd_ctx.stage_event_name(omni.usd.StageEventType.CLOSED),
            on_event=lambda *_: self._on_stage_closed(),
            observer_name="isaacsim.robot_setup.virtual_gantry.on_stage_closed",
        )
        # Cover the case where the timeline is already playing at load time.
        if omni.timeline.get_timeline_interface().is_playing():
            self._subscribe_physics()

    def on_shutdown(self) -> None:
        """Tear down the runtime subscriptions and drop the manager."""
        self._physics_sub = None
        self._play_sub = None
        self._stop_sub = None
        self._stage_sub = None

        _set_active_manager(None)
        if self._manager is not None:
            self._manager.reset(clear_overlay=False)
            self._manager = None
        gc.collect()

    # ------------------------------------------------------------------
    def _subscribe_physics(self) -> None:
        """Subscribe the manager's step to the physics pre-step (idempotent)."""
        if self._physics_sub is not None or self._physics_iface is None:
            return
        manager = self._manager

        def _pre_step(step_dt: float, _context: Any = None) -> None:
            try:
                manager.step(float(step_dt))
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(f"[virtual-gantry] physics step error: {exc!r}")

        self._physics_sub = self._physics_iface.subscribe_physics_on_step_events(
            on_update=_pre_step, pre_step=True, order=0
        )

    def _on_stop(self) -> None:
        """Timeline Stop: drop the physics subscription and reset (stage alive)."""
        self._physics_sub = None
        if self._manager is not None:
            self._manager.reset()

    def _on_stage_closed(self) -> None:
        """Stage close: reset without touching debug_draw (renderer tearing down)."""
        self._physics_sub = None
        if self._manager is not None:
            self._manager.reset(clear_overlay=False)
