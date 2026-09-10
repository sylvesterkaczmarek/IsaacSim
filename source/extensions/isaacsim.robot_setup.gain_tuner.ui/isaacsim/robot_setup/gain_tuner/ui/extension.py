# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Kit extension entry: window + menu wiring for the Gain Tuner."""

import asyncio
import gc

import carb
import carb.eventdispatcher
import omni
import omni.kit.actions.core
import omni.kit.app
import omni.physics.core
import omni.timeline
import omni.ui as ui
import omni.usd
from isaacsim.gui.components.menu import MenuItemDescription
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner.ui.global_variables import EXTENSION_TITLE
from isaacsim.robot_setup.gain_tuner.ui.ui_builder import UIBuilder
from omni.kit.menu.utils import add_menu_items, remove_menu_items

_OBSERVER_PREFIX = "isaacsim.robot_setup.gain_tuner.ui.Extension"


class Extension(omni.ext.IExt):
    """Kit extension that provides the Gain Tuner authoring UI.

    On startup it creates a dockable ``omni.ui.Window`` (titled by ``EXTENSION_TITLE``)
    that hosts the gain tuning workflow, and registers a menu item under
    ``Tools > Robotics > Asset Editors`` (backed by an action in ``omni.kit.actions.core``) that
    toggles the window's visibility. While the window is visible it subscribes to
    stage, timeline, physics-step, and render-step events to keep the UI in sync
    with the simulation, and it tears those subscriptions down when the window is
    hidden or the extension shuts down.
    """

    def on_startup(self, ext_id: str) -> None:
        self.ext_id = ext_id
        self._ext_name = omni.ext.get_extension_name(ext_id)
        self._usd_context = omni.usd.get_context()

        self._window = ui.Window(
            EXTENSION_TITLE,
            width=700,
            height=500,
            visible=False,
            dockPreference=ui.DockPreference.LEFT_BOTTOM,
        )
        self._window.set_visibility_changed_fn(self._on_window)

        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.register_action(
            self._ext_name,
            f"CreateUIExtension:{EXTENSION_TITLE}",
            self._menu_callback,
            description=f"Add {EXTENSION_TITLE} to UI toolbar",
        )
        self._menu_items = [
            MenuItemDescription(
                name=EXTENSION_TITLE,
                onclick_action=(self._ext_name, f"CreateUIExtension:{EXTENSION_TITLE}"),
            )
        ]
        self._menu_items = [MenuItemDescription(name="Robotics", sub_menu=self._menu_items)]
        add_menu_items(self._menu_items, "Tools")

        self.ui_builder = UIBuilder()

        # Register the built-in and registry-backed tests so all modes
        # (snap-to-limits, sinusoidal, step, stress, dt sweep) are runnable from
        # the UI.  SINUSOIDAL/STEP use the tuner's native code path; SNAP_TO_LIMITS,
        # STRESS_TEST, and DISCRETIZATION are dispatched through the registered
        # RobotTest instances (the dt sweep runs the DISCRETIZATION probe once per
        # timestep, orchestrated by the UI layer).
        tuner = self.ui_builder._gains_tuner
        tuner.register_test(gain_tuner.GainsTestMode.SINUSOIDAL, gain_tuner.SinusoidalTest())
        tuner.register_test(gain_tuner.GainsTestMode.STEP, gain_tuner.StepFunctionTest())
        tuner.register_test(gain_tuner.GainsTestMode.SNAP_TO_LIMITS, gain_tuner.SnapToLimitsTest())
        tuner.register_test(gain_tuner.GainsTestMode.STRESS_TEST, gain_tuner.StressTest())
        tuner.register_test(gain_tuner.GainsTestMode.DISCRETIZATION, gain_tuner.DiscretizationSweepTest())

        self._physics_simulation_interface = omni.physics.core.get_physics_simulation_interface()
        self._event_dispatcher = carb.eventdispatcher.get_eventdispatcher()
        self._timeline = omni.timeline.get_timeline_interface()

        # Subscription handles owned by this extension.  They are created while the
        # window is visible and torn down on hide/shutdown.  Initialize them up front
        # so the show/hide branches can guard against double-subscribing.
        self._stage_event_sub_opened = None
        self._stage_event_sub_closed = None
        self._stage_event_sub_assets_loaded = None
        self._timeline_event_sub_play = None
        self._timeline_event_sub_stop = None
        self._physics_subscription = None
        self._render_subscription = None
        self._task = None

    def on_shutdown(self) -> None:
        remove_menu_items(self._menu_items, "Tools")
        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.deregister_action(self._ext_name, f"CreateUIExtension:{EXTENSION_TITLE}")
        self._teardown_subscriptions()
        self.ui_builder.cleanup()
        # `omni.ui` keeps the window registered under its title, so dropping
        # `self._window` does NOT release the `self._on_window` bound method the window
        # holds -- that method keeps this extension object alive across a hot reload.
        # Clear the callback before `destroy()`, which would otherwise re-enter
        # `_on_window`; the teardown that branch performs (`_teardown_subscriptions`
        # + `ui_builder.cleanup()`) has already run above.
        if self._window is not None:
            self._window.set_visibility_changed_fn(None)
            self._window.destroy()
            self._window = None
        gc.collect()

    def _subscribe_physics(self) -> None:
        """Subscribe to physics-step events if not already subscribed."""
        if self._physics_subscription is None:
            self._physics_subscription = self._physics_simulation_interface.subscribe_physics_on_step_events(
                pre_step=False, order=0, on_update=self._on_physics_step
            )

    def _subscribe_render(self) -> None:
        """Subscribe to render (app update) events if not already subscribed.

        The render-step callback drives viewport sizing and the deferred
        joint-repopulation retry, so it must stay alive for as long as the window
        is visible (across stage open/close), not just while the timeline plays.
        """
        if self._render_subscription is None:
            self._render_subscription = self._event_dispatcher.observe_event(
                event_name=omni.kit.app.GLOBAL_EVENT_UPDATE,
                on_event=self.ui_builder.on_render_step,
                observer_name=f"{_OBSERVER_PREFIX}._on_render_step",
            )

    def _teardown_subscriptions(self) -> None:
        """Unsubscribe every owned stage/timeline/physics/render handle.

        Dropping the carb ``ObserverGuard`` / physics subscription objects (setting
        them to ``None``) unsubscribes them via RAII.  The pending dock task is
        cancelled here as well so it can no longer touch widgets after teardown.
        """
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._usd_context = None
        self._stage_event_sub_opened = None
        self._stage_event_sub_closed = None
        self._stage_event_sub_assets_loaded = None
        self._timeline_event_sub_play = None
        self._timeline_event_sub_stop = None
        # Drop the physics/render step subscriptions BEFORE any `cleanup()` so that
        # `on_render_step` (via `sync_viewport_from_scroll_frame`) can no longer
        # touch widgets destroyed by `ui_builder.cleanup()` (render-step
        # use-after-free / native crash).
        self._physics_subscription = None
        self._render_subscription = None

    def _on_window(self, visible) -> None:
        # Use the framework-supplied `visible` flag rather than `self._window.visible`:
        # it is the authoritative value for this notification and keeps the callback
        # independent of `self._window`, which `on_shutdown` clears and destroys.
        if visible:
            self._usd_context = omni.usd.get_context()
            if self._stage_event_sub_opened is None:
                self._stage_event_sub_opened = self._event_dispatcher.observe_event(
                    event_name=self._usd_context.stage_event_name(omni.usd.StageEventType.OPENED),
                    on_event=self._on_stage_opened,
                    observer_name=f"{_OBSERVER_PREFIX}._on_stage_opened",
                )
            if self._stage_event_sub_closed is None:
                self._stage_event_sub_closed = self._event_dispatcher.observe_event(
                    event_name=self._usd_context.stage_event_name(omni.usd.StageEventType.CLOSED),
                    on_event=self._on_stage_closed,
                    observer_name=f"{_OBSERVER_PREFIX}._on_stage_closed",
                )
            if self._stage_event_sub_assets_loaded is None:
                self._stage_event_sub_assets_loaded = self._event_dispatcher.observe_event(
                    event_name=self._usd_context.stage_event_name(omni.usd.StageEventType.ASSETS_LOADED),
                    on_event=self._on_assets_loaded,
                    observer_name=f"{_OBSERVER_PREFIX}._on_assets_loaded",
                )
            if self._timeline_event_sub_play is None:
                self._timeline_event_sub_play = self._event_dispatcher.observe_event(
                    event_name=omni.timeline.GLOBAL_EVENT_PLAY,
                    on_event=self._on_timeline_play,
                    observer_name=f"{_OBSERVER_PREFIX}._on_timeline_play",
                )
            if self._timeline_event_sub_stop is None:
                self._timeline_event_sub_stop = self._event_dispatcher.observe_event(
                    event_name=omni.timeline.GLOBAL_EVENT_STOP,
                    on_event=self._on_timeline_stop,
                    observer_name=f"{_OBSERVER_PREFIX}._on_timeline_stop",
                )
            # Keep the render-step subscription alive while the window is visible so
            # deferred joint repopulation fires even after a stage open/close.
            self._subscribe_render()
            if self._timeline.is_playing():
                self._subscribe_physics()
            self._build_ui()
        else:
            self._teardown_subscriptions()
            self.ui_builder.cleanup()

    def _build_ui(self) -> None:
        self._content_scroll_frame = None
        # Layout: an outer VStack that FILLS the window (no height=0), containing
        #   1. a sticky header (auto-height) that stays locked at the top, and
        #   2. a ScrollingFrame with Fraction(1) height that fills the remaining
        #      space and scrolls its content.  The ScrollingFrame.computed_height
        #      is a STABLE, reliable viewport measurement (it does not grow with
        #      content), which UIBuilder uses to size the tables to fill the window.
        with self._window.frame:
            with ui.VStack(spacing=0):
                try:
                    # 1. Sticky header — locked at top
                    self.ui_builder.build_header_ui()

                    # 2. Scrollable content — fills the rest of the window
                    self._content_scroll_frame = ui.ScrollingFrame(
                        height=ui.Fraction(1),
                        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    )
                    with self._content_scroll_frame:
                        self.ui_builder.build_content_ui()
                except Exception as exc:
                    import traceback as _tb

                    carb.log_error(f"[GainTuner] build_ui failed: {exc}\n{_tb.format_exc()}")
                    ui.Label(f"UI build error: {exc}", word_wrap=True, style={"color": 0xFF4444FF})

        # Give UIBuilder a reference to the scrolling frame so it can read the
        # stable viewport height (polled each render step).
        self.ui_builder.set_scroll_frame_ref(self._content_scroll_frame)

        async def _dock() -> None:
            await omni.kit.app.get_app().next_update_async()
            tgt = ui.Workspace.get_window("Viewport")
            win = ui.Workspace.get_window(EXTENSION_TITLE)
            if win and tgt:
                win.dock_in(tgt, omni.ui.DockPosition.LEFT, 0.55)
            # A few frames so the docked layout fully settles, then apply the
            # viewport height once (the render-step poller keeps it in sync after).
            await omni.kit.app.get_app().next_update_async()
            await omni.kit.app.get_app().next_update_async()
            self.ui_builder.sync_viewport_from_scroll_frame(force=True)

        # Retain the dock task so it can be cancelled on hide/shutdown; cancel any
        # prior task first (window re-shown before the previous task settled).
        if self._task is not None:
            self._task.cancel()
        self._task = asyncio.ensure_future(_dock())

    def _menu_callback(self) -> None:
        # Toggling visibility triggers `_on_window`, which owns all subscription
        # setup/teardown; this callback only flips visibility and notifies the UI.
        self._window.visible = not self._window.visible
        self.ui_builder.on_menu_callback()

    def _on_timeline_play(self, event) -> None:
        self._subscribe_physics()
        self._subscribe_render()
        self.ui_builder.on_timeline_event(event)

    def _on_timeline_stop(self, event) -> None:
        self._physics_subscription = None
        self.ui_builder.on_timeline_event(event)

    def _on_physics_step(self, step, context) -> None:
        self.ui_builder.on_physics_step(step)

    def _on_stage_opened(self, event) -> None:
        # Drop only the physics subscription (a fresh stage is not playing).  Keep
        # the render-step subscription intact so deferred joint repopulation still
        # fires for the newly opened stage.
        self._physics_subscription = None
        self.ui_builder.reset()
        self.ui_builder.on_stage_event(event)

    def _on_assets_loaded(self, event) -> None:
        self.ui_builder.on_stage_event(event)

    def _on_stage_closed(self, event) -> None:
        # Keep the render-step subscription; only the physics subscription is
        # invalidated by closing the stage.
        self._physics_subscription = None
        self.ui_builder.reset()
        self.ui_builder.on_stage_event(event)
