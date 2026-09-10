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

"""Property panel for IsaacVirtualGantry prims: schema attributes + Enable/Disable."""

from __future__ import annotations

import typing

import isaacsim.robot_setup.virtual_gantry_schema as gantry_schema
import omni
import omni.ui as ui
from omni.kit.property.usd.usd_property_widget import UiDisplayGroup, UsdPropertiesWidget
from omni.kit.property.usd.usd_property_widget_builder import UsdPropertiesWidgetBuilder
from pxr import Tf, Usd


class VirtualGantryPropertiesWidget(UsdPropertiesWidget):
    """Specialized property widget shown when an ``IsaacVirtualGantry`` prim is selected.

    Renders the ``isaac:gantry:*`` attributes and the attach-body / articulation
    relationships, with Enable/Disable buttons that toggle ``isaac:gantry:enabled``
    so the runtime picks the change up on the next physics step.

    Args:
        *args: Positional arguments forwarded to the base widget.
        **kwargs: Keyword arguments forwarded to the base widget.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._old_payload = []
        self._listener = None
        self.frames = []
        self.wrapped_ui_elements = []
        self.reset()
        self._request_refresh()

        A = gantry_schema.Attributes
        R = gantry_schema.Relations
        self.property_metadata = {
            A.GANTRY_ROPE_LENGTH.name: {
                "displayName": "Rope Length",
                "tooltip": "Slack rope length (m); -1 auto-derives on first step",
            },
            A.GANTRY_STIFFNESS.name: {"displayName": "Stiffness", "tooltip": "Rope spring stiffness kp (N/m)"},
            A.GANTRY_DAMPING.name: {
                "displayName": "Damping",
                "tooltip": "Rope damping kd (N*s/m), applied only while extending",
            },
            A.GANTRY_BODY_OFFSET.name: {
                "displayName": "Body Offset",
                "tooltip": "Attach point offset in the body's local frame (m)",
            },
            A.GANTRY_ENABLED.name: {"displayName": "Enabled", "tooltip": "Whether the rope applies force"},
            A.GANTRY_EMA_ALPHA.name: {
                "displayName": "EMA Alpha",
                "tooltip": "Smoothing factor for the rope extension-rate estimate",
            },
            A.GANTRY_MIN_ROPE_LENGTH.name: {
                "displayName": "Min Rope Length",
                "tooltip": "Lower clamp for the rope length (m)",
            },
            A.GANTRY_VISUALIZE.name: {"displayName": "Visualize", "tooltip": "Draw the rope line and anchor marker"},
            A.GANTRY_STATUS.name: {
                "displayName": "Status",
                "tooltip": "Runtime rope state",
                "allowedTokens": ["Slack", "Taut", "Disabled"],
            },
            R.GANTRY_ATTACH_BODY.name: {
                "displayName": "Attach Body",
                "tooltip": "Articulation link the rope pulls on",
                "relationshipTargetPaths": True,
            },
            R.GANTRY_ARTICULATION.name: {
                "displayName": "Articulation",
                "tooltip": "Articulation-root prim of the robot",
                "relationshipTargetPaths": True,
            },
        }

    def _request_refresh(self) -> None:
        """Rebuild the property window so this widget appears/disappears with selection.

        ``get_window()`` returns None when the property window is unavailable —
        e.g. during extension teardown, before an already-queued USD notice is
        delivered. Bail out instead of dereferencing it: this also runs from
        inside a live ``Tf.Notice`` callback (via :meth:`_on_usd_changed`), and
        an exception raised there would escape past the listener re-arm and
        leave the widget permanently deaf to further USD changes.
        """
        property_window = omni.kit.window.property.get_window()
        if property_window is None:
            return
        selection = omni.usd.get_context().get_selection()
        selected_paths = selection.get_selected_prim_paths()
        window = property_window._window  # noqa: SLF001
        selection.clear_selected_prim_paths()
        window.frame.rebuild()
        selection.set_selected_prim_paths(selected_paths, True)
        window.frame.rebuild()

    def _on_usd_changed(self, notice: Usd.Notice.ObjectsChanged, stage: Usd.Stage) -> None:
        # Only react to selection/existence changes; per-attribute value updates are
        # handled by the listener adapters added in build_items. Calling the parent
        # handler here would double-process after on_new_payload already revoked the
        # listener (caused the property panel to freeze on the per-step status write).
        if self._old_payload != self.on_new_payload(self._payload):
            self._old_payload = self._prims
            self._request_refresh()
        elif self._listener is None and self._prims:
            # on_new_payload (just called) revoked our structural listener, but the
            # payload is unchanged so build_items won't re-run to re-register it.
            # Re-arm it here, otherwise the runtime's status writes (Slack/Taut)
            # would permanently kill it after the first transition.
            st = self._prims[-1].GetStage()
            if st is not None:
                self._listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._on_usd_changed, st)

    def _get_prim(self, prim_path: str) -> Usd.Prim | None:
        if prim_path:
            stage = self._payload.get_stage()
            if stage:
                prim = stage.GetPrimAtPath(prim_path)
                if prim and prim.GetTypeName() == gantry_schema.Classes.VIRTUAL_GANTRY.value:
                    return prim
        return None

    def on_new_payload(self, payload: list) -> list | bool:
        """Accept the payload when it is a single ``IsaacVirtualGantry`` prim.

        Args:
            payload: The property-window selection payload.

        Returns:
            The resulting list.
        """
        if self._listener:
            self._listener.Revoke()
            self._listener = None
        if not super().on_new_payload(payload):
            return False
        prims = [self._get_prim(p) for p in self._payload.get_paths()]
        self._prims = [p for p in prims if p is not None]
        if not self._prims:
            return False
        return self._prims

    def build_items(self) -> None:
        """Build the Enable/Disable buttons and the ordered attribute rows."""
        self.reset()
        if not self._payload or len(self._payload) == 0:
            return
        last_prim = self._get_prim(self._payload[-1])
        stage = last_prim.GetStage() if last_prim else None
        if not stage:
            return
        self._listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._on_usd_changed, stage)

        shared_props = self._get_shared_properties_from_selected_prims(last_prim)
        if not shared_props:
            return

        with ui.HStack(height=0):
            ui.Button("Enable", clicked_fn=lambda: self._set_enabled(True))
            ui.Button("Disable", clicked_fn=lambda: self._set_enabled(False))

        shared_props = self._customize_props_layout(shared_props)
        filtered_props = shared_props
        if self._filter.name:

            def display_name(ui_prop: typing.Any) -> str:
                return UsdPropertiesWidgetBuilder.get_display_name(ui_prop.prop_name, ui_prop.metadata)

            filtered_props = [p for p in shared_props if self._filter.matches(display_name(p))]
            if not filtered_props:
                return

        A = gantry_schema.Attributes
        R = gantry_schema.Relations
        property_order = [
            A.GANTRY_ENABLED.name,
            A.GANTRY_STATUS.name,
            A.GANTRY_ROPE_LENGTH.name,
            A.GANTRY_STIFFNESS.name,
            A.GANTRY_DAMPING.name,
            A.GANTRY_BODY_OFFSET.name,
            A.GANTRY_MIN_ROPE_LENGTH.name,
            A.GANTRY_EMA_ALPHA.name,
            A.GANTRY_VISUALIZE.name,
            R.GANTRY_ATTACH_BODY.name,
            R.GANTRY_ARTICULATION.name,
        ]

        def sort_key(prop: typing.Any) -> int:
            try:
                return property_order.index(prop.prop_name)
            except ValueError:
                return len(property_order)

        filtered_props.sort(key=sort_key)
        self._any_item_visible = True
        self.add_listener_adapters([p.prop_name for p in filtered_props])
        grouped_props = UiDisplayGroup("", self._maintain_property_order, filtered_props)
        self.build_nested_group_frames(stage, grouped_props)

    def _customize_props_layout(self, props: list) -> list:
        for prop in props:
            if prop.prop_name in self.property_metadata:
                for key, value in self.property_metadata[prop.prop_name].items():
                    prop.metadata[key] = value
                prop.override_display_name(self.property_metadata[prop.prop_name]["displayName"])
        return props

    def _set_enabled(self, value: bool) -> None:
        """Toggle ``isaac:gantry:enabled`` on every selected gantry prim.

        Args:
            value: The new value.
        """
        for prim in self._prims:
            attr = prim.GetAttribute(gantry_schema.Attributes.GANTRY_ENABLED.name)
            if attr and attr.IsValid():
                attr.Set(bool(value))
