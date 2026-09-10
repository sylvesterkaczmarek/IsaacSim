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

"""Utility classes and functions for Newton physics UI property builders."""

import re
from collections.abc import Callable, Sequence
from typing import Any

import carb
import omni.kit.undo
import omni.kit.window.property
import omni.ui as ui
from omni.kit.commands import execute
from omni.kit.property.physics import database
from omni.kit.property.usd import PrimPathWidget
from omni.kit.property.usd.prim_selection_payload import PrimSelectionPayload
from omni.kit.property.usd.usd_attribute_model import UsdAttributeModel
from omni.kit.property.usd.usd_property_widget import UsdPropertiesWidget
from omni.kit.property.usd.usd_property_widget_builder import UsdPropertiesWidgetBuilder
from pxr import Sdf, Usd

from .array_widget import build_number_array_editor, supports_array_editing

try:
    from newton._src.usd.schema_resolver import PrimType
except ImportError:
    from enum import IntEnum

    class PrimType(IntEnum):  # Stub matching newton.PrimType values
        """Stub enum matching newton.PrimType values for use when Newton is not available."""

        SCENE = 0
        JOINT = 1
        SHAPE = 2
        BODY = 3
        MATERIAL = 4
        ACTUATOR = 5
        ARTICULATION = 6


_RESOLVERS = None
_RESOLVER_INDEX = None
_RESOLVER_NAMES = None


def make_hide_cb(
    own_resolver_name: str, prim_type: "PrimType", key: str | list, default: object = None
) -> Callable[[Any, Sequence[Any]], tuple[bool, str, str]]:
    """Create a disable callback for a property covered by both Newton and Mjc resolvers.

    Resolvers are ordered by priority (Newton first, Mjc second). The *preferred* resolver is
    the first one that has authored a value for any of the keys. If nothing is authored by
    anyone, the first resolver (Newton) is preferred by default.

    The callback returns a 3-tuple ``(disabled, resolver_display_name, usd_attr_name)``:

    - ``disabled``: ``False`` if own resolver is preferred, ``True`` otherwise.
    - ``resolver_display_name``: human-readable name of the preferred resolver (e.g.
      ``"MuJoCo"``), or ``""`` if own is preferred.
    - ``usd_attr_name``: USD attribute name authored by the preferred resolver (e.g.
      ``"newton:gravityEnabled"``), or the name from the preferred resolver's mapping if
      nothing is authored. Empty string if own is preferred.

    Args:
        own_resolver_name: ``"newton"`` or ``"mjc"`` — identifies which resolver owns
            this property (matched against ``SchemaResolver.name``).
        prim_type: :class:`PrimType` enum value for the prim category (SCENE, JOINT, …).
        key: Resolver key string (e.g. ``"armature"``), or a list of key strings when
            multiple keys map to the same USD attribute (e.g. ``"mjc:solref"``).
            The property is considered authored if *any* of the keys has a value.
        default: Unused; kept for call-site documentation of the simulation default.

    Returns:
        A callback with signature ``(stage, prim_paths) -> tuple[bool, str, str]``.
    """
    keys = key if isinstance(key, list) else [key]

    def _resolver_authored_attr(r: object, prim: object) -> str | None:
        """Returns the USD attribute name of the first authored key, or None.

        Args:
            r: The schema resolver to query for authored values.
            prim: The USD prim to check for authored attributes.

        Returns:
            The USD attribute name of the first authored key, or None.
        """
        for k in keys:
            if r.get_value(prim, prim_type, k) is not None:
                spec = r.mapping.get(prim_type, {}).get(k)
                return spec.name if spec else k
        return None

    def _cb(stage: Any, prim_paths: Sequence[Any]) -> tuple[bool, str, str]:
        global _RESOLVERS, _RESOLVER_INDEX, _RESOLVER_NAMES

        if not prim_paths or stage is None:
            return False, "", ""

        if _RESOLVERS is None:
            try:
                from newton._src.usd.schemas import SchemaResolverMjc, SchemaResolverNewton

                _RESOLVERS = [SchemaResolverNewton(), SchemaResolverMjc()]
                _RESOLVER_NAMES = ["Newton", "MuJoCo"]
                _RESOLVER_INDEX = {r.name: idx for idx, r in enumerate(_RESOLVERS)}
            except ImportError:
                _RESOLVERS = None
        if _RESOLVERS is None:
            return False, "", ""

        own_idx = _RESOLVER_INDEX.get(own_resolver_name, -1)
        if own_idx == -1:
            carb.log_warn(f"Own resolver {own_resolver_name} not found for {prim_type} {key}")
            return False, "", ""

        prim = stage.GetPrimAtPath(str(prim_paths[0]))

        # Find preferred resolver: first with an authored value, or first overall if nothing is authored.
        # authored_attr is the corresponding USD attribute name; falls back to the mapping name when nothing is authored.
        authored_attrs = [_resolver_authored_attr(r, prim) for r in _RESOLVERS]
        authored_idx = next((i for i, a in enumerate(authored_attrs) if a is not None), None)
        preferred_idx = authored_idx if authored_idx is not None else 0
        if authored_idx is not None:
            authored_attr = authored_attrs[preferred_idx]
        else:
            spec = _RESOLVERS[0].mapping.get(prim_type, {}).get(keys[0])
            authored_attr = spec.name if spec else keys[0]

        if preferred_idx == own_idx:
            return False, "", ""
        return True, _RESOLVER_NAMES[preferred_idx], authored_attr or ""

    return _cb


class DisableByCallbackBuilder(UsdPropertiesWidgetBuilder):
    """Widget builder that disable a property based on a callback result."""

    def __new__(
        cls,
        stage: Any,
        prop: Any,
        prim_paths: Sequence[Any],
        label_kwargs: dict[str, Any],
        widget_kwargs: dict[str, Any],
        disable_callback: Callable[[Any, Sequence[Any]], tuple[bool, str, str]],
        inner_builder: type | None = None,
    ) -> Any:
        """Create a new widget with an overlay that disables based on a callback.

        Args:
            stage: USD stage containing the edited property.
            prop: Property descriptor for the widget.
            prim_paths: Prim paths targeted by the widget.
            label_kwargs: Keyword arguments for the widget label.
            widget_kwargs: Keyword arguments for the widget body.
            disable_callback: Callback returning disabled state and tooltip metadata.
            inner_builder: Optional property builder class used instead of the default
                USD attribute widget. Used to compose disable overlays with custom editors
                (for example the numeric array pop-up).

        Returns:
            The created USD property model.
        """

        def _tooltip(resolver_name: str, attr_name: str) -> str:
            if not resolver_name:
                return ""
            return f"Controlled by {attr_name}" if attr_name else f"Controlled by {resolver_name}"

        disabled, resolver_name, attr_name = disable_callback(stage, prim_paths)
        with ui.ZStack():
            if inner_builder is not None:
                model = inner_builder(stage, prop, prim_paths, label_kwargs, widget_kwargs)
            else:
                model = cls.build(
                    stage, prop.prop_name, prop.metadata, prop.property_type, prim_paths, label_kwargs, widget_kwargs
                )
            overlay = ui.Rectangle(
                width=ui.Fraction(1),
                height=ui.Fraction(1),
                style={"background_color": ui.color(0, 0, 0, 64), "border_radius": 4},
                visible=disabled,
                tooltip=_tooltip(resolver_name, attr_name),
            )

        def _refresh(*_: Any) -> None:
            disabled, resolver_name, attr_name = disable_callback(stage, prim_paths)
            overlay.visible = disabled
            overlay.set_tooltip(_tooltip(resolver_name, attr_name))

        model._remove_if_default = True
        if not hasattr(model, "_newton_disable_subs"):
            model._newton_disable_subs = []
        model._newton_disable_subs.append(model.subscribe_value_changed_fn(_refresh))
        return model


class HideByCallbackBuilder(UsdPropertiesWidgetBuilder):
    """Widget builder that hide a property based on a callback result."""

    def __new__(
        cls,
        stage: Any,
        prop: Any,
        prim_paths: Sequence[Any],
        label_kwargs: dict[str, Any],
        widget_kwargs: dict[str, Any],
        hide_callback: Callable[[Any, Sequence[Any]], bool],
    ) -> Any:
        """Create a new widget that is hidden when the callback returns True.

        Args:
            stage: USD stage containing the edited property.
            prop: Property descriptor for the widget.
            prim_paths: Prim paths targeted by the widget.
            label_kwargs: Keyword arguments for the widget label.
            widget_kwargs: Keyword arguments for the widget body.
            hide_callback: Callback returning whether the widget should be hidden.

        Returns:
            The visible widget model or a hidden placeholder attribute model.
        """
        hidden = hide_callback(stage, prim_paths)
        if not hidden:
            model = cls.build(
                stage, prop.prop_name, prop.metadata, prop.property_type, prim_paths, label_kwargs, widget_kwargs
            )
            return model
        else:
            return UsdAttributeModel(
                stage, [path.AppendProperty(prop.prop_name) for path in prim_paths], False, prop.metadata
            )


def apply_codeless_api_with_dependencies(prim: Usd.Prim, dependencies: Sequence[str], api: str) -> None:
    """Apply a codeless API schema and its required API dependencies.

    Args:
        prim: Prim to modify.
        dependencies: API schema names to apply before the requested API.
        api: API schema name to apply.
    """
    for dependency in dependencies:
        if not prim.HasAPI(dependency):
            execute("ApplyCodelessAPISchemaCommand", api=dependency, prim=prim)
    if not prim.HasAPI(api):
        execute("ApplyCodelessAPISchemaCommand", api=api, prim=prim)


def request_property_window_refresh() -> None:
    """Rebuild the property window without touching USD selection."""
    property_window = omni.kit.window.property.get_window()
    if property_window and property_window._window:  # noqa: SLF001
        property_window._window.frame.rebuild()  # noqa: SLF001


def build_remove_schema_frame_header(
    collapsed: bool, text: str, schema: str, on_remove: Callable[[], None] | None
) -> None:
    """Build a CollapsableFrame header carrying the shared remove-schema button.

    Every Newton and MuJoCo schema frame uses this header so the remove button keeps the
    same icon, placement, and one-click behavior across all of them.

    Args:
        collapsed: Whether the frame is currently collapsed.
        text: The header label text.
        schema: Applied API schema the button unapplies.
        on_remove: Callback invoked when the button is clicked. Pass ``None`` to omit the
            button, for example while the schema is not applied to the selected prim.
    """
    if collapsed:
        alignment = ui.Alignment.RIGHT_CENTER
        width = 5
        height = 7
    else:
        alignment = ui.Alignment.CENTER_BOTTOM
        width = 7
        height = 5

    with ui.HStack(spacing=8):
        with ui.VStack(width=0):
            ui.Spacer()
            ui.Triangle(
                style_type_name_override="CollapsableFrame.Header", width=width, height=height, alignment=alignment
            )
            ui.Spacer()
        ui.Label(text, style_type_name_override="CollapsableFrame.Header", width=ui.Fraction(1))

        if on_remove is None:
            return

        button_style = {
            "Button.Image": {
                "color": 0xFFFFFFFF,
                "alignment": ui.Alignment.CENTER,
                "image_url": "${icons}/Cancel_64.png",
            }
        }

        ui.Spacer(width=ui.Fraction(6))
        with ui.ZStack(content_clipping=True, width=16, height=16):
            ui.Button(
                "",
                style=button_style,
                clicked_fn=on_remove,
                identifier=f"remove_{schema}",
                tooltip=f"Remove {schema}",
            )


def _usd_peroperty_name_to_display_name(name: str) -> str:
    # usd name are like "newton:selfCollisionEnabled"
    # we want to take the last part after :
    # then Capitalize the first entry and split where the capital letters are
    display_name = name.split(":")[-1]
    return re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", display_name).title()


# This is a slightly modified class based on _RobotSchemaWidgetBase, maybe we should reconcile the two
class NewtonWidgetBase(UsdPropertiesWidget):
    """Widget builder that adds menu items for adding schemas.

    Args:
        prefix: Menu path the Apply entry is added under, for example ``Physics/Mujoco``.
        title: Title of the property section.
        menu_label: Label of the Apply menu entry.
        schema: API schema name the widget displays and applies.
        collapsed: Whether the property section starts collapsed.
        attributes: Attribute specs of the schema to display; all others are hidden.
        ignored_attributes: Attribute names to drop from ``attributes``.
        apply_fn: Callable applying the schema to a prim; defaults to applying ``schema`` alone.
        show_fn: Callable deciding whether the Apply entry is offered for a prim.
        relationships: Relationship descriptors to display alongside the attributes.
        exclusive_classes: API schemas that suppress this widget when already applied.
    """

    def __init__(
        self,
        prefix: str,
        title: str,
        menu_label: str,
        schema: str,
        collapsed: bool = False,
        attributes: list | None = None,
        ignored_attributes: list | None = None,
        apply_fn: object = None,
        show_fn: object = None,
        relationships: object = None,
        exclusive_classes: object = None,
    ) -> None:
        super().__init__(title, collapsed)
        omni.kit.undo.subscribe_on_change(self._undo_redo_on_change)

        self._schema = schema
        ignored_attributes = ignored_attributes or []
        self._attributes = [attr for attr in (attributes or []) if attr.name not in ignored_attributes]
        self._relationships = relationships or []
        self._apply_fn = apply_fn
        self._show_fn = show_fn
        self._menu_label = menu_label
        self._attr_map = {attr.name: attr for attr in self._attributes}
        self._relationship_map = {rel.name: rel for rel in self._relationships}
        self._prim = None
        self._old_payload = None
        self._exclusive_classes = exclusive_classes
        self._menu_prefix = prefix
        self._menu_entries = [
            PrimPathWidget.add_button_menu_entry(
                f"{self._menu_prefix}/{menu_label}", show_fn=self._button_show, onclick_fn=self._button_onclick
            )
        ]

    def clean(self) -> None:
        """Remove menu entries and release subscriptions."""
        super().clean()
        omni.kit.undo.unsubscribe_on_change(self._undo_redo_on_change)
        for menu in self._menu_entries:
            PrimPathWidget.remove_button_menu_entry(menu)
        self._menu_entries = []

    def destroy(self) -> None:
        """Clean up resources and remove menu entries."""
        self.clean()

    def _button_show(self, objects: dict) -> bool:
        """Determines if the button should be shown based on prim selection.

        Args:
            objects: Dictionary containing stage and prim_list for evaluation.

        Returns:
            True if button should be shown, False otherwise.
        """
        stage = objects.get("stage")
        prim_list = objects.get("prim_list")
        if not stage or not prim_list:
            return False
        for item in prim_list:
            prim = stage.GetPrimAtPath(item) if isinstance(item, Sdf.Path) else item
            if prim and not self._has_exclusive_schema(prim) and (not self._show_fn or self._show_fn(prim)):
                return True
        return False

    def _button_onclick(self, payload: PrimSelectionPayload) -> None:
        """Handles button click to apply schema to selected prims.

        Args:
            payload: The prim selection payload containing paths to process.
        """
        omni.kit.undo.begin_group()
        stage = self._payload.get_stage() if self._payload else omni.usd.get_context().get_stage()
        if not stage:
            return

        for path in payload:
            if not path:
                continue
            prim = stage.GetPrimAtPath(path)
            if not prim or self._has_exclusive_schema(prim):
                continue
            if self._apply_fn:
                self._apply_fn(prim)
            else:
                # Call execute so the command is recorded in the undo stack
                execute("ApplyCodelessAPISchemaCommand", api=self._schema, prim=prim)

            instanceable = [p for p in Usd.PrimRange(prim) if p.IsInstanceable()]
            if instanceable:
                prim.SetInstanceable(True)
                prim.ClearMetadata("instanceable")
        self._request_refresh()
        omni.kit.undo.end_group()

    def _request_refresh(self) -> None:
        """Refresh the property window without touching USD selection."""
        request_property_window_refresh()

    def _on_usd_changed(self, notice: object, stage: object) -> None:
        """Handles USD change notifications.

        A full property-window rebuild is only needed when the prim gains the schema this
        widget displays, since that changes which widgets the window shows. Rebuilding on
        every notice would also discard the panel's scroll position on each value edit; the
        base widget updates its models in place instead.

        Args:
            notice: The USD change notice.
            stage: The USD stage that changed.
        """
        had_schema = self._prim is not None
        # `on_new_payload` refreshes `self._prim` from the current stage state.
        if self.on_new_payload(self._payload) and not had_schema:
            self._request_refresh()
            return
        super()._on_usd_changed(notice, stage)

    def _get_prim(self, prim_path: object) -> Usd.Prim | None:
        """Retrieves a prim with the required schema from the given path.

        Args:
            prim_path: The path to the prim.

        Returns:
            The prim if it exists and has the required schema, None otherwise.
        """
        if prim_path:
            stage = self._payload.get_stage()
            if stage:
                prim = stage.GetPrimAtPath(prim_path)
                if prim and prim.HasAPI(self._schema):
                    return prim
        return None

    def on_new_payload(self, payload: list) -> bool:
        """See ``PropertyWidget.on_new_payload``.

        Args:
            payload: The new prim selection payload.

        Returns:
            True if the prim if found, or ``False`` if the widget should not be shown.
        """
        if not super().on_new_payload(payload):
            return False

        if len(self._payload) != 1:
            return False
        prim_path = self._payload.get_paths()[0]
        self._prim = self._get_prim(prim_path)
        self._old_payload = self._prim
        if not self._prim:
            return False

        return True

    def on_remove_schema(self) -> None:
        """Removes the schema from the prim."""
        stage = self._payload.get_stage()
        if not stage or not self._payload:
            return

        prim = self._get_prim(self._payload.get_paths()[0])
        if not prim:
            return

        if prim.HasAPI(self._schema):
            omni.kit.undo.begin_group()
            execute("UnapplyCodelessAPISchemaCommand", api=self._schema, prim=prim)
            omni.kit.undo.end_group()
        self._request_refresh()

    def _filter_props_to_build(self, props: list) -> list:
        """Filters properties to build based on the schema's attributes and relationships.

        Args:
            props: List of properties to filter.

        Returns:
            Filtered list of properties with display names set.
        """
        filtered = []
        for prop in props:
            if isinstance(prop, Usd.Attribute) and prop.GetName() in self._attr_map:
                attr = self._attr_map[prop.GetName()]
                prop.SetDisplayName(attr.displayName or _usd_peroperty_name_to_display_name(prop.GetName()))
                filtered.append(prop)
            elif isinstance(prop, Usd.Relationship) and prop.GetName() in self._relationship_map:
                relationship = self._relationship_map[prop.GetName()]
                prop.SetDisplayName(relationship.display_name)
                filtered.append(prop)
        return filtered

    def _customize_props_layout(self, props: list) -> list:
        """Apply physics property builders and ordering to the custom schema frame.

        This widget derives from ``UsdPropertiesWidget``, which builds properties itself and
        never consults the ``omni.kit.property.physics`` builder database. Ignored schemas
        still register their builders and ordering in that database, so attach them through
        each property's ``build_fn`` and sort the resulting entries here.

        Args:
            props: Property UI entries about to be built.

        Returns:
            The ordered property UI entries with their registered builders attached.
        """
        props = super()._customize_props_layout(props)
        for prop in props:
            available_builders = database.get_available_builders(prop.prop_name)
            if prop.build_fn is None and available_builders:
                builder_data = available_builders[0]

                def _build_with_physics_builder(
                    stage: Any,
                    _attr_name: str,
                    _metadata: dict,
                    _property_type: Any,
                    prim_paths: Sequence[Any],
                    label_kwargs: dict | None,
                    widget_kwargs: dict | None,
                    *,
                    property_entry: Any = prop,
                    physics_builder_data: list = builder_data,
                ) -> Any:
                    return physics_builder_data[0](
                        stage,
                        property_entry,
                        prim_paths,
                        label_kwargs or {},
                        widget_kwargs,
                        *physics_builder_data[1:],
                    )

                prop.build_fn = _build_with_physics_builder
            elif prop.build_fn is None and supports_array_editing(prop.metadata):
                prop.build_fn = build_number_array_editor

        order = database.get_property_order(self._schema)
        indices = {name: index for index, name in enumerate(order)}
        return sorted(props, key=lambda prop: indices.get(getattr(prop, "base_name", prop.prop_name), len(indices)))

    def _has_exclusive_schema(self, prim: object) -> bool:
        """Checks if the prim has any exclusive schema applied.

        Args:
            prim: The prim to check.

        Returns:
            True if prim has exclusive schema, False otherwise.
        """
        if self._exclusive_classes:
            return any(prim.HasAPI(schema) for schema in self._exclusive_classes)
        else:
            return False

    def build_items(self) -> None:
        """Builds property widget items for the schema.

        Constructs the property items only when the collapsible frame is expanded and a valid prim is available.
        """
        if self._collapsable_frame and not self._collapsable_frame.collapsed and self._prim:
            super().build_items()

    def _build_frame_header(self, collapsed: bool, text: str, id: str | None = None) -> None:
        """Build a custom header for the CollapsableFrame with a remove button.

        Args:
            collapsed: Whether the frame is currently collapsed.
            text: The header label text.
            id: Optional identifier for the header.
        """
        build_remove_schema_frame_header(collapsed, text, self._schema, self.on_remove_schema)

    def _undo_redo_on_change(self, cmds: list) -> None: ...


from omni.kit.property.physics.builders import PrettyPrintTokenComboBuilder


class ValueChangeByCallbackTokenBuilder(PrettyPrintTokenComboBuilder):
    """Widget builder that triggers a callback when the property value changes."""

    def __init__(self, stage, prop, prim_paths, label_kwargs, widget_kwargs, pretty_names, additions, call_back):
        super().__init__(stage, prop, prim_paths, label_kwargs, widget_kwargs, pretty_names, additions)
        if not hasattr(self, "_newton_value_change_call_back"):
            self._newton_value_change_call_back = []
        self._newton_value_change_call_back.append(self.subscribe_item_changed_fn(call_back))
