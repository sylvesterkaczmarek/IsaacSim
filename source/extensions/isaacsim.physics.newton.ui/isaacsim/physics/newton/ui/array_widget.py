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

"""Pop-up editor for numeric array attributes (e.g. MuJoCo keyframe ``double[]`` vectors).

The default USD property panel renders scalar-array attributes as a read-only string. These
builders replace that with a compact summary row and an ``Edit`` button that opens a window
listing every entry by index in a vertical list, with the attribute display name and its
documentation shown at the top.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

import carb
import omni.kit.app
import omni.kit.commands
import omni.kit.undo
import omni.kit.window.property
import omni.ui as ui
from omni.kit.property.usd import widgets as usd_widgets
from omni.kit.property.usd.usd_attribute_model import UsdAttributeModel
from omni.kit.property.usd.usd_property_widget_builder import UsdPropertiesWidgetBuilder
from omni.kit.window.property import HORIZONTAL_SPACING
from pxr import Sdf, Vt

# Scalar-array value types this editor understands, mapped to their VtArray constructor.
_ARRAY_CTORS = {
    Sdf.ValueTypeNames.DoubleArray: Vt.DoubleArray,
    Sdf.ValueTypeNames.FloatArray: Vt.FloatArray,
    Sdf.ValueTypeNames.IntArray: Vt.IntArray,
    Sdf.ValueTypeNames.Int64Array: Vt.Int64Array,
    Sdf.ValueTypeNames.UIntArray: Vt.UIntArray,
    Sdf.ValueTypeNames.HalfArray: Vt.HalfArray,
}

_INT_TYPES = {
    Sdf.ValueTypeNames.IntArray,
    Sdf.ValueTypeNames.Int64Array,
    Sdf.ValueTypeNames.UIntArray,
}

# Keep opened editor windows alive and single-instance per attribute path.
_OPEN_EDITORS: dict[str, _ArrayEditorWindow] = {}

# Compact spacing between entry rows and within a row (0xAABBGGRR colors used below).
_ROW_SPACING = 2
_ROW_ITEM_SPACING = 4
# Every item in an entry row is given this height so they share a common baseline.
_ROW_HEIGHT = 20
# Minimum height for the scrolling entry list so it never collapses to zero.
_LIST_MIN_HEIGHT = 200
# Fixed height for the scrollable description panel so long text does not stretch the window.
_DOC_PANEL_HEIGHT = 72
# Description panel background, darker than the default window background (0xAABBGGRR).
_DOC_PANEL_BG = 0xFF1A1A1A


@dataclass(frozen=True)
class _ArrayConstraint:
    """MuJoCo size limit and per-index defaults for one numeric array.

    Attributes:
        size: Maximum number of entries the editor may hold.
        defaults: Per-index MuJoCo / schema defaults used when padding on commit.
        pad_on_commit: When True, OK expands a short array to ``size`` using
            ``defaults``. When False, ``size`` is only an upper bound (for
            attributes MuJoCo accepts at more than one legal length).
    """

    size: int
    defaults: tuple[int | float, ...]
    pad_on_commit: bool = True


# These lengths come from MuJoCo's fixed-size model fields (`mjN*` constants and
# fixed C arrays), not from USD. The schema uses arrays because USD has no
# fixed-size numeric-array type.
_ARRAY_CONSTRAINTS = {
    "mjc:biasPrm": _ArrayConstraint(10, (0.0,) * 10),
    "mjc:dynPrm": _ArrayConstraint(10, (1.0,) + (0.0,) * 9),
    "mjc:gainPrm": _ArrayConstraint(10, (1.0,) + (0.0,) * 9),
    "mjc:gear": _ArrayConstraint(6, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    "mjc:option:o_friction": _ArrayConstraint(5, (1.0, 1.0, 0.005, 0.0001, 0.0001)),
    "mjc:option:o_solimp": _ArrayConstraint(5, (0.9, 0.95, 0.001, 0.5, 2.0)),
    "mjc:option:o_solref": _ArrayConstraint(2, (0.02, 1.0)),
    "mjc:solimp": _ArrayConstraint(5, (0.9, 0.95, 0.001, 0.5, 2.0)),
    "mjc:solimpfriction": _ArrayConstraint(5, (0.9, 0.95, 0.001, 0.5, 2.0)),
    "mjc:solimplimit": _ArrayConstraint(5, (0.9, 0.95, 0.001, 0.5, 2.0)),
    "mjc:solref": _ArrayConstraint(2, (0.02, 1.0)),
    "mjc:solreffriction": _ArrayConstraint(2, (0.02, 1.0)),
    "mjc:solreflimit": _ArrayConstraint(2, (0.02, 1.0)),
    "mjc:springdamper": _ArrayConstraint(2, (0.0, 0.0)),
    # MuJoCo accepts one rest length or a two-value dead-band; do not pad a
    # single authored value to two entries (that would change semantics).
    "mjc:springlength": _ArrayConstraint(2, (-1.0, -1.0), pad_on_commit=False),
}


def _get_array_constraint(attr_path: Sdf.Path) -> _ArrayConstraint | None:
    """Return the MuJoCo fixed-size rule for an attribute path, if one exists.

    Args:
        attr_path: Full property path of the array attribute.

    Returns:
        The matching constraint, or ``None`` for a variable-length array.
    """
    return _ARRAY_CONSTRAINTS.get(attr_path.name)


def _pad_constrained_values(values: list, constraint: _ArrayConstraint | None) -> list:
    """Fill missing fixed-size entries with their MuJoCo per-index defaults.

    Args:
        values: Current array values to extend in place.
        constraint: Size rule for the attribute, or ``None`` when it is variable length.

    Returns:
        The same list, padded to the constrained size when padding applies.
    """
    if constraint is None or not constraint.pad_on_commit or len(values) >= constraint.size:
        return values
    values.extend(constraint.defaults[len(values) :])
    return values


def _array_ctor(type_name: Any) -> Any:
    """Return the VtArray constructor for a Sdf array value type.

    Args:
        type_name: Sdf value type name of the attribute.

    Returns:
        The matching ``Vt`` array constructor, or ``None`` for an unsupported type.
    """
    return _ARRAY_CTORS.get(type_name)


def _append_element(values: list, is_int: bool) -> list:
    """Append one default element to the value list, mutating and returning it.

    Args:
        values: Current array values to extend in place.
        is_int: Whether the array holds integers (default ``0``) or floats (default ``0.0``).

    Returns:
        The same list with one default element appended.
    """
    values.append(0 if is_int else 0.0)
    return values


def _remove_element(values: list, index: int) -> list:
    """Remove the element at the given index, mutating and returning the list.

    Args:
        values: Current array values to shrink in place.
        index: Index of the element to remove; out-of-range indices are ignored.

    Returns:
        The same list with the element removed when the index was valid.
    """
    if 0 <= index < len(values):
        values.pop(index)
    return values


def _read_values(stage: Any, attr_path: Sdf.Path) -> list:
    """Read the resolved array value of an attribute as a plain list.

    Unauthored attributes resolve to their schema fallback; attributes with neither an
    authored opinion nor a fallback read as an empty list.

    Args:
        stage: USD stage holding the attribute.
        attr_path: Full property path of the array attribute.

    Returns:
        The resolved array values as a plain list.
    """
    attr = stage.GetAttributeAtPath(attr_path)
    if not attr or not attr.IsValid():
        return []
    value = attr.Get()
    if value is None:
        return []
    return list(value)


def _summary_text(values: Sequence[Any]) -> str:
    """Build a compact one-line preview of the array contents.

    Args:
        values: Array values to summarize.

    Returns:
        A preview of up to eight entries followed by the total count.
    """
    count = len(values)
    if count == 0:
        return "empty (0)"
    preview = ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in values[:8])
    if count > 8:
        preview += ", ..."
    return f"[{preview}]  ({count})"


def close_all_editors() -> None:
    """Hide and release every open array editor window.

    Safe to call from extension shutdown. Window destruction is deferred so it does not
    run while ``omni.ui`` is still inside an event or draw.
    """
    editors = list(_OPEN_EDITORS.values())
    _OPEN_EDITORS.clear()
    for editor in editors:
        editor.request_close()


def _defer_once(owner: Any, slot_attr: str, callback: Callable[[], None]) -> None:
    """Schedule ``callback`` for the next app update, ignoring duplicate schedules.

    Args:
        owner: Object that owns the pending-task attribute.
        slot_attr: Attribute name on ``owner`` used to store the pending ``asyncio.Task``.
        callback: Zero-argument callable invoked on the next update when ``owner`` is still live.
    """
    if getattr(owner, slot_attr, None) is not None:
        return

    async def _run() -> None:
        await omni.kit.app.get_app().next_update_async()
        setattr(owner, slot_attr, None)
        # Drop the callback if the owner was torn down while we waited.
        if getattr(owner, "_destroyed", False):
            return
        callback()

    setattr(owner, slot_attr, asyncio.ensure_future(_run()))


# Owner of the deferred scroll restore. It lives at module scope because the editor window
# is torn down in the same update as the commit that needs the restore.
_SCROLL_OWNER = SimpleNamespace(_restore_task=None)


def _preserve_property_scroll() -> None:
    """Keep the Property panel scrolled where it was across a commit-triggered rebuild.

    Writing an attribute rebuilds the Property window, which replaces its
    ``ui.ScrollingFrame`` and drops the scroll offset. ``save_scroll_pos`` records the
    offset for the rebuild to restore; the extra deferred ``restore_scroll_pos`` covers
    rebuild paths that do not restore on their own. Both are no-ops on a Kit build that
    does not expose the API.
    """
    window = omni.kit.window.property.get_window()
    save = getattr(window, "save_scroll_pos", None)
    restore = getattr(window, "restore_scroll_pos", None)
    if save is None or restore is None:
        return
    save()
    # `restore_scroll_pos` schedules its own coroutine, so nothing mutates UI containers here.
    _defer_once(_SCROLL_OWNER, "_restore_task", restore)


class _ArrayEditorWindow:
    """A modeless window that edits a single numeric-array attribute by index.

    Args:
        stage: USD stage holding the attribute.
        attr_path: Full property path of the array attribute.
        type_name: Sdf value type name of the attribute.
        display_name: Human-readable attribute name shown in the title and header.
        documentation: Schema documentation shown in the description panel.
    """

    def __init__(
        self,
        stage: Any,
        attr_path: Sdf.Path,
        type_name: Any,
        display_name: str,
        documentation: str,
    ) -> None:
        self._stage = stage
        self._attr_path = attr_path
        self._type_name = type_name
        self._ctor = _array_ctor(type_name)
        self._is_int = type_name in _INT_TYPES
        self._is_unsigned = type_name == Sdf.ValueTypeNames.UIntArray
        self._display_name = display_name
        self._documentation = documentation or "No description available."
        self._constraint = _get_array_constraint(attr_path)
        self._models: list[ui.AbstractValueModel] = []
        # `self._values` is a working copy edited in memory; USD is written only on OK.
        self._values: list = _read_values(stage, attr_path)
        # Snapshot of the values the editor opened with, used to skip no-op writes.
        self._initial_values: list = list(self._values)
        self._list_frame: ui.Frame | None = None
        self._destroyed = False
        self._teardown_task = None

        self._window = ui.Window(
            f"Edit Array - {display_name}",
            width=420,
            height=480,
            flags=ui.WINDOW_FLAGS_NO_COLLAPSE,
        )
        self._window.set_visibility_changed_fn(self._on_visibility_changed)
        self._build_ui()

    def focus(self) -> None:
        """Bring the window to front and refresh the working copy from the latest authored values."""
        if self._destroyed or self._window is None:
            return
        self._values = _read_values(self._stage, self._attr_path)
        self._initial_values = list(self._values)
        self._window.visible = True
        self._rebuild_list()

    def request_close(self) -> None:
        """Hide the window; actual destruction is deferred via the visibility callback."""
        if self._destroyed or self._window is None:
            return
        # Already closing — avoid re-entering visibility handling.
        if not self._window.visible:
            self._schedule_teardown()
            return
        self._window.visible = False

    def destroy(self) -> None:
        """Release the window, its models, and any pending teardown task."""
        if self._destroyed:
            return
        self._destroyed = True
        task = self._teardown_task
        self._teardown_task = None
        if task is not None and not task.done():
            task.cancel()
        self._models = []
        self._list_frame = None
        if self._window is not None:
            self._window.set_visibility_changed_fn(None)
            self._window.destroy()
            self._window = None

    def _on_visibility_changed(self, visible: bool) -> None:
        if not visible:
            self._schedule_teardown()

    def _schedule_teardown(self) -> None:
        """Defer window teardown to the next app update.

        Destroying an ``omni.ui`` container synchronously from a visibility/event callback is
        unsupported, so the actual :meth:`destroy` runs after the next update instead.
        """
        if self._destroyed:
            return
        _OPEN_EDITORS.pop(str(self._attr_path), None)
        _defer_once(self, "_teardown_task", self.destroy)

    def _build_ui(self) -> None:
        with self._window.frame:
            with ui.VStack(spacing=1, style={"margin": 1}):
                ui.Label(self._display_name, style={"font_size": 16}, height=0)
                # Description lives in its own scrollable, darker panel so long text
                # does not stretch the window height.
                with ui.ScrollingFrame(
                    height=ui.Pixel(_DOC_PANEL_HEIGHT),
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    style={"ScrollingFrame": {"background_color": _DOC_PANEL_BG, "border_radius": 4}},
                ):
                    ui.Label(
                        self._documentation,
                        word_wrap=True,
                        height=0,
                        style={"color": 0xFFB0B0B0, "margin_height": 4, "margin_width": 6},
                    )
                ui.Separator(height=2)
                entry_label = "Entries (ordered by index):"
                if self._constraint is not None:
                    if self._constraint.pad_on_commit:
                        entry_label += f" {self._constraint.size} required"
                    else:
                        entry_label += f" max {self._constraint.size}"
                ui.Label(entry_label, height=0)
                # The entry list is the only element that absorbs free window height; the
                # spacer keeps a non-zero minimum so the list never collapses.
                with ui.ZStack(height=ui.Fraction(1)):
                    ui.Spacer(height=ui.Pixel(_LIST_MIN_HEIGHT))
                    with ui.ScrollingFrame(
                        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    ):
                        self._list_frame = ui.Frame(height=0, build_fn=self._build_list)
                ui.Separator(height=2)
                with ui.HStack(height=0, spacing=8):
                    ui.Spacer()
                    ui.Button("OK", width=80, identifier="array_ok", clicked_fn=self._on_ok)
                    ui.Button("Cancel", width=80, identifier="array_cancel", clicked_fn=self._on_cancel)

    def _icon_button_style(self, image_url: str) -> dict:
        """Build a transparent icon-button style using a kit-provided glyph.

        Args:
            image_url: Path to the SVG glyph rendered on the button.

        Returns:
            Style dictionary matching the Robot Schema add/remove buttons.
        """
        return {
            "Button": {"background_color": 0x0},
            "Button.Image": {"image_url": image_url, "color": 0xFFFFFFFF, "alignment": ui.Alignment.CENTER},
        }

    def _rebuild_list(self) -> None:
        """Request a rebuild of the entry rows.

        ``Frame.rebuild`` is designed to be invoked from callbacks (Robot Schema relationship
        lists use the same pattern), so this is safe during a button click while ``omni.ui`` is
        drawing.
        """
        if self._destroyed or self._list_frame is None:
            return
        self._list_frame.rebuild()

    def _build_list(self) -> None:
        if self._destroyed:
            return
        self._models = []
        add_icon = str(usd_widgets.ICON_PATH.joinpath("plus.svg"))
        remove_icon = str(usd_widgets.ICON_PATH.joinpath("remove.svg"))
        with ui.VStack(height=0, spacing=0):
            if not self._values:
                ui.Label("(empty)", style={"color": 0xFF808080}, height=0)
            for index, value in enumerate(self._values):
                # The field decides the row height. Fixed-height items are wrapped in a spacer
                # sandwich because a bare item anchors to the top of the row instead of centering.
                with ui.HStack(height=0, spacing=0):
                    with ui.VStack(width=20):
                        ui.Spacer()
                        ui.Label(f"[{index}]", width=20, height=_ROW_HEIGHT, alignment=ui.Alignment.LEFT_CENTER)
                        ui.Spacer()
                    if self._is_int:
                        field = ui.IntDrag(min=0) if self._is_unsigned else ui.IntDrag()
                        field.model.set_value(int(value))
                    else:
                        field = ui.FloatDrag()
                        field.model.set_value(float(value))
                    self._models.append(field.model)
                    with ui.VStack(width=0):
                        ui.Spacer()
                        ui.Button(
                            "",
                            width=20,
                            height=_ROW_HEIGHT,
                            identifier=f"array_remove_{index}",
                            tooltip="Remove entry",
                            clicked_fn=partial(self._remove_entry, index),
                            style=self._icon_button_style(remove_icon),
                        )
                        ui.Spacer()
            # A single add control at the end of the list appends one element, right-aligned
            # under the remove buttons.
            with ui.HStack(height=0, spacing=_ROW_ITEM_SPACING):
                ui.Spacer()
                ui.Button(
                    "",
                    width=20,
                    height=_ROW_HEIGHT,
                    enabled=self._constraint is None or len(self._values) < self._constraint.size,
                    identifier="array_add",
                    tooltip=(
                        f"Maximum {self._constraint.size} entries"
                        if self._constraint is not None and len(self._values) >= self._constraint.size
                        else "Add entry"
                    ),
                    clicked_fn=self._add_entry,
                    style=self._icon_button_style(add_icon),
                )

    def _collect_values(self) -> list:
        """Snapshot the current field values into the working copy.

        The fields are only read when they still match the working copy. After an add or remove
        the rebuild is deferred by one frame, and the stale fields would otherwise overwrite the
        pending change when the user clicks again before the rebuild runs.

        Returns:
            The working copy of the array values.
        """
        if len(self._models) != len(self._values):
            return self._values
        self._values = [
            model.get_value_as_int() if self._is_int else model.get_value_as_float() for model in self._models
        ]
        return self._values

    def _add_entry(self) -> None:
        if self._destroyed:
            return
        self._collect_values()
        if self._constraint is not None and len(self._values) >= self._constraint.size:
            return
        _append_element(self._values, self._is_int)
        self._rebuild_list()

    def _remove_entry(self, index: int) -> None:
        if self._destroyed:
            return
        self._collect_values()
        _remove_element(self._values, index)
        self._rebuild_list()

    def _commit(self) -> None:
        """Flush the working copy to USD as a single undoable change.

        Nothing is written when the working copy still matches the values the editor opened
        with, so confirming an untouched attribute leaves it unauthored.
        """
        if self._ctor is None:
            carb.log_warn(f"Unsupported array type for {self._attr_path}")
            return
        values = self._collect_values()
        _pad_constrained_values(values, self._constraint)
        if values == self._initial_values:
            return
        if self._is_unsigned and any(value < 0 for value in values):
            carb.log_warn(f"Cannot write negative values to unsigned array {self._attr_path}")
            return
        attr = self._stage.GetAttributeAtPath(self._attr_path)
        prev = attr.Get() if attr and attr.IsValid() and attr.HasAuthoredValueOpinion() else None
        value = self._ctor(values)
        # Record the scroll offset before the write; the property rebuild it triggers
        # otherwise snaps the panel back to the top.
        _preserve_property_scroll()
        omni.kit.commands.execute(
            "ChangePropertyCommand",
            prop_path=self._attr_path,
            value=value,
            prev=prev,
            # The attribute may only exist as a schema fallback, with no spec to write into yet.
            type_to_create_if_not_exist=self._type_name,
            usd_context_name=self._stage,
        )
        self._initial_values = list(values)

    def _on_ok(self) -> None:
        if self._destroyed:
            return
        self._commit()
        self.request_close()

    def _on_cancel(self) -> None:
        # Drop the working copy without writing to USD.
        self.request_close()


def _open_editor(
    stage: Any,
    attr_path: Sdf.Path,
    type_name: Any,
    display_name: str,
    documentation: str,
) -> None:
    """Open (or focus) the pop-up editor for the given array attribute.

    Args:
        stage: USD stage holding the attribute.
        attr_path: Full property path of the array attribute.
        type_name: Sdf value type name of the attribute.
        display_name: Human-readable attribute name shown in the window.
        documentation: Schema documentation shown in the description panel.
    """
    key = str(attr_path)
    existing = _OPEN_EDITORS.get(key)
    if existing is not None and not existing._destroyed:  # noqa: SLF001
        existing.focus()
        return
    _OPEN_EDITORS[key] = _ArrayEditorWindow(stage, attr_path, type_name, display_name, documentation)


class NumberArrayEditorBuilder(UsdPropertiesWidgetBuilder):
    """Property builder that renders numeric array attributes with a pop-up index editor.

    This entry point matches the ``omni.kit.property.physics`` builder signature, which
    passes a property descriptor object. Widgets built on ``UsdPropertiesWidget`` instead
    call :func:`build_number_array_editor` through ``UsdPropertyUiEntry.build_fn``.
    """

    def __new__(
        cls,
        stage: Any,
        prop: Any,
        prim_paths: Sequence[Sdf.Path],
        label_kwargs: dict[str, Any],
        widget_kwargs: dict[str, Any] | None,
        *args: Any,
    ) -> Any:
        """Build a summary row plus an ``Edit`` button opening the array editor window.

        Args:
            stage: USD stage containing the edited property.
            prop: Property descriptor exposing ``prop_name``, ``metadata`` and ``property_type``.
            prim_paths: Prim paths targeted by the widget.
            label_kwargs: Keyword arguments for the property label.
            widget_kwargs: Keyword arguments for the property body (unused here).
            *args: Ignored extra builder arguments.

        Returns:
            The :class:`UsdAttributeModel` backing the attribute.
        """
        return build_number_array_editor(
            stage,
            prop.prop_name,
            prop.metadata,
            prop.property_type,
            prim_paths,
            label_kwargs,
            widget_kwargs,
        )


def supports_array_editing(metadata: dict[str, Any]) -> bool:
    """Return whether an attribute's value type can be edited by the pop-up editor.

    Args:
        metadata: USD property metadata carrying the attribute's value type.

    Returns:
        True when the type is one of the supported scalar numeric arrays.
    """
    return _array_ctor(UsdPropertiesWidgetBuilder.get_type_name(metadata)) is not None


def build_number_array_editor(
    stage: Any,
    attr_name: str,
    metadata: dict[str, Any],
    property_type: Any,
    prim_paths: Sequence[Sdf.Path],
    label_kwargs: dict[str, Any] | None,
    widget_kwargs: dict[str, Any] | None,
) -> Any:
    """Build the array summary row and ``Edit`` button for one numeric array attribute.

    The signature matches ``UsdPropertiesWidgetBuilder.build`` so it can be used directly
    as a ``UsdPropertyUiEntry.build_fn``.

    Args:
        stage: USD stage containing the edited property.
        attr_name: Namespaced attribute name (for example ``mjc:solreflimit``).
        metadata: USD property metadata for the attribute.
        property_type: USD property type passed through to the default builder.
        prim_paths: Prim paths targeted by the widget.
        label_kwargs: Keyword arguments for the property label.
        widget_kwargs: Keyword arguments for the property body (unused here).

    Returns:
        The :class:`UsdAttributeModel` backing the attribute.
    """
    label_kwargs = label_kwargs or {}
    type_name = UsdPropertiesWidgetBuilder.get_type_name(metadata)

    # Fall back to the default rendering for anything we don't recognise as a numeric array.
    if _array_ctor(type_name) is None:
        return UsdPropertiesWidgetBuilder.build(
            stage,
            attr_name,
            metadata,
            property_type,
            prim_paths,
            label_kwargs,
            widget_kwargs,
        )

    display_name = UsdPropertiesWidgetBuilder.get_display_name(attr_name, metadata)
    documentation = metadata.get(Sdf.PropertySpec.DocumentationKey, "")

    # Editing arrays across multiple prims at once is ambiguous; target the first prim.
    target_path = prim_paths[0].AppendProperty(attr_name)

    with ui.HStack(spacing=HORIZONTAL_SPACING):
        model = UsdAttributeModel(
            stage,
            [path.AppendProperty(attr_name) for path in prim_paths],
            False,
            metadata,
        )
        UsdPropertiesWidgetBuilder.create_label(attr_name, metadata, label_kwargs)

        summary_field = ui.StringField(name="models_readonly", enabled=False, read_only=True)
        summary_field.identifier = f"array_summary_{attr_name}"

        def _refresh_summary(*_: Any) -> None:
            # Model value writes are safe from value-changed callbacks; no structural UI mutation.
            summary_field.model.set_value(_summary_text(_read_values(stage, target_path)))

        _refresh_summary()
        ui.Spacer(width=4)
        ui.Button(
            "Edit",
            width=48,
            identifier=f"array_edit_{attr_name}",
            tooltip=f"Edit {display_name} entries",
            clicked_fn=lambda: _open_editor(stage, target_path, type_name, display_name, documentation),
        )

    if not hasattr(model, "_newton_array_subs"):
        model._newton_array_subs = []  # noqa: SLF001
    model._newton_array_subs.append(model.subscribe_value_changed_fn(_refresh_summary))  # noqa: SLF001
    return model
