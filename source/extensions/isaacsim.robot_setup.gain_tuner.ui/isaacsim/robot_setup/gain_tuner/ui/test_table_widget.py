# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Provides UI widgets for configuring joint test parameters in the gains tuner interface."""

from enum import Enum, IntEnum
from functools import partial
from math import inf

import carb
import numpy as np
import omni.ui as ui
import pxr
from isaacsim.robot_setup import gain_tuner
from omni.physics.tensors import DofType

from .base_table_widget import ITEM_HEIGHT, TableItem, TableItemDelegate, TableModel, TableWidget
from .cell_widget import CellLabelField
from .gains_tuner_backend import GainsTestMode
from .style import get_style


class ColumnIndex(IntEnum):
    """Enumeration defining column indices for the test joint table widget.

    This enum maps column positions to their corresponding data types in the joint testing interface,
    including joint identification, test parameters, and motion characteristics for gain tuning operations.
    """

    JOINT = 0
    """Column index for the joint name display."""
    TEST = 1
    """Column index for the test enable/disable checkbox."""
    SEQUENCE = 2
    """Column index for the test sequence number."""
    AMPLITUDE = 3
    """Column index for the sinusoidal test amplitude value."""
    OFFSET = 4
    """Column index for the position offset value."""
    PERIOD = 5
    """Column index for the test period duration."""
    PHASE = 6
    """Column index for the test phase shift value."""
    STEP_MAX = 7
    """Column index for the maximum step value in step tests."""
    STEP_MIN = 8
    """Column index for the minimum step value in step tests."""
    USER_PROVIDED = 9  # User provided gains is still not implemented.
    """Column index for user-provided test data source."""


class TestMode(IntEnum):
    """Enumeration defining the different test modes available for joint testing.

    This enumeration specifies the types of test signals that can be applied to joints during gain tuning.
    Each mode represents a different mathematical function used to generate reference trajectories for testing
    joint control performance.
    """

    SINUSOIDAL = 0
    """Sinusoidal test mode for joint movement patterns."""
    STEP = 1
    """Step test mode for joint movement patterns."""
    USER_PROVIDED = 2
    """User-provided test mode for custom joint movement patterns."""
    SNAP_TO_LIMITS = 3
    """Snap-to-limits test mode that drives joints to their lower and upper limits."""
    STRESS_TEST = 4
    """Stress test mode that bombards joints with extreme random commands."""
    DISCRETIZATION = 5
    """dt physics sweep test mode that characterizes accuracy degradation across physics timesteps."""


def default_test_column_id_map() -> dict:
    """Return the per-mode Test-table column layout keyed by :class:`TestMode`.

    Each test mode maps to the ordered list of :class:`ColumnIndex` columns the
    Test Gains joint table renders for that mode.  Every selectable test mode must
    have an entry here: the table view indexes this map by the active mode when it
    builds its tree, so a missing mode raises ``KeyError`` inside the frame build
    and the joint table renders empty.  SNAP_TO_LIMITS, STRESS_TEST, and
    DISCRETIZATION share the minimal Joint / Test / Sequence layout (they run all
    tested joints together, partitioned only by sequence).

    Returns:
        A fresh dict mapping each :class:`TestMode` to its ordered column list.
    """
    return {
        TestMode.SINUSOIDAL: [
            ColumnIndex.JOINT,
            ColumnIndex.TEST,
            ColumnIndex.SEQUENCE,
            ColumnIndex.AMPLITUDE,
            ColumnIndex.OFFSET,
            ColumnIndex.PERIOD,
            ColumnIndex.PHASE,
        ],
        TestMode.STEP: [
            ColumnIndex.JOINT,
            ColumnIndex.TEST,
            ColumnIndex.SEQUENCE,
            ColumnIndex.STEP_MIN,
            ColumnIndex.STEP_MAX,
            ColumnIndex.PERIOD,
            ColumnIndex.PHASE,
        ],
        TestMode.USER_PROVIDED: [ColumnIndex.JOINT, ColumnIndex.TEST, ColumnIndex.USER_PROVIDED],
        TestMode.SNAP_TO_LIMITS: [ColumnIndex.JOINT, ColumnIndex.TEST, ColumnIndex.SEQUENCE],
        TestMode.STRESS_TEST: [ColumnIndex.JOINT, ColumnIndex.TEST, ColumnIndex.SEQUENCE],
        TestMode.DISCRETIZATION: [ColumnIndex.JOINT, ColumnIndex.TEST, ColumnIndex.SEQUENCE],
    }


class TestJointItem(TableItem):
    """Represents a joint item in the test configuration table for gain tuning.

    This class encapsulates the configuration parameters for testing individual joints during gain tuning
    operations. It manages test parameters like amplitude, period, phase, and step values, while handling
    the conversion between radians and degrees for rotational joints. The item provides properties for
    adjusting test parameters and maintains references to UI models for interactive editing.

    Args:
        name: Display name of the joint.
        joint_index: Index of the joint in the articulation.
        test: Whether this joint should be included in testing.
        sequence: Order sequence for testing this joint.
        dof_type: Type of degree of freedom (rotation or translation).
        amplitude: Amplitude value for sinusoidal test patterns.
        offset: Offset value applied to the joint during testing.
        period: Period duration for test patterns in seconds.
        phase: Phase shift for test patterns in seconds.
        step_max: Maximum step value for step test patterns.
        step_min: Minimum step value for step test patterns.
        user_provided: User-provided data source for custom test patterns.
        model: Parent model that manages this joint item.
        value_changed_fn: Callback function triggered when values change.
    """

    def __init__(
        self,
        name,
        joint_index,
        test,
        sequence,
        dof_type,
        amplitude,
        offset,
        period,
        phase,
        step_max,
        step_min,
        user_provided,
        model,
        value_changed_fn=None,
    ):
        super().__init__(joint_index, value_changed_fn)
        self.model = model
        self.dof_type = dof_type
        self.values_scale = 1.0
        if dof_type == DofType.Rotation:
            self.values_scale = 180.0 / np.pi
        if np.isinf(step_max) or step_max > 1e10:
            step_max = float("inf")
        else:
            step_max = step_max * self.values_scale
        if np.isinf(step_min) or step_min < -1e10:
            step_min = float("-inf")
        else:
            step_min = step_min * self.values_scale
        self.model_cols = [
            ui.SimpleStringModel(name),
            ui.SimpleBoolModel(test),
            ui.SimpleIntModel(joint_index + 1),
            ui.SimpleFloatModel(amplitude),
            ui.SimpleFloatModel(offset * self.values_scale),
            ui.SimpleFloatModel(period),
            ui.SimpleFloatModel(phase),
            ui.SimpleFloatModel(step_max),
            ui.SimpleFloatModel(step_min),
            ui.SimpleStringModel(user_provided),
        ]
        self.joint_index = joint_index
        for i in range(1, 9):
            self.model_cols[i].add_value_changed_fn(partial(self._on_value_changed, adjusted_col_id=i))

        # Add Config update callbacs
        self.model_cols[ColumnIndex.SEQUENCE].add_value_changed_fn(
            partial(
                self.on_update_sequence,
            )
        )
        self.model_cols[ColumnIndex.AMPLITUDE].add_value_changed_fn(
            partial(
                self.on_update_amplitude,
            )
        )
        self.model_cols[ColumnIndex.PERIOD].add_value_changed_fn(
            partial(
                self.on_update_period,
            )
        )
        self.model_cols[ColumnIndex.OFFSET].add_value_changed_fn(
            partial(
                self.on_update_offset,
            )
        )
        self.model_cols[ColumnIndex.PHASE].add_value_changed_fn(
            partial(
                self.on_update_phase,
            )
        )
        self.model_cols[ColumnIndex.STEP_MAX].add_value_changed_fn(
            partial(
                self.on_update_step_max,
            )
        )
        self.model_cols[ColumnIndex.STEP_MIN].add_value_changed_fn(
            partial(
                self.on_update_step_min,
            )
        )
        self.model_cols[ColumnIndex.USER_PROVIDED].add_value_changed_fn(
            partial(
                self.on_update_user_provided,
            )
        )
        self.model_cols[ColumnIndex.TEST].add_value_changed_fn(
            partial(
                self.on_update_test,
            )
        )
        self.model_cols[ColumnIndex.SEQUENCE].add_value_changed_fn(
            partial(
                self.on_update_sequence,
            )
        )
        self.value_field = {}
        self.mode = GainsTestMode.SINUSOIDAL

    def on_update_position(self, model, *args):
        """Updates the joint position value in the model.

        Args:
            model: The model containing the position value.
            *args: Additional arguments passed to the callback.
        """
        self.model.joint_positions[self.joint_index] = model.get_value_as_float()

    def on_update_velocity(self, model, *args):
        """Updates the maximum velocity value in the model.

        Args:
            model: The model containing the velocity value.
            *args: Additional arguments passed to the callback.
        """
        self.model.v_max[self.joint_index] = model.get_value_as_float()

    def on_update_step_max(self, model, *args):
        """Updates the maximum step value for the joint test.

        Args:
            model: The model containing the step maximum value.
            *args: Additional arguments passed to the callback.
        """
        pass

    def on_update_step_min(self, model, *args):
        """Updates the minimum step value for the joint test.

        Args:
            model: The model containing the step minimum value.
            *args: Additional arguments passed to the callback.
        """
        pass

    def on_update_sequence(self, model, *args):
        """Updates the test sequence value for the joint.

        Args:
            model: The model containing the sequence value.
            *args: Additional arguments passed to the callback.
        """
        pass

    def on_update_amplitude(self, model, *args):
        """Updates the amplitude value for the joint test.

        Args:
            model: The model containing the amplitude value.
            *args: Additional arguments passed to the callback.
        """
        pass

    def on_update_period(self, model, *args):
        """Updates the period value for the joint test.

        Args:
            model: The model containing the period value.
            *args: Additional arguments passed to the callback.
        """
        pass

    def on_update_offset(self, model, *args):
        """Updates the offset value for the joint test.

        Args:
            model: The model containing the offset value.
            *args: Additional arguments passed to the callback.
        """
        pass

    def on_update_phase(self, model, *args):
        """Updates the phase value for the joint test.

        Args:
            model: The model containing the phase value.
            *args: Additional arguments passed to the callback.
        """
        pass

    def on_update_test(self, model, *args):
        """Updates the test enable/disable state for the joint.

        Args:
            model: The model containing the test state value.
            *args: Additional arguments passed to the callback.
        """
        pass

    def on_update_user_provided(self, model, *args):
        """Handles updates to the user-provided data source parameter.

        Args:
            model: The model that triggered the update.
            *args: Variable length argument list for additional parameters.
        """
        pass

    @property
    def test(self) -> bool:
        """Whether this joint is enabled for testing.

        Returns:
            True if the joint is enabled for testing.
        """
        return self.model_cols[ColumnIndex.TEST].get_value_as_bool()

    @test.setter
    def test(self, value: bool):
        self.model_cols[ColumnIndex.TEST].set_value(value)
        self.model._item_changed(self)

    @property
    def amplitude(self) -> float:
        """Amplitude value for the sinusoidal test motion.

        Returns:
            The amplitude value as a float.
        """
        return self.model_cols[ColumnIndex.AMPLITUDE].get_value_as_float()

    @amplitude.setter
    def amplitude(self, value: float):
        self.model_cols[ColumnIndex.AMPLITUDE].set_value(value)

    @property
    def step_max(self) -> float:
        """Maximum step value for the test motion.

        Returns:
            The maximum step value as a float.
        """
        return self.model_cols[ColumnIndex.STEP_MAX].get_value_as_float()

    @step_max.setter
    def step_max(self, value: float):
        self.model_cols[ColumnIndex.STEP_MAX].set_value(value)

    @property
    def step_min(self) -> float:
        """Minimum step value for the test motion.

        Returns:
            The minimum step value as a float.
        """
        return self.model_cols[ColumnIndex.STEP_MIN].get_value_as_float()

    @step_min.setter
    def step_min(self, value: float):
        self.model_cols[ColumnIndex.STEP_MIN].set_value(value)

    @property
    def period(self) -> float:
        """Period duration for the test motion in seconds.

        Returns:
            The period value as a float.
        """
        return self.model_cols[ColumnIndex.PERIOD].get_value_as_float()

    @period.setter
    def period(self, value: float):
        self.model_cols[ColumnIndex.PERIOD].set_value(value)

    @property
    def phase(self) -> float:
        """Phase offset for the test motion in seconds.

        Returns:
            The phase value as a float.
        """
        return self.model_cols[ColumnIndex.PHASE].get_value_as_float()

    @phase.setter
    def phase(self, value: float):
        self.model_cols[ColumnIndex.PHASE].set_value(value)

    @property
    def offset(self) -> float:
        """Offset value for the test motion position.

        Returns:
            The offset value as a float.
        """
        return self.model_cols[ColumnIndex.OFFSET].get_value_as_float()

    @offset.setter
    def offset(self, value: float):
        self.model_cols[ColumnIndex.OFFSET].set_value(value)

    @property
    def sequence(self) -> int:
        """Sequence order for the joint test execution.

        Returns:
            The sequence number as an integer.
        """
        return self.model_cols[ColumnIndex.SEQUENCE].get_value_as_int()

    @sequence.setter
    def sequence(self, value: int):
        self.model_cols[ColumnIndex.SEQUENCE].set_value(value)

    @property
    def user_provided(self) -> str:
        """User-provided data source for the test motion.

        Returns:
            The user-provided data source as a string.
        """
        return self.model_cols[ColumnIndex.USER_PROVIDED].get_value_as_string()

    @user_provided.setter
    def user_provided(self, value: str):
        self.model_cols[ColumnIndex.USER_PROVIDED].set_value(value)

    def get_item_value(self, col_id: int = 0):
        """Retrieves the value for the specified column.

        Args:
            col_id: Column index to get the value from.

        Returns:
            The value from the column as a string for column 0, bool for column 1, or float for other columns.
        """
        if col_id == 0:
            return self.model_cols[col_id].get_value_as_string()
        elif col_id == 1:
            return self.model_cols[col_id].get_value_as_bool()
        return self.model_cols[col_id].get_value_as_float()

    def set_item_value(self, col_id: int, value):
        """Sets the value for the specified column.

        Args:
            col_id: Column index to set the value for.
            value: The value to set in the column.
        """
        self.model_cols[col_id].set_value(value)

    def get_value_model(self, col_id: int = 0):
        """Gets the UI model for the specified column.

        Args:
            col_id: Column index to get the model from.

        Returns:
            The UI model for the specified column.
        """
        return self.model_cols[col_id]


class TestJointItemDelegate(TableItemDelegate):
    """A delegate class that handles the rendering and interaction of test joint configuration items in a table view.

    This delegate is responsible for creating and managing UI widgets for each column of the test joint table,
    including joint names, test checkboxes, sequence numbers, amplitude controls, offset fields, period settings,
    phase adjustments, step limits, and data source specifications. It provides column headers with tooltips
    and handles different test modes (sinusoidal, step, user-provided) by dynamically showing or hiding
    relevant columns based on the current mode.

    The delegate creates interactive widgets such as checkboxes for enabling/disabling joint tests, drag fields
    for numeric parameters with appropriate units and limits, and string fields for custom data sources.
    It supports bulk editing operations and maintains proper styling through the Omni UI framework.

    Args:
        model: The TestJointModel instance that provides the data and structure for the table view.
    """

    header_tooltip = {
        ColumnIndex.JOINT: "Joint",
        ColumnIndex.TEST: "Test",
        ColumnIndex.SEQUENCE: "Sequence",
        ColumnIndex.AMPLITUDE: "Amplitude",
        ColumnIndex.OFFSET: "Offset",
        ColumnIndex.PERIOD: "Period",
        ColumnIndex.PHASE: "Phase",
        ColumnIndex.STEP_MAX: "Step Max",
        ColumnIndex.STEP_MIN: "Step Min",
        ColumnIndex.USER_PROVIDED: "Data Source",
    }
    """Dictionary mapping column indices to their tooltip text displayed in the table header."""
    header = {
        ColumnIndex.JOINT: "Joint",
        ColumnIndex.TEST: "Test",
        ColumnIndex.SEQUENCE: "Sequence",
        ColumnIndex.AMPLITUDE: "Amplitude",
        ColumnIndex.OFFSET: "Offset",
        ColumnIndex.PERIOD: "Period",
        ColumnIndex.PHASE: "Phase",
        ColumnIndex.STEP_MAX: "Step Max",
        ColumnIndex.STEP_MIN: "Step Min",
        ColumnIndex.USER_PROVIDED: "Data Source",
    }
    """Dictionary mapping column indices to their display text shown in the table header."""

    def __init__(self, model):

        self.column_headers = {}
        super().__init__(model)

    def init_model(self):
        """Initializes the model with default test mode settings."""
        self.mode = TestMode.SINUSOIDAL

    @property
    def mode(self):
        """Current test mode of the delegate.

        Returns:
            The test mode from the underlying model.
        """
        return self._model.mode

    @mode.setter
    def mode(self, mode):
        self._model.mode = mode

    def build_branch(self, model, item=None, column_id=0, level=0, expanded=False):
        """Builds branch widgets for tree view items.

        Args:
            model: The model containing the data.
            item: The item to build the branch for.
            column_id: Column identifier for the branch.
            level: Hierarchy level of the item.
            expanded: Whether the branch is expanded.
        """
        pass

    def model_col_id(self, column_id):
        """Maps display column ID to model column ID based on current mode.

        Args:
            column_id: The display column identifier.

        Returns:
            The corresponding model column ID, or None if invalid.
        """
        if column_id < len(self._model.column_id_map[self.mode]):
            return self._model.column_id_map[self.mode][column_id]
        return None

    def build_header(self, column_id=0):
        """Builds header widgets for table columns.

        Args:
            column_id: Column identifier for the header.
        """
        if self.model_col_id(column_id) is None:
            return
        model_col_id = self.model_col_id(column_id)
        with ui.ZStack(style_type_name_override="TreeView"):
            ui.Rectangle(
                name="Header",
                style_type_name_override="TreeView",
            )
            alignment = ui.Alignment.CENTER
            offset = 10
            if model_col_id in [ColumnIndex.JOINT, ColumnIndex.SEQUENCE]:
                alignment = ui.Alignment.LEFT
            if model_col_id in [ColumnIndex.AMPLITUDE, ColumnIndex.PERIOD, ColumnIndex.PHASE, ColumnIndex.OFFSET]:
                offset = 0
            with ui.HStack():
                ui.Spacer(width=offset)
                if model_col_id not in [ColumnIndex.JOINT, ColumnIndex.SEQUENCE, ColumnIndex.TEST]:
                    ui.Spacer()
                with ui.VStack():
                    ui.Spacer(height=ui.Pixel(3))
                    ui.Label(
                        TestJointItemDelegate.header[model_col_id],
                        tooltip=TestJointItemDelegate.header_tooltip[model_col_id],
                        name="header",
                        style_type_name_override="TreeView",
                        # elided_text=True,
                        alignment=alignment,
                    )
                ui.Spacer()
                self.build_sort_button(column_id)

    def update_defaults(self):
        """Updates default values for all value fields in the model items."""
        for item in self.__model.get_item_children():
            for i in [1, 2, 3, 4, 5]:
                field = item.value_field.get(i)
                if field:
                    field.update_default_value()

    def build_widget(self, model, item=None, index=0, level=0, expanded=False):
        """Builds UI widgets for table cells based on column type and test mode.

        Args:
            model: The model containing the data.
            item: The item to build the widget for.
            index: Column index for the widget.
            level: Hierarchy level of the item.
            expanded: Whether the item is expanded.
        """
        if self.model_col_id(index) is None:
            return
        if item:
            item.mode = self.mode
            model_col_id = self.model_col_id(index)
            if model_col_id == ColumnIndex.JOINT:
                with ui.ZStack(height=ITEM_HEIGHT, style_type_name_override="TreeView"):
                    ui.Rectangle(name="treeview_first_item")
                    with ui.VStack():
                        ui.Spacer()
                        with ui.HStack(height=0):
                            ui.Label(
                                str(model.get_item_value(item, model_col_id)),
                                tooltip=model.get_item_value(item, model_col_id),
                                name="treeview_item",
                                elided_text=True,
                                height=0,
                            )
                            ui.Spacer(width=1)
                        ui.Spacer()

            if model_col_id == ColumnIndex.TEST:
                with ui.ZStack(height=ITEM_HEIGHT, style_type_name_override="TreeView"):
                    ui.Rectangle(name="treeview_first_item")
                    with ui.VStack():
                        ui.Spacer()
                        with ui.HStack(height=0):
                            ui.Spacer()
                            check_box = ui.CheckBox(width=10, height=0)
                            check_box.model.set_value(item.test)

                            def on_click(value_model):
                                item.test = value_model.get_value_as_bool()

                            check_box.model.add_value_changed_fn(on_click)
                            ui.Spacer()
                        ui.Spacer()
            if model_col_id == ColumnIndex.SEQUENCE:
                with ui.ZStack(height=ITEM_HEIGHT):
                    ui.Rectangle(name="treeview_item")
                    with ui.VStack():
                        ui.Spacer()
                        item.value_field[model_col_id] = CellLabelField(
                            model.get_item_value_model(item, model_col_id), ui.IntDrag, "%d"
                        )
                        model.get_item_value_model(item, model_col_id)
                        item.value_field[model_col_id].field.min = 1
                        item.value_field[model_col_id].field.max = len(self.get_children())
                        ui.Spacer()
            elif model_col_id in [
                ColumnIndex.AMPLITUDE,
                ColumnIndex.PERIOD,
                ColumnIndex.PHASE,
                ColumnIndex.OFFSET,
                ColumnIndex.STEP_MAX,
                ColumnIndex.STEP_MIN,
            ]:
                min_value = 0
                max_value = 100.0
                if model_col_id in [ColumnIndex.STEP_MAX, ColumnIndex.STEP_MIN, ColumnIndex.OFFSET]:
                    min_value = item.step_min
                    max_value = item.step_max

                # Position-valued columns carry the joint's own unit: a rotational
                # DOF is scaled to degrees by ``TestJointItem``, a translational
                # one is left in stage linear units.
                unit = "deg" if item.dof_type == DofType.Rotation else "m"

                if model_col_id in [ColumnIndex.AMPLITUDE]:
                    unit = "%"
                elif model_col_id == ColumnIndex.PERIOD:
                    unit = "s"
                elif model_col_id == ColumnIndex.PHASE:
                    unit = "s"

                with ui.ZStack(height=ITEM_HEIGHT):
                    ui.Rectangle(name="treeview_item")
                    with ui.VStack():
                        ui.Spacer()
                        item.value_field[model_col_id] = CellLabelField(
                            model.get_item_value_model(item, model_col_id), ui.FloatDrag, "%.2f"
                        )
                        item.value_field[model_col_id].field.min = min_value
                        item.value_field[model_col_id].field.max = max_value
                        ui.Spacer()
                    if unit:
                        with ui.VStack():
                            ui.Spacer()
                            with ui.HStack():
                                ui.Spacer()
                                ui.Label(unit, width=0, name="unit")
                                ui.Spacer(width=10)
                            ui.Spacer()
            elif model_col_id == ColumnIndex.USER_PROVIDED:
                with ui.ZStack(height=ITEM_HEIGHT):
                    ui.Rectangle(name="treeview_item")
                    with ui.VStack():
                        ui.Spacer()
                        item.value_field[index] = CellLabelField(
                            model.get_item_value_model(item, model_col_id), ui.StringField, "%s"
                        )


class TestJointModel(TableModel):
    """Model for managing joint test configurations in the gains tuner.

    This model provides a data structure for storing and managing test parameters for individual joints during
    gains tuning operations. It supports multiple test modes including sinusoidal, step, and user-provided data
    modes, with appropriate parameter sets for each mode.

    The model automatically filters joints to include only testable joints (excluding fixed joints and mimic
    joints) and initializes test parameters with appropriate defaults based on joint limits and DOF types.
    Rotational joints have their values scaled to degrees for user interface purposes.

    Args:
        gains_tuner: The gains tuner instance that provides access to articulation and joint information.
        value_changed_fn: Callback function invoked when joint test parameters are modified.
        **kwargs: Additional keyword arguments passed to the parent TableModel class.
    """

    def __init__(self, gains_tuner, value_changed_fn, **kwargs):
        self.column_id_map = default_test_column_id_map()
        super().__init__(value_changed_fn)
        self._articulation = gains_tuner.get_articulation()
        self._gains_tuner = gains_tuner
        self.lower_joint_limits, self.upper_joint_limits = [i[0].list() for i in self._articulation.get_dof_limits()]

        self.joint_indices = [
            joint_index
            for joint_index in self._gains_tuner.get_all_joint_indices()
            if self.is_joint_testable(joint_index)
        ]

        joint_entries = self._gains_tuner.get_joint_entries()
        self._children = [
            TestJointItem(
                name=joint_entries[i].display_name,
                joint_index=joint_index,
                test=True,
                sequence=i,
                dof_type=self._gains_tuner.get_dof_type(joint_index),
                amplitude=100.0,
                offset=0.0,
                period=1.0,
                phase=0.0,
                step_max=self.upper_joint_limits[joint_index],
                step_min=self.lower_joint_limits[joint_index],
                user_provided="",
                model=self,
                value_changed_fn=self._on_joint_changed,
            )
            for i, joint_index in enumerate(self.joint_indices)
        ]

    def is_joint_testable(self, joint_index: int) -> bool:
        """Determines if a joint at the given index can be tested.

        Args:
            joint_index: Index of the joint to check.

        Returns:
            True if the joint is testable (not a fixed joint or mimic joint), False otherwise.
        """
        joint = self._gains_tuner.get_joint_entries()[joint_index].joint
        return not (pxr.UsdPhysics.FixedJoint(joint) or gain_tuner.is_joint_mimic(joint))

    @property
    def mode(self) -> GainsTestMode:
        """Current test mode for the joints.

        Returns:
            The test mode enumeration value.
        """
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode
        self._item_changed(None)

    def init_model(self):
        """Initializes the model by setting the test mode to sinusoidal."""
        self.mode = GainsTestMode.SINUSOIDAL

    def get_item_value_model_count(self, item) -> int:
        """The number of columns based on the current test mode.

        Args:
            item: The table item to get column count for.

        Returns:
            Number of columns for the current test mode.
        """
        if self.mode in self.column_id_map:
            return len(self.column_id_map[self.mode])
        return 1


class TestJointWidget(TableWidget):
    """A widget for configuring joint test parameters in the gains tuner interface.

    Provides a table-based interface for setting up joint testing configurations including test modes
    (sinusoidal, step, user-provided), amplitudes, periods, phases, and other parameters. The widget
    allows users to select which joints to test and configure their specific test parameters through
    an interactive table view.

    Supports bulk editing functionality where changes to one selected joint can be applied to multiple
    joints simultaneously. The widget automatically filters out non-testable joints (fixed joints and
    mimic joints) and provides appropriate input controls based on the joint's degrees of freedom type.

    Args:
        gains_tuner: The gains tuner instance that manages the articulation and joint configurations.
        value_changed_fn: Callback function invoked when joint test parameters are modified.
        no_name_column: When True the JOINT name column is excluded from the table.  Use this
            when the widget is embedded in a split-pane that already shows joint names on the left.
    """

    # Minimum pixel widths indexed by ColumnIndex value.
    _min_widths = {
        ColumnIndex.JOINT: 80,
        ColumnIndex.TEST: 36,
        ColumnIndex.SEQUENCE: 50,
        ColumnIndex.AMPLITUDE: 60,
        ColumnIndex.OFFSET: 60,
        ColumnIndex.PERIOD: 60,
        ColumnIndex.PHASE: 60,
        ColumnIndex.STEP_MAX: 60,
        ColumnIndex.STEP_MIN: 60,
        ColumnIndex.USER_PROVIDED: 80,
    }

    def __init__(self, gains_tuner, value_changed_fn=None, no_name_column=False, hidden_columns=None):
        # TEST (checkbox) stays narrow and fixed; everything else fills proportionally.
        self.column_widths = [
            ui.Fraction(2),  # JOINT
            ui.Pixel(44),  # TEST  — fixed, just wide enough for the checkbox
            ui.Fraction(1),  # SEQUENCE
            ui.Fraction(2),  # AMPLITUDE
            ui.Fraction(2),  # OFFSET
            ui.Fraction(2),  # PERIOD
            ui.Fraction(2),  # PHASE
            ui.Fraction(2),  # STEP_MAX
            ui.Fraction(2),  # STEP_MIN
            ui.Fraction(2),  # USER_PROVIDED
        ]
        # Columns hidden via the options menu (set of ColumnIndex values).
        self._hidden_columns = set(hidden_columns) if hidden_columns else set()
        if gains_tuner.get_articulation():
            model = TestJointModel(gains_tuner, self._on_value_changed)
            if no_name_column:
                # Drop the JOINT column from all mode mappings so it is never rendered.
                model.column_id_map = {
                    mode: [col for col in cols if col != ColumnIndex.JOINT]
                    for mode, cols in model.column_id_map.items()
                }
            # Remember the full (post no-name) column map so visibility toggles are
            # non-destructive — hiding then showing a column restores it exactly.
            self._base_column_id_map = {m: list(cols) for m, cols in model.column_id_map.items()}
            if self._hidden_columns:
                model.column_id_map = {
                    m: [c for c in cols if c not in self._hidden_columns]
                    for m, cols in self._base_column_id_map.items()
                }
            delegate = TestJointItemDelegate(model)
            mode = GainsTestMode.SINUSOIDAL
            super().__init__(value_changed_fn, model, delegate, mode)
            self._enable_bulk_edit = True

    def set_column_visible(self, col_index, visible: bool):
        """Show/hide a Test table column by ``ColumnIndex`` and rebuild the tree."""
        if not hasattr(self, "_base_column_id_map"):
            return
        if visible:
            self._hidden_columns.discard(col_index)
        else:
            self._hidden_columns.add(col_index)
        self.model.column_id_map = {
            m: [c for c in cols if c not in self._hidden_columns] for m, cols in self._base_column_id_map.items()
        }
        frame = getattr(self, "_tree_frame", None)
        if frame is not None:
            frame.rebuild()

    def is_column_visible(self, col_index) -> bool:
        return col_index not in self._hidden_columns

    def switch_mode(self, mode):
        """Switches the test mode and updates the tree view column widths.

        Args:
            mode: The test mode to switch to.
        """
        super().switch_mode(mode)
        # ``list`` (the TreeView) is created lazily by ``_build_test_tree_impl``.
        # ``switch_mode`` can be called before that build_fn runs (e.g. right after
        # construction in ``_build_gains_tuning_frame``), so guard against the
        # attribute not existing yet — the tree picks up the mode's columns when it
        # is built. Without this guard the AttributeError escapes the frame build_fn
        # and the frame rebuilds every frame (UI blinking).
        if getattr(self, "list", None):
            active_cols = self.model.column_id_map[self.model.mode]
            self.list.column_widths = [self.column_widths[i] for i in active_cols]
            # ``min_column_widths`` requires ``ui.Length`` values; ``_min_widths`` holds
            # plain-int pixel widths, so wrap them (the TreeView constructor coerces
            # ints, but this post-build property setter does not).
            self.list.min_column_widths = [ui.Pixel(self._min_widths.get(c, 50)) for c in active_cols]

    # def switch_radian_degree(self, radian_degree_mode):
    #     # TODO: Implement this
    #     carb.log_error("switch_radian_degree is not implemented")

    def _on_value_changed(self, joint_item, col_id=1, adjusted_col_id=None):
        """Handles value changes in joint test parameters.

        Args:
            joint_item: The joint item that had its value changed.
            col_id: The column ID in the current view.
            adjusted_col_id: The actual column ID if different from col_id.
        """
        if adjusted_col_id is None:
            value_col_id = self.model.column_id_map[self.model.mode][col_id]
        else:
            value_col_id = adjusted_col_id
        if self._enable_bulk_edit:
            if joint_item not in self.list.selection:
                self.list.selection = [joint_item]
            self.set_bulk_edit(False)
            for item in self.list.selection:
                if item is not joint_item:
                    item.set_item_value(value_col_id, joint_item.get_item_value(value_col_id))
                    self.model._item_changed(item)
            self.set_bulk_edit(True)
        if self._value_changed_fn:
            self._value_changed_fn(joint_item.joint)

    def build_tree_view(self):
        """Builds the tree view inside a frame so column-visibility changes can rebuild it."""
        self._tree_frame = ui.Frame(build_fn=self._build_test_tree_impl)

    def _build_test_tree_impl(self):
        """Creates the TreeView for the currently visible columns."""
        active_cols = self.model.column_id_map[self.model.mode]
        self.list = ui.TreeView(
            self.model,
            delegate=self.delegate,
            alignment=ui.Alignment.LEFT_TOP,
            column_widths=[self.column_widths[i] for i in active_cols],
            min_column_widths=[self._min_widths.get(c, 50) for c in active_cols],
            columns_resizable=True,
            header_visible=True,
            height=ui.Fraction(1),
        )

    def select_all(self):
        """Selects all joints for testing by setting their test property to True."""
        for item in self.model.get_item_children():
            item.test = True

    def clear_all(self):
        """Deselects all joints from testing by setting their test property to False."""
        for item in self.model.get_item_children():
            item.test = False
