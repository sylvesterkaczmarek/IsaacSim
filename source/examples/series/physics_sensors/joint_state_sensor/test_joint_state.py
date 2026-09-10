# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `joint_state`.

Run with ``pytest test_joint_state.py`` or ``python main.py --self-test``; plain assert functions,
so both work.

The scene ``main.py`` drives is deliberately clean -- both DOFs are undriven and their axes are
vertical -- which hides mistakes a messier articulation would expose. These tests use synthetic
states instead: DOF counts that do not match, mass units that differ from length units, windows
whose timestamps repeat or rewind, and selections that reorder or drop DOFs.
"""

from __future__ import annotations

import warnings
from typing import Any

import joint_state
from joint_state import ROTATION, TRANSLATION, ArticulationState, Dof, JointStateSensor

TOLERANCE = 1.0e-12

# A centimetre stage with grams for mass, so a length scale and a mass scale that differ cannot be
# swapped without a test noticing.
CENTIMETRES = 0.01
GRAMS = 0.001

LIFT = Dof("lift", TRANSLATION)
SPIN = Dof("spin", ROTATION)
BOTH = [LIFT, SPIN]


def _assert_close(actual: Any, expected: Any, tolerance: Any = TOLERANCE) -> None:
    """Assert two sequences agree element-wise.

    Args:
        actual: Value under test.
        expected: Value it should match.
        tolerance: Largest permitted difference per element.
    """
    left = [float(value) for value in actual]
    right = [float(value) for value in expected]
    assert len(left) == len(right), f"length {len(left)} != {len(right)}"
    worst = max((abs(a - b) for a, b in zip(left, right)), default=0.0)
    assert worst <= tolerance, f"{left} != {right} (worst {worst})"


def _assert_raises(error_type: Any, call: Any, description: Any) -> None:
    """Assert a call raises a given error.

    Args:
        error_type: Expected exception type.
        call: Zero-argument callable under test.
        description: What was expected to be rejected, for the failure message.
    """
    try:
        call()
    except error_type:
        return
    raise AssertionError(f"{description} was not rejected")


def _state(positions: Any, velocities: Any, efforts: Any = None) -> Any:
    """Build a raw articulation state.

    Args:
        positions: Raw DOF positions.
        velocities: Raw DOF velocities.
        efforts: Raw measured efforts, or ``None`` when the backend reports none.

    Returns:
        The state.
    """
    return ArticulationState(positions=positions, velocities=velocities, efforts=efforts)


def _drive(sensor: Any, states: Any, time_step: Any = 1.0) -> Any:
    """Push states one time step apart and read.

    Args:
        sensor: Sensor to drive.
        states: Raw articulation states, oldest first.
        time_step: Interval between samples, in seconds.

    Returns:
        The reading after the last sample.
    """
    for step, state in enumerate(states):
        sensor.sample(step * time_step, state)
    return sensor.read()


def _sensor(**overrides: Any) -> Any:
    """Build a sensor over both DOFs of a centimetre-and-gram stage.

    Args:
        **overrides: Keyword arguments overriding the defaults.

    Returns:
        The sensor.
    """
    arguments = {"meters_per_unit": CENTIMETRES, "kilograms_per_unit": GRAMS}
    arguments.update(overrides)
    return JointStateSensor(arguments.pop("dofs", BOTH), **arguments)


# ----------------------------------------------------------------------------------------------
# DOF metadata.
# ----------------------------------------------------------------------------------------------


def test_a_dof_type_the_module_does_not_define_is_rejected() -> None:
    """A DOF is a rotation or a translation, and nothing else."""
    _assert_raises(ValueError, lambda: Dof("elbow", "spherical"), "an unknown DOF type")
    _assert_raises(ValueError, lambda: Dof("elbow", "Rotation"), "a mis-cased DOF type")


def test_the_shipped_numeric_dof_types_map_onto_the_named_ones() -> None:
    """The C++ reading encodes the same distinction as 0 and 1."""
    assert joint_state.dof_type_from_code(0) == ROTATION
    assert joint_state.dof_type_from_code(1) == TRANSLATION
    _assert_raises(ValueError, lambda: joint_state.dof_type_from_code(2), "an undefined DOF type code")


def test_a_dof_type_code_that_is_not_an_integer_is_rejected() -> None:
    """Converting first would round a malformed code onto a valid DOF type.

    ``int(0.5)`` is ``0`` and ``int(True)`` is ``1``, so both would name a real DOF type instead of
    failing at the boundary.
    """
    for code in (0.5, 1.0, True, False, "1", None):
        _assert_raises(ValueError, lambda: joint_state.dof_type_from_code(code), f"the code {code!r}")


def test_a_dof_type_code_from_the_reading_is_accepted() -> None:
    """The reading carries `dof_types` as ``uint8``, which is an integer without being an ``int``.

    NumPy registers its scalar types with `numbers.Integral` rather than subclassing ``int``, so a
    stub registered the same way stands in for one and the test needs no NumPy of its own.
    """
    import numbers

    class _Uint8:
        """Provide only the parts of an integer this conversion touches.

        Args:
            value: Integer value represented by the stand-in.
        """

        def __init__(self, value: int) -> None:
            self._value = value

        def __int__(self) -> int:
            return self._value

    numbers.Integral.register(_Uint8)

    assert joint_state.dof_type_from_code(_Uint8(0)) == ROTATION
    assert joint_state.dof_type_from_code(_Uint8(1)) == TRANSLATION


def test_an_articulation_with_no_dofs_is_rejected() -> None:
    """There is no joint state to report for an articulation without DOFs."""
    _assert_raises(ValueError, lambda: _sensor(dofs=[]), "an empty DOF list")


# ----------------------------------------------------------------------------------------------
# Unit conversion.
# ----------------------------------------------------------------------------------------------


def test_a_translation_dof_is_scaled_by_the_linear_unit() -> None:
    """Positions and velocities of a translation DOF are stage units."""
    sensor = JointStateSensor([LIFT], meters_per_unit=CENTIMETRES)
    reading = _drive(sensor, [_state([-80.0], [-25.0])] * 2)
    assert reading is not None
    _assert_close(reading.positions, [-0.8])
    _assert_close(reading.velocities, [-0.25])


def test_a_rotation_dof_is_not_scaled_by_the_linear_unit() -> None:
    """Radians are radians whatever the stage's linear unit."""
    for meters_per_unit in (0.01, 1.0, 100.0):
        sensor = JointStateSensor([SPIN], meters_per_unit=meters_per_unit)
        reading = _drive(sensor, [_state([1.5], [3.0])] * 2)
        assert reading is not None
        _assert_close(reading.positions, [1.5])
        _assert_close(reading.velocities, [3.0])


def test_an_effort_carries_one_more_power_of_length_than_a_position() -> None:
    """A force scales by length once and a torque twice, and both by the mass unit.

    Driving both DOFs with the same raw effort makes the two scales impossible to confuse.
    """
    sensor = _sensor()
    reading = _drive(sensor, [_state([0.0, 0.0], [0.0, 0.0], [1000.0, 1000.0])] * 2)
    assert reading is not None
    force, torque = reading.efforts[0], reading.efforts[1]
    _assert_close([force], [1000.0 * GRAMS * CENTIMETRES])
    _assert_close([torque], [1000.0 * GRAMS * CENTIMETRES * CENTIMETRES])


def test_the_mass_unit_reaches_only_the_efforts() -> None:
    """Positions and velocities do not depend on the mass unit."""
    readings = []
    for kilograms_per_unit in (1.0, GRAMS):
        sensor = JointStateSensor(BOTH, meters_per_unit=CENTIMETRES, kilograms_per_unit=kilograms_per_unit)
        readings.append(_drive(sensor, [_state([3.0, 1.0], [4.0, 2.0], [5.0, 5.0])] * 2))
    assert readings[0] is not None and readings[1] is not None
    _assert_close(readings[0].positions, readings[1].positions)
    _assert_close(readings[0].velocities, readings[1].velocities)
    _assert_close(readings[0].efforts, [value / GRAMS for value in readings[1].efforts])


def test_an_acceleration_uses_the_same_scale_as_its_velocity() -> None:
    """Differencing a converted velocity converts the acceleration too."""
    sensor = _sensor()
    # Raw velocities of -100 stage units/s per second on the lift, and 2 rad/s per second on the spin.
    states = [_state([0.0, 0.0], [-100.0 * step, 2.0 * step]) for step in range(2)]
    reading = _drive(sensor, states)
    assert reading is not None
    _assert_close(reading.accelerations, [-100.0 * CENTIMETRES, 2.0])


def test_a_non_positive_or_non_finite_unit_scale_is_rejected() -> None:
    """A unit scale multiplies every reading, so a bad one is caught at construction."""
    for scale in (0.0, -1.0, float("inf"), float("nan")):
        _assert_raises(ValueError, lambda: _sensor(meters_per_unit=scale), f"meters_per_unit={scale}")
        _assert_raises(ValueError, lambda: _sensor(kilograms_per_unit=scale), f"kilograms_per_unit={scale}")


# ----------------------------------------------------------------------------------------------
# Selection by name.
# ----------------------------------------------------------------------------------------------


def test_selection_reorders_a_reading_without_changing_it() -> None:
    """A consumer with a fixed message layout picks the order it needs."""
    every = _drive(_sensor(), [_state([3.0, 1.0], [4.0, 2.0], [5.0, 6.0])] * 2)
    reversed_reading = _drive(_sensor(names=["spin", "lift"]), [_state([3.0, 1.0], [4.0, 2.0], [5.0, 6.0])] * 2)
    assert every is not None and reversed_reading is not None
    assert every.names == ("lift", "spin")
    assert reversed_reading.names == ("spin", "lift")
    assert reversed_reading.types == (ROTATION, TRANSLATION)
    _assert_close(reversed_reading.positions, list(reversed(every.positions)))
    _assert_close(reversed_reading.velocities, list(reversed(every.velocities)))
    # The efforts differ per DOF type, so reversing them also checks the scales moved with them.
    _assert_close(reversed_reading.efforts, list(reversed(every.efforts)))


def test_selection_can_drop_dofs() -> None:
    """Reporting a subset still indexes the raw arrays by articulation order."""
    sensor = _sensor(names=["spin"])
    reading = _drive(sensor, [_state([3.0, 1.0], [4.0, 2.0])] * 2)
    assert reading is not None
    assert reading.names == ("spin",)
    _assert_close(reading.positions, [1.0])
    _assert_close(reading.velocities, [2.0])


def test_selection_can_repeat_a_dof() -> None:
    """Nothing stops a layout naming the same DOF twice."""
    sensor = _sensor(names=["lift", "lift"])
    reading = _drive(sensor, [_state([3.0, 1.0], [4.0, 2.0])] * 2)
    assert reading is not None
    assert reading.names == ("lift", "lift")
    _assert_close(reading.positions, [0.03, 0.03])


def test_an_unknown_selected_name_is_rejected() -> None:
    """Silently dropping a name a caller asked for would publish a short message."""
    _assert_raises(ValueError, lambda: _sensor(names=["elbow"]), "an unknown DOF name")


def test_an_ambiguous_selected_name_is_rejected() -> None:
    """Two DOFs can share a name, and then selecting it has no single answer."""
    twins = [Dof("wheel", ROTATION), Dof("wheel", ROTATION)]
    # Reporting every DOF is still fine: only selection needs names to be unique.
    assert _sensor(dofs=twins).names == ("wheel", "wheel")
    _assert_raises(ValueError, lambda: _sensor(dofs=twins, names=["wheel"]), "an ambiguous DOF name")


# ----------------------------------------------------------------------------------------------
# Sensor lifecycle and argument validation.
# ----------------------------------------------------------------------------------------------


def test_read_returns_none_until_the_window_is_full() -> None:
    """A reading is undefined until the window is full."""
    for filter_width in (1, 2, 5):
        sensor = _sensor(filter_width=filter_width)
        for step in range(2 * filter_width):
            assert sensor.read() is None
            sensor.sample(step, _state([0.0, 0.0], [0.0, 0.0]))
        assert sensor.read() is not None


def test_the_buffer_saturates_instead_of_growing() -> None:
    """Only ``2 * filter_width`` samples are retained."""
    sensor = _sensor(filter_width=2)
    for step in range(20):
        sensor.sample(step, _state([0.0, 0.0], [0.0, 0.0]))
    assert sensor.sample_count == 4


def test_reset_discards_buffered_samples() -> None:
    """Resetting makes readings undefined, so a discontinuity cannot leak into a difference."""
    sensor = _sensor()
    _drive(sensor, [_state([0.0, 0.0], [0.0, 0.0])] * 2)
    assert sensor.read() is not None
    sensor.reset()
    assert sensor.sample_count == 0
    assert sensor.read() is None


def test_a_filter_width_below_one_is_rejected() -> None:
    """A window narrower than one sample has no meaning."""
    for filter_width in (0, -1):
        _assert_raises(ValueError, lambda: _sensor(filter_width=filter_width), f"filter_width={filter_width}")


def test_the_linear_unit_is_required() -> None:
    """There is no default linear unit; it depends on the stage."""
    _assert_raises(TypeError, lambda: JointStateSensor(BOTH), "a missing meters_per_unit")


def test_a_state_that_does_not_cover_every_dof_is_rejected() -> None:
    """The raw arrays are indexed by articulation order, so a short one silently misaligns.

    A selection that reports one DOF still needs the full-length raw arrays, because index 1 means
    the second DOF of the articulation and not the second reported value.
    """
    for sensor in (_sensor(), _sensor(names=["spin"])):
        _assert_raises(ValueError, lambda: sensor.sample(0.0, _state([0.0], [0.0, 0.0])), "short positions")
        _assert_raises(ValueError, lambda: sensor.sample(0.0, _state([0.0, 0.0], [0.0])), "short velocities")
        _assert_raises(ValueError, lambda: sensor.sample(0.0, _state([0.0, 0.0], [0.0, 0.0], [0.0])), "short efforts")
        _assert_raises(ValueError, lambda: sensor.sample(0.0, _state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])), "long arrays")


# ----------------------------------------------------------------------------------------------
# Efforts.
# ----------------------------------------------------------------------------------------------


def test_a_backend_without_efforts_reports_zeros_and_says_so() -> None:
    """The shipped sensor substitutes zeros; the flag keeps that from reading as a measurement."""
    sensor = _sensor()
    reading = _drive(sensor, [_state([1.0, 1.0], [2.0, 2.0])] * 2)
    assert reading is not None
    assert reading.efforts_available is False
    _assert_close(reading.efforts, [0.0, 0.0])
    # Nothing else is affected.
    _assert_close(reading.velocities, [2.0 * CENTIMETRES, 2.0])


def test_efforts_are_unavailable_if_any_sample_in_the_window_lacks_them() -> None:
    """A window that straddles a backend change cannot report a mean of half a window."""
    sensor = _sensor(filter_width=2)
    with_efforts = _state([0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
    without = _state([0.0, 0.0], [0.0, 0.0])
    reading = _drive(sensor, [with_efforts, with_efforts, without, with_efforts])
    assert reading is not None
    assert reading.efforts_available is False
    # Samples older than the window do not decide it: they are only differenced against.
    reading = _drive(_sensor(filter_width=2), [without, without, with_efforts, with_efforts])
    assert reading is not None
    assert reading.efforts_available is True


# ----------------------------------------------------------------------------------------------
# Filtering.
# ----------------------------------------------------------------------------------------------


def test_a_position_comes_from_the_newest_sample() -> None:
    """A joint position is the pose, so averaging it would lag the robot."""
    sensor = _sensor(filter_width=2)
    states = [_state([100.0 * step, 0.0], [0.0, 0.0]) for step in range(4)]
    reading = _drive(sensor, states)
    assert reading is not None
    _assert_close([reading.positions[0]], [300.0 * CENTIMETRES])


def test_a_velocity_is_the_mean_of_the_filter_window() -> None:
    """The noisy quantities are averaged, at the cost of half a window of lag."""
    sensor = _sensor(filter_width=2)
    states = [_state([0.0, 0.0], [0.0, 10.0 * step]) for step in range(4)]
    reading = _drive(sensor, states)
    assert reading is not None
    # The window holds the two newest samples: 30 and 20.
    _assert_close([reading.velocities[1]], [25.0])


def test_an_effort_is_the_mean_of_the_filter_window() -> None:
    """Efforts are filtered the same way velocities are."""
    sensor = JointStateSensor([SPIN], meters_per_unit=1.0, filter_width=2)
    states = [_state([0.0], [0.0], [4.0 * step]) for step in range(4)]
    reading = _drive(sensor, states)
    assert reading is not None
    _assert_close([reading.efforts[0]], [10.0])


def test_an_unfiltered_sensor_reports_the_newest_sample_untouched() -> None:
    """A width of one averages nothing."""
    sensor = _sensor(filter_width=1)
    reading = _drive(sensor, [_state([0.0, 0.0], [0.0, 1.0]), _state([0.0, 0.0], [0.0, 7.0])])
    assert reading is not None
    _assert_close([reading.velocities[1]], [7.0])


# ----------------------------------------------------------------------------------------------
# Acceleration.
# ----------------------------------------------------------------------------------------------


def test_a_constant_velocity_reads_zero_acceleration() -> None:
    """Nothing is changing, so nothing is differenced into an acceleration."""
    sensor = _sensor(filter_width=3)
    reading = _drive(sensor, [_state([0.0, 0.0], [-5.0, 2.0])] * 6)
    assert reading is not None
    _assert_close(reading.accelerations, [0.0, 0.0])


def test_a_constant_acceleration_survives_filtering() -> None:
    """Averaging equal differences returns that difference, whatever the window width.

    The reported velocity lags by half a window, but the slope does not, which is what makes an
    acceleration checkable against a closed form while a velocity is not.
    """
    rate = 2.5
    for filter_width in (1, 2, 5):
        sensor = JointStateSensor([SPIN], meters_per_unit=1.0, filter_width=filter_width)
        states = [_state([0.0], [rate * step]) for step in range(2 * filter_width)]
        reading = _drive(sensor, states)
        assert reading is not None
        _assert_close(reading.accelerations, [rate], 1.0e-9)


def test_the_difference_spans_the_whole_window_not_the_last_step() -> None:
    """Each pair is a window apart, so a half-second step size is not assumed to be one second."""
    sensor = JointStateSensor([SPIN], meters_per_unit=1.0, filter_width=1)
    reading = _drive(sensor, [_state([0.0], [0.0]), _state([0.0], [4.0])], time_step=0.5)
    assert reading is not None
    _assert_close(reading.accelerations, [8.0])


def test_each_dof_is_differenced_on_its_own() -> None:
    """One DOF accelerating does not leak into another."""
    sensor = _sensor(filter_width=1)
    states = [_state([0.0, 0.0], [-100.0 * step, 0.0]) for step in range(2)]
    reading = _drive(sensor, states)
    assert reading is not None
    _assert_close(reading.accelerations, [-100.0 * CENTIMETRES, 0.0])


# ----------------------------------------------------------------------------------------------
# Timestamp handling.
# ----------------------------------------------------------------------------------------------


def test_a_reading_is_stamped_with_its_newest_sample() -> None:
    """Nothing else would notice the whole series shifted by a step."""
    sensor = _sensor(filter_width=2)
    for step, time in enumerate((0.5, 1.5, 2.5, 3.5)):
        sensor.sample(time, _state([0.0, 0.0], [0.0, 0.0]))
    reading = sensor.read()
    assert reading is not None
    assert reading.time == 3.5


def test_repeated_timestamps_do_not_divide_by_zero() -> None:
    """A zero interval contributes nothing rather than an infinity."""
    sensor = _sensor(filter_width=1)
    sensor.sample(1.0, _state([0.0, 0.0], [0.0, 0.0]))
    sensor.sample(1.0, _state([0.0, 0.0], [0.0, 5.0]))
    reading = sensor.read()
    assert reading is not None
    _assert_close(reading.accelerations, [0.0, 0.0])


def test_a_stalled_timestamp_does_not_bias_the_average_toward_zero() -> None:
    """Unusable differences are left out of the mean rather than counted as zero.

    A rewound final timestamp makes one of the two pairs unusable, so averaging over the whole
    window instead would halve the result.
    """
    rate = 3.0
    sensor = JointStateSensor([SPIN], meters_per_unit=1.0, filter_width=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for time in (0.0, 1.0, 2.0, 1.0):
            sensor.sample(time, _state([0.0], [rate * time]))
    reading = sensor.read()
    assert reading is not None
    _assert_close(reading.accelerations, [rate])


def test_a_rewound_timestamp_keeps_the_sample_and_warns() -> None:
    """Out-of-order time invalidates the difference, not the sample."""
    sensor = JointStateSensor([SPIN], meters_per_unit=1.0, filter_width=1)
    sensor.sample(1.0, _state([1.0], [0.0]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sensor.sample(0.0, _state([2.0], [7.0]))
    assert len(caught) == 1, f"expected one warning, got {len(caught)}"
    assert issubclass(caught[0].category, RuntimeWarning)

    assert sensor.sample_count == 2
    reading = sensor.read()
    assert reading is not None
    # The rewound sample is newest, so it is reported ...
    _assert_close(reading.positions, [2.0])
    _assert_close(reading.velocities, [7.0])
    # ... but nothing is differenced across the rewind.
    _assert_close(reading.accelerations, [0.0])


def test_increasing_timestamps_do_not_warn() -> None:
    """Increasing and repeated timestamps stay silent."""
    sensor = _sensor()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for time in (0.0, 1.0, 1.0, 2.0):
            sensor.sample(time, _state([0.0, 0.0], [0.0, 0.0]))
    assert len(caught) == 0, f"expected no warnings, got {[str(item.message) for item in caught]}"


def run_all() -> Any:
    """Run every test in this module and report the outcome.

    Returns:
        The number of failures.

    """
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    failures = []
    for name, test in tests:
        try:
            test()
        except Exception as error:  # noqa: BLE001 - report every failure kind.
            failures.append((name, error))
            print(f"  FAIL {name}: {error}")
    passed = len(tests) - len(failures)
    if failures:
        # A distinct line, so the example's stdout assertion cannot match a partial pass.
        print(f"Self-test: {passed}/{len(tests)} joint_state.py unit tests passed, {len(failures)} failed.")
    else:
        print(f"Self-test: all {len(tests)} joint_state.py unit tests passed.")
    return len(failures)


if __name__ == "__main__":
    raise SystemExit(1 if run_all() else 0)
