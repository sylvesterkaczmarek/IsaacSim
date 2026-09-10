# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive a joint state reading from the raw degree-of-freedom state of an articulation.

Callers push one `ArticulationState` per step with `sample()` and pull a filtered
`JointStateReading` with `read()`, so the state can come from a physics engine, a recording, or a
closed-form trajectory.

Everything here is per-DOF scalars, so the module works in plain Python floats and depends on
nothing outside the standard library. A degree of freedom is either a rotation or a translation,
and that distinction is the whole of the arithmetic: a rotation reads radians whatever the stage's
linear unit, while a translation reads stage units and has to be scaled to meters.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

#: A degree of freedom about an axis. Positions are radians and efforts are newton-metres.
ROTATION = "rotation"
#: A degree of freedom along an axis. Positions are metres and efforts are newtons.
TRANSLATION = "translation"

#: The ``uint8`` encoding the shipped ``JointStateSensorReading.dof_types`` uses.
DOF_TYPE_CODES = {0: ROTATION, 1: TRANSLATION}


def dof_type_from_code(code: int) -> str:
    """Convert the shipped sensor's numeric DOF type to this module's name.

    ``isaacsim.sensors.experimental.physics.JointStateSensorReading`` reports ``dof_types`` as
    ``uint8``. USD reports the same distinction as the strings this module uses, so a caller
    reading from the C++ sensor converts here and a caller reading from USD does not.

    Args:
        code: Numeric DOF type, ``0`` for a rotation and ``1`` for a translation. The reading
            carries these as ``uint8``, so a NumPy integer is as acceptable as a Python one.

    Returns:
        `ROTATION` or `TRANSLATION`.

    Raises:
        ValueError: If the code is not an integer, or not one the shipped sensor defines.

    """
    # Converting first would quietly turn 0.5 into a rotation and True into a translation, so a
    # code that is not an integer is rejected rather than rounded onto a valid DOF type. Testing
    # against `numbers.Integral` accepts the reading's NumPy integers without importing NumPy.
    if isinstance(code, bool) or not isinstance(code, numbers.Integral):
        raise ValueError(f"DOF type code {code!r} is not an integer.")
    try:
        return DOF_TYPE_CODES[int(code)]
    except KeyError:
        raise ValueError(f"Unknown DOF type code {code}, expected one of {sorted(DOF_TYPE_CODES)}.") from None


@dataclass(frozen=True)
class Dof:
    """One degree of freedom of an articulation, in the order the engine reports it.

    Args:
        name: DOF name, unique within the articulation for name-based selection to work.
        type: `ROTATION` or `TRANSLATION`.

    Raises:
        ValueError: If the type is neither.

    """

    name: str
    type: str

    def __post_init__(self) -> None:
        """Validate the DOF type."""
        if self.type not in (ROTATION, TRANSLATION):
            raise ValueError(f"DOF {self.name!r} has type {self.type!r}, expected {ROTATION!r} or {TRANSLATION!r}.")


@dataclass
class ArticulationState:
    """Raw per-DOF state of an articulation, in stage units and articulation order.

    Args:
        positions: DOF positions, radians or stage length units.
        velocities: DOF velocities, per second.
        efforts: Measured joint efforts, or ``None`` when the backend does not report them. The
            shipped sensor substitutes zeros in that case; passing ``None`` keeps the distinction
            so `JointStateReading.efforts_available` can carry it.

    """

    positions: Sequence[float]
    velocities: Sequence[float]
    efforts: Sequence[float] | None = None


@dataclass
class JointStateReading:
    """One filtered reading, in SI units, for the DOFs the sensor was asked to report.

    The fields are not all from the same instant. `positions` is the newest sample, because a joint
    position is the pose and averaging it would lag the robot. `velocities` and `efforts` are means
    over the filter window, so they lag the newest sample by ``(filter_width - 1) / 2`` steps.
    """

    #: Simulation time of the newest sample, in seconds.
    time: float
    #: DOF names, in the reported order.
    names: tuple[str, ...]
    #: Per-DOF `ROTATION` or `TRANSLATION`.
    types: tuple[str, ...]
    #: Positions, rad or m.
    positions: tuple[float, ...]
    #: Velocities, rad/s or m/s.
    velocities: tuple[float, ...]
    #: Measured joint efforts, N*m or N. Zero throughout when `efforts_available` is false.
    efforts: tuple[float, ...]
    #: Velocities differenced across the filter window, rad/s^2 or m/s^2.
    accelerations: tuple[float, ...]
    #: Whether every sample in the window carried measured efforts.
    efforts_available: bool


@dataclass
class _Sample:
    """One converted sample, buffered for the rolling average."""

    time: float
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    efforts: tuple[float, ...]
    efforts_available: bool


def _check_scale(name: str, value: float) -> float:
    """Validate a unit scale.

    Args:
        name: Argument name, for the error message.
        value: Scale to check.

    Returns:
        The scale as a float.

    Raises:
        ValueError: If the scale is not finite and positive.

    """
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"`{name}` must be finite and positive, got {value}.")
    return scale


def _mean(window: Sequence[Sequence[float]], index: int) -> float:
    """Average one DOF's value over a window of samples.

    Args:
        window: Per-sample value sequences.
        index: DOF index to average.

    Returns:
        The mean.

    """
    return sum(entry[index] for entry in window) / len(window)


class JointStateSensor:
    """Reports the state of an articulation's degrees of freedom in SI units.

    ``2 * filter_width`` samples are buffered, so a reading can average a window of width
    ``filter_width`` and difference it against the window before it.

    Args:
        dofs: Every DOF of the articulation, in the order the raw state arrays are indexed.
        meters_per_unit: Stage metres per linear unit, the ``stage_meters_per_unit`` the shipped
            sensor reports. Required, because a translation DOF is meaningless without it.
        kilograms_per_unit: Stage kilograms per mass unit. It only reaches the efforts.
        names: DOFs to report, in the order to report them. ``None`` reports every DOF in
            articulation order.
        filter_width: Rolling average window applied to the velocities and efforts.

    Raises:
        ValueError: If `dofs` is empty, a unit scale is not positive, `filter_width` is less than
            one, or a selected name is unknown or names more than one DOF.

    """

    def __init__(
        self,
        dofs: Sequence[Dof],
        *,
        meters_per_unit: float,
        kilograms_per_unit: float = 1.0,
        names: Sequence[str] | None = None,
        filter_width: int = 1,
    ) -> None:
        if len(dofs) == 0:
            raise ValueError("An articulation with no DOFs has no joint state to report.")
        if filter_width < 1:
            raise ValueError(f"`filter_width` must be at least 1, got {filter_width}.")
        meters_per_unit = _check_scale("meters_per_unit", meters_per_unit)
        kilograms_per_unit = _check_scale("kilograms_per_unit", kilograms_per_unit)

        self._dof_count = len(dofs)
        self._indices = tuple(range(self._dof_count) if names is None else self._resolve(dofs, names))
        self._names = tuple(dofs[index].name for index in self._indices)
        self._types = tuple(dofs[index].type for index in self._indices)
        # A rotation is radians in every stage, so only a translation picks up the linear unit.
        self._position_scales = tuple(1.0 if dof_type == ROTATION else meters_per_unit for dof_type in self._types)
        # An effort is a torque on a rotation DOF and a force on a translation DOF: mass times
        # length squared, or mass times length, over time squared. So the two differ by a power of
        # length, and both carry the mass unit that neither a position nor a velocity sees.
        self._effort_scales = tuple(
            kilograms_per_unit * meters_per_unit * (meters_per_unit if dof_type == ROTATION else 1.0)
            for dof_type in self._types
        )
        self._filter_width = filter_width
        self._samples: deque[_Sample] = deque(maxlen=2 * filter_width)

    @staticmethod
    def _resolve(dofs: Sequence[Dof], names: Sequence[str]) -> list[int]:
        """Resolve selected DOF names to indices into the articulation's DOF list.

        Args:
            dofs: Every DOF of the articulation.
            names: Names to select, in the order to report them.

        Returns:
            One index per selected name.

        Raises:
            ValueError: If a name matches no DOF or more than one.

        """
        indices = []
        for name in names:
            matches = [index for index, dof in enumerate(dofs) if dof.name == name]
            if len(matches) != 1:
                known = ", ".join(sorted({dof.name for dof in dofs}))
                raise ValueError(f"{name!r} names {len(matches)} DOFs, expected exactly 1. Known DOFs: {known}.")
            indices.append(matches[0])
        return indices

    @property
    def names(self) -> tuple[str, ...]:
        """DOF names in the order readings report them."""
        return self._names

    @property
    def types(self) -> tuple[str, ...]:
        """Per-DOF `ROTATION` or `TRANSLATION`, in the order readings report them."""
        return self._types

    @property
    def sample_count(self) -> int:
        """Number of samples currently buffered."""
        return len(self._samples)

    def reset(self) -> None:
        """Discard every buffered sample.

        Call this when the articulation state becomes discontinuous, such as after the DOFs are
        written directly or the simulation restarts, so stale samples cannot contaminate the finite
        difference.
        """
        self._samples.clear()

    def sample(self, time: float, state: ArticulationState) -> None:
        """Select, convert, and buffer one raw articulation state.

        Every sample is buffered whatever its timestamp; `read()` leaves unusable differences out
        of the accelerations rather than dividing by a zero or negative interval. A time that moves
        backwards warns, since it means a discontinuity that `reset()` should have cleared.

        Args:
            time: Simulation time of the state, in seconds. Must increase between calls for the
                finite difference to be meaningful.
            state: Raw per-DOF state, in stage units and articulation order.

        Raises:
            ValueError: If an array does not hold one value per DOF of the articulation.

        """
        self._check_length("positions", state.positions)
        self._check_length("velocities", state.velocities)
        if state.efforts is not None:
            self._check_length("efforts", state.efforts)

        if self._samples and time < self._samples[-1].time:
            warnings.warn(
                f"Time moved backwards, from {self._samples[-1].time} to {time}. The sample is "
                "kept, but no acceleration is differenced across the rewind. Call reset() when "
                "the articulation state becomes discontinuous.",
                RuntimeWarning,
                stacklevel=2,
            )

        efforts = state.efforts
        self._samples.append(
            _Sample(
                time=float(time),
                positions=tuple(
                    float(state.positions[index]) * scale for index, scale in zip(self._indices, self._position_scales)
                ),
                velocities=tuple(
                    float(state.velocities[index]) * scale for index, scale in zip(self._indices, self._position_scales)
                ),
                # Zeros when unavailable, matching the shipped sensor. The flag keeps a caller from
                # having to read that zero as a measurement.
                efforts=tuple(
                    0.0 if efforts is None else float(efforts[index]) * scale
                    for index, scale in zip(self._indices, self._effort_scales)
                ),
                efforts_available=efforts is not None,
            )
        )

    def _check_length(self, name: str, values: Sequence[float]) -> None:
        """Check that a raw state array holds one value per DOF.

        Args:
            name: Field name, for the error message.
            values: Array to check.

        Raises:
            ValueError: If the length does not match the articulation's DOF count.

        """
        if len(values) != self._dof_count:
            raise ValueError(f"`{name}` holds {len(values)} values, expected {self._dof_count}, one per DOF.")

    def read(self) -> JointStateReading | None:
        """Produce a filtered reading from the buffered samples.

        Returns:
            The reading, or ``None`` until ``2 * filter_width`` samples have been buffered. The
            shipped sensor spells the same state as ``is_valid = False`` on a reading that carries
            no data.

        """
        if len(self._samples) < 2 * self._filter_width:
            return None

        # Index newest first, so the window indices match the difference below.
        def raw(index: int) -> _Sample:
            return self._samples[len(self._samples) - 1 - index]

        newest = raw(0)
        window = [raw(index) for index in range(self._filter_width)]
        velocity_window = [entry.velocities for entry in window]
        effort_window = [entry.efforts for entry in window]

        # Each pair is one filter window apart, so a difference spans the window rather than a
        # single step, and a pair whose interval is zero or negative is left out entirely rather
        # than dividing by it.
        pairs = [(raw(index), raw(index + self._filter_width)) for index in range(self._filter_width)]
        usable = [(newer, older) for newer, older in pairs if newer.time > older.time]

        accelerations = []
        for dof in range(len(self._indices)):
            total = sum(
                (newer.velocities[dof] - older.velocities[dof]) / (newer.time - older.time) for newer, older in usable
            )
            # Average over the usable differences, not the whole window, or a repeated or rewound
            # timestamp biases the result toward zero.
            accelerations.append(total / len(usable) if usable else 0.0)

        return JointStateReading(
            time=newest.time,
            names=self._names,
            types=self._types,
            positions=newest.positions,
            velocities=tuple(_mean(velocity_window, dof) for dof in range(len(self._indices))),
            efforts=tuple(_mean(effort_window, dof) for dof in range(len(self._indices))),
            accelerations=tuple(accelerations),
            efforts_available=all(entry.efforts_available for entry in window),
        )
