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

"""Where a recorded joint response stops being physically meaningful.

A joint that loses stability during a gains test stops producing usable samples in
one of two ways: it records NaN or infinite values outright, or it winds up to a
magnitude no real joint reaches -- a diverging UR10 revolute joint gets to ``1e19``.
Deciding where that happens is a question about the joint, not about the chart, so
it lives here next to the stability verdicts in :mod:`stress_test`; the presentation
extension only consumes the answer.

The bound comes from the joint's own travel limits, which makes it independent of
the test mode, of the commanded amplitude, and of which joints the user happens to
have selected for plotting.
"""

from __future__ import annotations

import numpy as np

DIVERGENCE_HEADROOM = 4.0
"""Multiple of a joint's travel past which a response counts as diverged.

Generous enough to keep real overshoot and settling on record -- several times the
joint's own range of motion -- while still catching a joint that has wound up by
orders of magnitude.
"""

_UNBOUNDED = 1e30
"""Magnitude at or past which a joint limit is treated as absent.

Continuous revolute joints carry limits reported as infinite or as the float
sentinels USD uses for "unlimited", neither of which bounds anything useful.
"""

_NEGLIGIBLE = 1e-9
"""Magnitude at or below which a reach or peak is treated as zero.

A joint whose limits collapse to a point, or a reference that never moves, carries
no travel to scale a bound from.
"""


def travel_bound(lower: float, upper: float, headroom: float = DIVERGENCE_HEADROOM) -> float | None:
    """Return the largest magnitude a response can reach and still be meaningful.

    Args:
        lower: The joint's lower position limit, in the units the response is
            recorded in.
        upper: The joint's upper position limit, in the same units.
        headroom: Multiple of the joint's travel allowed before divergence.

    Returns:
        The magnitude bound, or ``None`` when the limits do not bound anything --
        absent, non-finite, effectively unlimited, or a zero-width range.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner.divergence import travel_bound

        >>> travel_bound(-360.0, 360.0, headroom=4.0)
        1440.0
        >>> travel_bound(float("-inf"), float("inf")) is None
        True
    """
    try:
        lo, hi = float(lower), float(upper)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    reach = max(abs(lo), abs(hi))
    if reach >= _UNBOUNDED or reach <= _NEGLIGIBLE:
        return None
    return reach * headroom


def peak_bound(series: np.ndarray | None, headroom: float = DIVERGENCE_HEADROOM) -> float | None:
    """Return a magnitude bound calibrated from a reference series.

    The fallback for a joint whose limits do not bound anything: a commanded
    trajectory is bounded by construction, so its peak stands in for the joint's
    travel.  Weaker than :func:`travel_bound`, because a run that commands only a
    small part of the joint's range yields a correspondingly small bound.

    Args:
        series: The reference trajectory, or ``None``.
        headroom: Multiple of the reference peak allowed before divergence.

    Returns:
        The magnitude bound, or ``None`` when the reference carries no usable
        magnitude -- empty, wholly non-finite, or all zero.

    Example:

    .. code-block:: python

        >>> import numpy as np
        >>> from isaacsim.robot_setup.gain_tuner.divergence import peak_bound

        >>> peak_bound(np.array([-90.0, 0.0, 90.0]), headroom=4.0)
        360.0
    """
    if series is None or len(series) == 0:
        return None
    finite = np.asarray(series, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return None
    peak = float(np.max(np.abs(finite)))
    if peak <= _NEGLIGIBLE:
        return None
    return peak * headroom


def divergence_limit(
    lower: float | None = None,
    upper: float | None = None,
    reference: np.ndarray | None = None,
    headroom: float = DIVERGENCE_HEADROOM,
) -> float | None:
    """Return the divergence bound for one joint, preferring its travel limits.

    Args:
        lower: The joint's lower position limit, or ``None`` when unknown.
        upper: The joint's upper position limit, or ``None`` when unknown.
        reference: Commanded trajectory to fall back on when the limits do not
            bound anything.
        headroom: Multiple of the reference travel allowed before divergence.

    Returns:
        The magnitude bound, or ``None`` when neither source bounds anything, in
        which case only non-finite samples can be identified.  A missing bound is
        not replaced with a guess, which would risk discarding valid data.
    """
    if lower is not None and upper is not None:
        bound = travel_bound(lower, upper, headroom=headroom)
        if bound is not None:
            return bound
    return peak_bound(reference, headroom=headroom)


def valid_prefix_length(
    x_series: np.ndarray | None = None,
    y_series: np.ndarray | None = None,
    *,
    limit: float | None = None,
) -> int:
    """Return the length of the leading run of physically meaningful samples.

    Both failure modes have to be cut, because a chart's axis range is shared: one
    NaN blanks every series, and one astronomical sample stretches the range so far
    that the real motion compresses into less than a pixel.  Either way the chart
    looks empty.

    Only the leading run is measured, so a joint that recovers after diverging keeps
    nothing past its first unusable sample.

    ``limit`` bounds ``y_series`` only.  The x series carries sample times, whose
    scale is unrelated to the plotted quantity, so it is checked for finiteness
    alone -- hence keyword-only, so a transposed call cannot silently bound the
    timestamps instead.

    Args:
        x_series: Sample times, or ``None`` when the caller has no x array.
        y_series: Recorded values, or ``None``.
        limit: Largest magnitude ``y_series`` may reach, as returned by
            :func:`divergence_limit`; ``None`` applies no bound.

    Returns:
        The number of leading samples usable in every supplied array.

    Example:

    .. code-block:: python

        >>> import numpy as np
        >>> from isaacsim.robot_setup.gain_tuner.divergence import valid_prefix_length

        >>> valid_prefix_length(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, np.nan]))
        2
        >>> valid_prefix_length(y_series=np.array([1.0, 2.0, 1e19]), limit=100.0)
        2
    """
    arrays = [a for a in (x_series, y_series) if a is not None]
    if not arrays:
        return 0
    length = min(len(a) for a in arrays)
    if length == 0:
        return 0
    valid = np.ones(length, dtype=bool)
    for a in arrays:
        valid &= np.isfinite(np.asarray(a, dtype=float)[:length])
    if limit is not None and y_series is not None:
        y = np.asarray(y_series, dtype=float)[:length]
        # Non-finite samples already read as invalid above; suppress the compare
        # warning they raise here rather than letting it reach the log.
        with np.errstate(invalid="ignore"):
            valid &= np.abs(y) <= limit
    if bool(valid.all()):
        return length
    return int(np.argmin(valid))


def clip_series_to_valid(
    x_data: list[np.ndarray],
    y_data: list[np.ndarray],
    *,
    limits: list[float | None] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Truncate each parallel ``(x, y)`` series pair where its data stops being usable.

    Args:
        x_data: Per-series x arrays; may be shorter than ``y_data``.
        y_data: Per-series y arrays.
        limits: Per-series magnitude bounds aligned with ``y_data``, as returned by
            :func:`divergence_limit`.  A ``None`` entry, or a list shorter than
            ``y_data``, cuts only non-finite samples in that series.  Bounds are
            per-series so that one joint's range cannot govern another's, and so
            that adding or removing a joint from the chart cannot change how the
            others are clipped.

    Returns:
        Tuple of the clipped ``(x_data, y_data)`` lists.

    Example:

    .. code-block:: python

        >>> import numpy as np
        >>> from isaacsim.robot_setup.gain_tuner.divergence import clip_series_to_valid

        >>> xs, ys = clip_series_to_valid([np.array([0.0, 1.0])], [np.array([3.0, np.inf])])
        >>> ys[0].tolist()
        [3.0]
    """
    clipped_x: list[np.ndarray] = []
    clipped_y: list[np.ndarray] = []
    for i, y in enumerate(y_data):
        x = x_data[i] if i < len(x_data) else None
        limit = limits[i] if (limits is not None and i < len(limits)) else None
        count = valid_prefix_length(x, y, limit=limit)
        if x is not None:
            clipped_x.append(x[:count])
        clipped_y.append(y[:count])
    return clipped_x, clipped_y
