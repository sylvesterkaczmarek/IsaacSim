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

"""Where a recorded joint response stops being physically meaningful."""

from __future__ import annotations

import numpy as np
import omni.kit.test
from isaacsim.robot_setup.gain_tuner.divergence import (
    DIVERGENCE_HEADROOM,
    clip_series_to_valid,
    divergence_limit,
    peak_bound,
    travel_bound,
    valid_prefix_length,
)


class TestTravelBound(omni.kit.test.AsyncTestCase):
    """Divergence bound derived from a joint's own travel limits."""

    async def test_scales_widest_reach_by_headroom(self) -> None:
        """The bound is a multiple of the furthest the joint can travel."""
        self.assertAlmostEqual(travel_bound(-360.0, 360.0, headroom=4.0), 1440.0)

    async def test_asymmetric_limits_use_the_wider_side(self) -> None:
        """A joint that travels further one way is bounded by that side."""
        self.assertAlmostEqual(travel_bound(-10.0, 90.0, headroom=2.0), 180.0)

    async def test_applies_the_default_headroom(self) -> None:
        """Callers that do not pass a headroom get the module default."""
        self.assertAlmostEqual(travel_bound(-90.0, 90.0), 90.0 * DIVERGENCE_HEADROOM)

    async def test_unlimited_joint_yields_none(self) -> None:
        """A continuous revolute joint's limits do not bound anything."""
        self.assertIsNone(travel_bound(float("-inf"), float("inf")))
        self.assertIsNone(travel_bound(-np.nan, 1.0))
        # The float sentinels USD uses for "unlimited" are not a real range either.
        self.assertIsNone(travel_bound(-3.4e38, 3.4e38))

    async def test_zero_width_and_unusable_limits_yield_none(self) -> None:
        """A locked joint or a non-numeric limit applies no bound."""
        self.assertIsNone(travel_bound(0.0, 0.0))
        self.assertIsNone(travel_bound(None, 1.0))
        self.assertIsNone(travel_bound("lower", 1.0))


class TestPeakBound(omni.kit.test.AsyncTestCase):
    """Fallback bound calibrated from a commanded trajectory."""

    async def test_scales_commanded_peak_by_headroom(self) -> None:
        """The bound is a multiple of the largest commanded magnitude."""
        self.assertAlmostEqual(peak_bound(np.array([-90.0, 0.0, 45.0]), headroom=4.0), 360.0)

    async def test_non_finite_reference_samples_are_ignored(self) -> None:
        """A NaN in the command does not become an infinite bound."""
        self.assertAlmostEqual(peak_bound(np.array([1.0, np.nan, np.inf]), headroom=3.0), 3.0)

    async def test_no_usable_reference_yields_none(self) -> None:
        """An absent, empty, flat-zero, or non-finite reference applies no bound."""
        self.assertIsNone(peak_bound(None))
        self.assertIsNone(peak_bound(np.array([])))
        self.assertIsNone(peak_bound(np.array([0.0, 0.0])))
        self.assertIsNone(peak_bound(np.array([np.nan])))


class TestDivergenceLimit(omni.kit.test.AsyncTestCase):
    """Composition of the two bound sources, limits first."""

    async def test_prefers_limits_over_the_command(self) -> None:
        """A joint's travel bounds it even when the run commands a small part of it.

        This is what keeps a legitimate tracking failure on the chart: a joint
        commanded a few degrees that sags far further is still within its own range
        of motion, so it is recorded rather than cut and blamed on divergence.
        """
        limit = divergence_limit(-180.0, 180.0, np.array([0.0, 5.0]), headroom=4.0)
        self.assertAlmostEqual(limit, 720.0)

    async def test_falls_back_to_the_command_when_unlimited(self) -> None:
        """A continuous joint has only its commanded trajectory to calibrate on."""
        limit = divergence_limit(float("-inf"), float("inf"), np.array([0.0, 90.0]), headroom=2.0)
        self.assertAlmostEqual(limit, 180.0)

    async def test_falls_back_when_limits_are_unknown(self) -> None:
        """An articulation that does not report limits still bounds what it can."""
        self.assertAlmostEqual(divergence_limit(None, None, np.array([10.0]), headroom=2.0), 20.0)

    async def test_no_source_yields_none(self) -> None:
        """With neither limits nor a command, only non-finite samples are knowable."""
        self.assertIsNone(divergence_limit(None, None, None))
        self.assertIsNone(divergence_limit(float("inf"), float("inf"), np.array([0.0])))


class TestValidPrefixLength(omni.kit.test.AsyncTestCase):
    """Leading-usable-run detection across parallel series."""

    async def test_all_finite_returns_full_length(self) -> None:
        """A clean series is not truncated."""
        xs = np.array([0.0, 0.1, 0.2])
        ys = np.array([1.0, 2.0, 3.0])
        self.assertEqual(valid_prefix_length(xs, ys), 3)

    async def test_truncates_at_first_nan(self) -> None:
        """A diverging joint is cut at the sample where it went non-finite."""
        ys = np.array([1.0, 2.0, np.nan, np.nan])
        self.assertEqual(valid_prefix_length(y_series=ys), 2)

    async def test_truncates_at_first_inf(self) -> None:
        """Infinite samples are treated the same as NaN."""
        ys = np.array([1.0, np.inf, 3.0])
        self.assertEqual(valid_prefix_length(y_series=ys), 1)

    async def test_uses_earliest_break_across_arrays(self) -> None:
        """The prefix ends at whichever array goes non-finite first."""
        xs = np.array([0.0, 0.1, np.nan, 0.3])
        ys = np.array([1.0, np.nan, 3.0, 4.0])
        self.assertEqual(valid_prefix_length(xs, ys), 1)

    async def test_leading_nan_yields_empty_prefix(self) -> None:
        """A series that is non-finite from the first sample has no valid prefix."""
        self.assertEqual(valid_prefix_length(y_series=np.array([np.nan, 1.0])), 0)

    async def test_empty_and_none_inputs(self) -> None:
        """Empty arrays and None entries do not raise."""
        self.assertEqual(valid_prefix_length(), 0)
        self.assertEqual(valid_prefix_length(None, None), 0)
        self.assertEqual(valid_prefix_length(y_series=np.array([])), 0)
        self.assertEqual(valid_prefix_length(None, np.array([1.0, 2.0])), 2)

    async def test_shorter_array_bounds_the_prefix(self) -> None:
        """Mismatched lengths clamp to the shortest array."""
        xs = np.array([0.0, 0.1])
        ys = np.array([1.0, 2.0, 3.0])
        self.assertEqual(valid_prefix_length(xs, ys), 2)

    async def test_truncates_at_implausible_magnitude(self) -> None:
        """A finite but wound-up sample is cut, since it would flatten the axis."""
        ys = np.array([0.0, 1.0, 2.0, 1.4e19])
        self.assertEqual(valid_prefix_length(y_series=ys, limit=100.0), 3)

    async def test_magnitude_limit_applies_to_both_signs(self) -> None:
        """Divergence in the negative direction is cut the same way."""
        ys = np.array([0.0, -1.0, -5000.0])
        self.assertEqual(valid_prefix_length(y_series=ys, limit=100.0), 2)

    async def test_magnitude_limit_is_inclusive(self) -> None:
        """A sample exactly at the limit is still valid data."""
        ys = np.array([0.0, 100.0])
        self.assertEqual(valid_prefix_length(y_series=ys, limit=100.0), 2)

    async def test_magnitude_limit_does_not_bound_x(self) -> None:
        """Sample times are unrelated to the plotted quantity's scale."""
        xs = np.array([0.0, 5.0, 30.0])
        ys = np.array([0.1, 0.2, 0.3])
        self.assertEqual(valid_prefix_length(xs, ys, limit=1.0), 3)

    async def test_no_limit_keeps_finite_divergence(self) -> None:
        """Without a bound to calibrate against, finite samples are all kept."""
        ys = np.array([0.0, 1.0, 1.4e19])
        self.assertEqual(valid_prefix_length(y_series=ys), 3)


class TestClipSeriesToValid(omni.kit.test.AsyncTestCase):
    """Per-series clipping of parallel (x, y) chart data."""

    async def test_clips_each_series_independently(self) -> None:
        """One diverging series is clipped without shortening the healthy one."""
        x_data = [np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.1, 0.2])]
        y_data = [np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 3.0])]
        xs, ys = clip_series_to_valid(x_data, y_data)
        self.assertEqual(ys[0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(ys[1].tolist(), [1.0])
        self.assertEqual(xs[0].tolist(), [0.0, 0.1, 0.2])
        self.assertEqual(xs[1].tolist(), [0.0])

    async def test_x_and_y_stay_aligned(self) -> None:
        """Clipped x and y keep equal length so time indexing stays valid."""
        xs, ys = clip_series_to_valid(
            [np.array([0.0, 0.1, 0.2, 0.3])],
            [np.array([1.0, 2.0, np.inf, 4.0])],
        )
        self.assertEqual(len(xs[0]), len(ys[0]))
        self.assertEqual(len(ys[0]), 2)

    async def test_missing_x_series_still_clips_y(self) -> None:
        """A y series with no matching x entry is clipped and produces no x entry."""
        xs, ys = clip_series_to_valid([], [np.array([1.0, np.nan])])
        self.assertEqual(xs, [])
        self.assertEqual(ys[0].tolist(), [1.0])

    async def test_each_series_uses_its_own_bound(self) -> None:
        """One joint's range cannot govern another's.

        A narrow-travel joint plotted alongside a wide-travel one keeps its own
        bound, so which joints the user has selected cannot change how any of them
        is clipped.
        """
        x = np.arange(3, dtype=float)
        narrow = np.array([0.0, 5.0, 400.0])
        wide = np.array([0.0, 5.0, 400.0])
        _, ys = clip_series_to_valid([x, x], [narrow, wide], limits=[40.0, 4000.0])
        self.assertEqual(ys[0].tolist(), [0.0, 5.0])
        self.assertEqual(ys[1].tolist(), [0.0, 5.0, 400.0])

    async def test_absent_and_short_limits_clip_only_non_finite(self) -> None:
        """A missing bound is not guessed at; only unusable samples are cut."""
        x = np.arange(3, dtype=float)
        y = np.array([0.0, 1e19, np.nan])
        _, ys = clip_series_to_valid([x], [y], limits=[None])
        self.assertEqual(ys[0].tolist(), [0.0, 1e19])
        _, ys = clip_series_to_valid([x, x], [y, y], limits=[1.0])
        self.assertEqual(ys[0].tolist(), [0.0])
        self.assertEqual(ys[1].tolist(), [0.0, 1e19])
