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

"""Tests for zmq_server.teleop.TeleopKeys.

The case that matters is keyboard auto-repeat: X11 delivers a held key as repeated
release+press pairs, so a handler that acts on releases immediately sees the key as up for most
of every repeat cycle and publishes a stop command on the majority of sends. These tests replay
that event pattern with a synthetic clock -- no GPU, no window, no wall-clock sleeps.

Run:  PYTHONPATH=src python -m unittest zmq_server.tests.test_teleop -v
"""

from __future__ import annotations

import unittest

from zmq_server.teleop import KEY_HOLD_GRACE_S, TeleopKeys

_W = ord("W")
_S = ord("S")

# A typical X11 repeat: ~500 ms before the first repeat, then ~30 Hz.
_REPEAT_DELAY_S = 0.5
_REPEAT_INTERVAL_S = 1.0 / 30.0


def _hold_with_auto_repeat(keys, code, start, end, sample_dt=0.001):
    """Replay a physically-held key from ``start`` to ``end``, yielding (t, held) at each sample.

    The event stream is the real one: an initial press, then after the repeat delay a
    release immediately followed by a press, once per repeat interval.
    """
    keys.press(code, start)
    next_repeat = start + _REPEAT_DELAY_S
    t = start
    while t < end:
        t += sample_dt
        while next_repeat <= t:
            keys.release(code, next_repeat)
            keys.press(code, next_repeat)
            next_repeat += _REPEAT_INTERVAL_S
        yield t, keys.is_held(code, t)


class TestTeleopKeys(unittest.TestCase):
    def test_press_makes_key_held(self):
        keys = TeleopKeys()
        keys.press(_W, 10.0)
        self.assertTrue(keys.is_held(_W, 10.0))
        self.assertTrue(keys.is_held(_W, 10.0 + KEY_HOLD_GRACE_S * 10))

    def test_untouched_key_is_not_held(self):
        keys = TeleopKeys()
        keys.press(_W, 10.0)
        self.assertFalse(keys.is_held(_S, 10.0))

    def test_held_through_auto_repeat(self):
        """A key held for 3 s must read as held at every sample in between.

        This is the regression: acting on auto-repeat releases made the key read as up for most
        of each repeat cycle, so the large majority of published commands were a stop.
        """
        keys = TeleopKeys()
        for t, held in _hold_with_auto_repeat(keys, _W, 0.0, 3.0):
            self.assertTrue(held, f"key read as released at t={t:.3f}s")

    def test_held_across_the_initial_repeat_delay(self):
        """The ~500 ms before the first repeat carries no events at all; the key must stay held.

        Guards against 'fix' by press-recency alone, which lapses partway through this gap and
        stalls the robot at the start of every press.
        """
        keys = TeleopKeys()
        keys.press(_W, 0.0)
        for t in (0.05, 0.12, 0.3, _REPEAT_DELAY_S - 0.01):
            self.assertTrue(keys.is_held(_W, t), f"key lapsed during the repeat delay at t={t}s")

    def test_real_release_stops_within_the_grace_window(self):
        keys = TeleopKeys()
        keys.press(_W, 1.0)
        keys.release(_W, 2.0)
        self.assertTrue(keys.is_held(_W, 2.0 + KEY_HOLD_GRACE_S * 0.5))
        self.assertFalse(keys.is_held(_W, 2.0 + KEY_HOLD_GRACE_S * 1.5))

    def test_release_after_a_long_hold_still_stops(self):
        """Release at the end of an auto-repeat run must not be cancelled by a stale press."""
        keys = TeleopKeys()
        last = None
        for t, _ in _hold_with_auto_repeat(keys, _W, 0.0, 2.0):
            last = t
        keys.release(_W, last)
        self.assertFalse(keys.is_held(_W, last + KEY_HOLD_GRACE_S * 1.5))

    def test_grace_covers_a_repeat_interval(self):
        """Must exceed one repeat interval, or repeats cannot sustain a key."""
        self.assertGreater(KEY_HOLD_GRACE_S, _REPEAT_INTERVAL_S)

    def test_grace_does_not_cover_the_repeat_delay(self):
        """Must stay well under the repeat delay, or letting go coasts absurdly far."""
        self.assertLess(KEY_HOLD_GRACE_S, _REPEAT_DELAY_S / 2)

    def test_opposing_keys_are_independent(self):
        keys = TeleopKeys()
        keys.press(_W, 5.0)
        keys.press(_S, 5.0)
        keys.release(_W, 5.0)
        self.assertFalse(keys.is_held(_W, 5.0 + KEY_HOLD_GRACE_S * 1.5))
        self.assertTrue(keys.is_held(_S, 5.0 + KEY_HOLD_GRACE_S * 1.5))

    def test_release_of_an_untracked_key_is_ignored(self):
        keys = TeleopKeys()
        keys.release(_W, 1.0)  # must not resurrect or crash
        self.assertFalse(keys.is_held(_W, 1.0))

    def test_clear_forgets_everything(self):
        keys = TeleopKeys()
        keys.press(_W, 1.0)
        keys.clear()
        self.assertFalse(keys.is_held(_W, 1.0))


if __name__ == "__main__":
    unittest.main()
