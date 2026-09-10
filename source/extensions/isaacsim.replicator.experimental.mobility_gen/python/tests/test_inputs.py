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

"""Test KeyboardButton event consumption for tracked keyboard input.

These tests verify that press, repeat, and release events update only the
configured key, while unrelated keys and CHAR events continue through Kit's
input pipeline.
"""

import asyncio

import carb
import omni.kit.test
from isaacsim.replicator.experimental.mobility_gen.impl.inputs import _KEY_HOLD_GRACE_S, KeyboardButton


class _FakeKeyboardEvent:
    """Keyboard event stand-in exposing the fields read by KeyboardButton.

    Args:
        input_key: Keyboard input key to expose on the fake event.
        event_type: Keyboard event type to expose on the fake event.
    """

    # Lightweight stand-in for `carb.input.KeyboardEvent`. The real type cannot be
    # constructed from Python, but `KeyboardButton._event_callback` only reads
    # `.input` and `.type`, so duck typing is sufficient here.
    def __init__(self, input_key: carb.input.KeyboardInput, event_type: carb.input.KeyboardEventType) -> None:
        self.input = input_key
        self.type = event_type


# Regression tests: After driving with WASD, clicking a UI text field showed
# buffered keystrokes because `_event_callback` did not return True for events
# it consumed, so Kit kept propagating them to UI widgets.
class TestKeyboardButton(omni.kit.test.AsyncTestCase):
    """KeyboardButton input-event consumption regression tests."""

    async def test_event_callback_consumes_press_for_tracked_key(self) -> None:
        """Verify a tracked key press is consumed and marks the button pressed."""
        button = KeyboardButton(carb.input.KeyboardInput.W)
        event = _FakeKeyboardEvent(carb.input.KeyboardInput.W, carb.input.KeyboardEventType.KEY_PRESS)
        self.assertTrue(button._event_callback(event))
        self.assertTrue(button.value)

    async def test_event_callback_consumes_repeat_for_tracked_key(self) -> None:
        """Verify a tracked key repeat is consumed and keeps the button pressed."""
        button = KeyboardButton(carb.input.KeyboardInput.A)
        event = _FakeKeyboardEvent(carb.input.KeyboardInput.A, carb.input.KeyboardEventType.KEY_REPEAT)
        self.assertTrue(button._event_callback(event))
        self.assertTrue(button.value)

    async def test_event_callback_consumes_release_for_tracked_key(self) -> None:
        """Verify a tracked key release is consumed and clears the button after the grace window."""
        button = KeyboardButton(carb.input.KeyboardInput.S)
        # Pre-set the button so we can confirm release flips it back to False.
        button._event_callback(_FakeKeyboardEvent(carb.input.KeyboardInput.S, carb.input.KeyboardEventType.KEY_PRESS))
        event = _FakeKeyboardEvent(carb.input.KeyboardInput.S, carb.input.KeyboardEventType.KEY_RELEASE)
        self.assertTrue(button._event_callback(event))
        # The release is provisional until the grace window expires.
        self.assertTrue(button.value)
        await asyncio.sleep(_KEY_HOLD_GRACE_S * 1.5)
        self.assertFalse(button.value)

    async def test_release_then_press_within_grace_window_keeps_button_held(self) -> None:
        """Verify an auto-repeat RELEASE->PRESS pair does not drop the button."""
        # X11 auto-repeat emits this pair every ~33 ms while the key is physically
        # held. The release must be cancelled by the following press.
        button = KeyboardButton(carb.input.KeyboardInput.W)
        button._event_callback(_FakeKeyboardEvent(carb.input.KeyboardInput.W, carb.input.KeyboardEventType.KEY_PRESS))
        for _ in range(3):
            button._event_callback(
                _FakeKeyboardEvent(carb.input.KeyboardInput.W, carb.input.KeyboardEventType.KEY_RELEASE)
            )
            button._event_callback(
                _FakeKeyboardEvent(carb.input.KeyboardInput.W, carb.input.KeyboardEventType.KEY_PRESS)
            )
            self.assertTrue(button.value)
        # The cancelled releases must not expire into a key-up later on.
        await asyncio.sleep(_KEY_HOLD_GRACE_S * 1.5)
        self.assertTrue(button.value)

    async def test_release_without_prior_press_leaves_button_up(self) -> None:
        """Verify a stray release on an unpressed button does not latch it held."""
        button = KeyboardButton(carb.input.KeyboardInput.D)
        event = _FakeKeyboardEvent(carb.input.KeyboardInput.D, carb.input.KeyboardEventType.KEY_RELEASE)
        self.assertTrue(button._event_callback(event))
        self.assertFalse(button.value)

    async def test_event_callback_does_not_consume_other_keys(self) -> None:
        """Verify events for other keys are ignored and not consumed."""
        # A WASD button must let other keys (e.g. Q) fall through so UI widgets
        # can still receive them. Returning True here is the bug.
        button = KeyboardButton(carb.input.KeyboardInput.W)
        event = _FakeKeyboardEvent(carb.input.KeyboardInput.Q, carb.input.KeyboardEventType.KEY_PRESS)
        self.assertFalse(button._event_callback(event))
        self.assertFalse(button.value)

    async def test_event_callback_does_not_consume_unhandled_event_type(self) -> None:
        """Verify CHAR events are ignored even for the tracked key."""
        # `CHAR` events are produced for text input; we should never consume those
        # or text fields will be starved.
        button = KeyboardButton(carb.input.KeyboardInput.D)
        event = _FakeKeyboardEvent(carb.input.KeyboardInput.D, carb.input.KeyboardEventType.CHAR)
        self.assertFalse(button._event_callback(event))
        self.assertFalse(button.value)
