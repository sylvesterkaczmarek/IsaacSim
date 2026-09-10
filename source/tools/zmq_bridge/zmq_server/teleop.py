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

"""Held-key tracking for the teleop viewer, tolerant of keyboard auto-repeat."""

from __future__ import annotations

# How long a release waits to be confirmed. Must exceed one auto-repeat interval (~33 ms at 30 Hz).
KEY_HOLD_GRACE_S = 0.1


class TeleopKeys:
    """Which drive keys are held, with releases confirmed by a short delay.

    A release is indistinguishable from an auto-repeat artifact at the moment it arrives; only
    the absence of a following press tells them apart. Deferring the release by
    ``KEY_HOLD_GRACE_S`` gives that following press a chance to arrive and cancel it.
    """

    def __init__(self, grace_s: float = KEY_HOLD_GRACE_S) -> None:
        self._grace_s = grace_s
        self._down: set[int] = set()
        self._release_t: dict[int, float] = {}

    def press(self, code: int, now: float) -> None:
        """Record a key-down event, cancelling any release still awaiting confirmation."""
        code = int(code)
        self._down.add(code)
        self._release_t.pop(code, None)

    def release(self, code: int, now: float) -> None:
        """Record a key-up event. Takes effect only if no press follows within the grace window."""
        code = int(code)
        if code in self._down:
            self._release_t.setdefault(code, now)

    def is_held(self, code: int, now: float) -> bool:
        """True while ``code`` is down and any pending release is still unconfirmed."""
        code = int(code)
        if code not in self._down:
            return False
        released_at = self._release_t.get(code)
        if released_at is not None and (now - released_at) >= self._grace_s:
            self._down.discard(code)
            del self._release_t[code]
            return False
        return True

    def clear(self) -> None:
        """Forget every key, e.g. when the window loses focus."""
        self._down.clear()
        self._release_t.clear()
