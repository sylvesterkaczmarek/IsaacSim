# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify profiling through the Kit carrier adapter."""

from __future__ import annotations

import isaacsim.common.profiling as profiling
import omni.kit.test
from isaacsim.common.profiling import frame, instant, is_enabled, start, value, zone


class TestProfilingCarrier(omni.kit.test.AsyncTestCase):
    """Exercise the carried profiling module with Kit's profiler."""

    async def test_profiling_events_use_the_kit_backend(self) -> None:
        """Verify facade admission follows Kit while forwarding representative events."""
        with self.assertRaisesRegex(RuntimeError, "Another profiling host"):
            start()

        enabled = is_enabled()
        with zone("isaacsim.common/carrier/profiling_probe") as active_zone:
            self.assertEqual(active_zone.active, enabled)
            value("isaacsim.common/carrier/integer", 42)
            value("isaacsim.common/carrier/float", 4.2)
            instant("isaacsim.common/carrier/instant")
        frame("isaacsim.common/carrier/frame")

    async def test_module_qualified_zone_usage(self) -> None:
        """Verify zones can be authored through the profiling module namespace."""
        with profiling.zone("isaacsim.common/carrier/module_qualified") as active_zone:
            self.assertEqual(active_zone.active, profiling.is_enabled())

    async def test_profile_decorator_usage(self) -> None:
        """Verify the profiling decorator preserves the wrapped function's behavior."""

        @profiling.profile(zone_name="isaacsim.common/carrier/decorated")
        def increment(value: int) -> int:
            return value + 1

        self.assertEqual(increment(41), 42)
