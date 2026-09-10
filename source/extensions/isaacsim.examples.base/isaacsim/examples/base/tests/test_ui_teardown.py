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

"""Tests that BaseSampleUITemplate teardown does not raise on in-flight work."""

from __future__ import annotations

import asyncio

import omni.kit.test
from isaacsim.examples.base.base_sample_extension_experimental import BaseSampleUITemplate


class _DelayedSample:
    """Sample stub that yields so shutdown can race with load and reset."""

    def __init__(self) -> None:
        self.physics_cleanup_calls = 0
        self.load_completed = False
        self.reset_completed = False

    async def load_world_async(self) -> None:
        for _ in range(5):
            await asyncio.sleep(0)
        self.load_completed = True

    async def reset_async(self) -> None:
        for _ in range(5):
            await asyncio.sleep(0)
        self.reset_completed = True

    def _physics_cleanup(self) -> None:
        self.physics_cleanup_calls += 1


class TestBaseSampleUITeardown(omni.kit.test.AsyncTestCase):
    """Test teardown races in ``BaseSampleUITemplate``."""

    async def _await_pending(self, ui: BaseSampleUITemplate) -> None:
        pending = list(ui._pending_tasks)
        if pending:
            await asyncio.wait(pending)

    async def test_stage_event_after_shutdown_does_not_raise(self) -> None:
        """Ignore stage-closed cleanup after the sample has already been torn down."""
        sample = _DelayedSample()
        ui = BaseSampleUITemplate(sample=sample)
        ui.on_shutdown()
        ui.on_stage_event(None)
        self.assertIsNone(ui.sample)
        self.assertEqual(sample.physics_cleanup_calls, 0)

    async def test_stage_event_cleans_up_while_sample_is_alive(self) -> None:
        """Run physics cleanup on stage close while the sample is still active."""
        sample = _DelayedSample()
        ui = BaseSampleUITemplate(sample=sample)
        ui.on_stage_event(None)
        self.assertEqual(sample.physics_cleanup_calls, 1)
        ui.on_shutdown()

    async def test_load_after_shutdown_does_not_start_work(self) -> None:
        """Ignore Load World clicks after shutdown."""
        sample = _DelayedSample()
        ui = BaseSampleUITemplate(sample=sample)
        ui.on_shutdown()
        ui._on_load_world()
        await self._await_pending(ui)
        self.assertFalse(sample.load_completed)
        self.assertEqual(ui._pending_tasks, set())

    async def test_in_flight_load_is_cancelled_on_shutdown(self) -> None:
        """Cancel an in-flight load instead of calling methods on a torn-down sample."""
        sample = _DelayedSample()
        ui = BaseSampleUITemplate(sample=sample)
        ui._on_load_world()
        self.assertTrue(ui._pending_tasks)
        ui.on_shutdown()
        await self._await_pending(ui)
        self.assertIsNone(ui.sample)
        self.assertEqual(ui._pending_tasks, set())
        self.assertFalse(sample.load_completed)

    async def test_in_flight_reset_is_cancelled_on_shutdown(self) -> None:
        """Cancel an in-flight reset instead of calling methods on a torn-down sample."""
        sample = _DelayedSample()
        ui = BaseSampleUITemplate(sample=sample)
        ui._on_reset()
        self.assertTrue(ui._pending_tasks)
        ui.on_shutdown()
        await self._await_pending(ui)
        self.assertIsNone(ui.sample)
        self.assertEqual(ui._pending_tasks, set())
        self.assertFalse(sample.reset_completed)

    async def test_load_skips_post_await_work_when_shutdown_flag_is_set(self) -> None:
        """Skip post-load UI work when shutdown is flagged even if the sample remains."""
        sample = _DelayedSample()
        ui = BaseSampleUITemplate(sample=sample)
        ui._on_load_world()
        await asyncio.sleep(0)
        ui._is_shutdown = True
        await self._await_pending(ui)
        self.assertTrue(sample.load_completed)
        self.assertIsNotNone(ui.sample)
        self.assertIsNone(ui._stage_event_subscription)
