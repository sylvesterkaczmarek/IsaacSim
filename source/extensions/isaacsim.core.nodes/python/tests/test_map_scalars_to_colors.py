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

"""Test the IsaacMapScalarsToColors OmniGraph node."""

from __future__ import annotations

import ctypes
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import carb
import numpy as np
import omni.graph.core as og
import omni.kit.app
import omni.kit.test
import omni.usd

# RGBA float color, matching the pxr::GfVec4f the node writes
_FLOATS_PER_COLOR = 4
_COLOR_NBYTES = _FLOATS_PER_COLOR * 4

# Ramp stops the node interpolates between, low scalar (blue) to high scalar (red)
_RAMP_BLUE = (0.0, 0.0, 1.0)
_RAMP_CYAN = (0.0, 1.0, 1.0)
_RAMP_GREEN = (0.0, 1.0, 0.0)
_RAMP_YELLOW = (1.0, 1.0, 0.0)
_RAMP_RED = (1.0, 0.0, 0.0)


@contextmanager
def _capture_errors() -> Iterator[list[str]]:
    """Capture carb ERROR-level log messages emitted while in the context.

    Yields:
        Mutable list populated with captured error messages.
    """
    messages: list[str] = []

    def _on_event(e: Any) -> None:
        if (e.get("source") or "") == "omni.kit.app._impl":
            return
        msg = e.get("message") or ""
        if msg:
            messages.append(msg)

    sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.kit.app.GLOBAL_EVENT_ERROR_LOG_IMMEDIATE,
        on_event=_on_event,
        observer_name="test_map_scalars_to_colors._capture_errors",
    )
    try:
        yield messages
    finally:
        sub = None  # noqa: F841


class TestMapScalarsToColors(omni.kit.test.AsyncTestCase):
    """Verify scalar buffers map to the expected color ramp."""

    GRAPH_PATH = "/ActionGraph"
    NODE_NAME = "MapScalars"

    async def setUp(self) -> None:
        """Create a fresh stage and reset test-owned buffers."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        # Keep a reference to the scalar buffer alive so its pointer stays valid through compute()
        self._scalars = None
        # Each _compute() builds a fresh graph so a single test can run several computes
        self._graph_count = 0

    async def tearDown(self) -> None:
        """Close the stage after each test."""
        await omni.usd.get_context().close_stage_async()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _compute(
        self,
        *,
        scalar_ptr: int,
        scalar_buffer_size: int,
        base_color: tuple[float, float, float, float] | None = None,
        log_scale: bool = False,
    ) -> tuple[int, int]:
        """Build and evaluate an IsaacMapScalarsToColors graph.

        Args:
            scalar_ptr: Address of the input scalar buffer.
            scalar_buffer_size: Size of the input scalar buffer in bytes.
            base_color: Color multiplier applied to the generated ramp, or ``None`` to use the node default.
            log_scale: Whether to map scalar values logarithmically.

        Returns:
            Address of the generated color buffer and its size in bytes.
        """
        # Use a fresh graph path each call so a test may run more than one compute
        graph_path = f"{self.GRAPH_PATH}_{self._graph_count}"
        self._graph_count += 1

        set_values = [
            (f"{self.NODE_NAME}.inputs:scalarPtr", int(scalar_ptr)),
            (f"{self.NODE_NAME}.inputs:scalarBufferSize", int(scalar_buffer_size)),
            (f"{self.NODE_NAME}.inputs:logScaleMode", bool(log_scale)),
            ("OnImpulse.inputs:onlyPlayback", False),
        ]
        if base_color is not None:
            set_values.append((f"{self.NODE_NAME}.inputs:baseColor", list(base_color)))

        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                    (self.NODE_NAME, "isaacsim.core.nodes.IsaacMapScalarsToColors"),
                ],
                og.Controller.Keys.SET_VALUES: set_values,
                og.Controller.Keys.CONNECT: [
                    ("OnImpulse.outputs:execOut", f"{self.NODE_NAME}.inputs:execIn"),
                ],
            },
        )
        await omni.kit.app.get_app().next_update_async()
        og.Controller.attribute(f"{graph_path}/OnImpulse.state:enableImpulse").set(True)
        await og.Controller.evaluate(graph_path)

        colors_ptr = int(og.Controller.get(f"{graph_path}/{self.NODE_NAME}.outputs:colorsPtr"))
        colors_buffer_size = int(og.Controller.get(f"{graph_path}/{self.NODE_NAME}.outputs:colorsBufferSize"))
        return colors_ptr, colors_buffer_size

    async def _compute_from_scalars(
        self,
        scalars: Sequence[float] | np.ndarray,
        **kwargs: object,
    ) -> tuple[int, int, np.ndarray]:
        """Map scalar values and decode the resulting color buffer.

        Args:
            scalars: Scalar values to map to colors.
            **kwargs: Additional keyword arguments forwarded to `_compute()`.

        Returns:
            Color buffer address, buffer size in bytes, and decoded colors.
        """
        self._scalars = np.asarray(scalars, dtype=np.float32)
        colors_ptr, colors_buffer_size = await self._compute(
            scalar_ptr=self._scalars.ctypes.data, scalar_buffer_size=self._scalars.nbytes, **kwargs
        )
        return colors_ptr, colors_buffer_size, self._read_colors(colors_ptr, colors_buffer_size)

    @staticmethod
    def _read_colors(colors_ptr: int, colors_buffer_size: int) -> np.ndarray:
        """Decode a host color buffer as a two-dimensional float array.

        Args:
            colors_ptr: Address of the color buffer.
            colors_buffer_size: Size of the color buffer in bytes.

        Returns:
            Decoded colors with shape ``(N, 4)``, or an empty array when the buffer is unavailable.
        """
        if colors_ptr == 0 or colors_buffer_size == 0:
            return np.empty((0, _FLOATS_PER_COLOR), dtype=np.float32)
        num_floats = colors_buffer_size // 4
        buffer = (ctypes.c_float * num_floats).from_address(colors_ptr)
        return np.ctypeslib.as_array(buffer).reshape(-1, _FLOATS_PER_COLOR).copy()

    def _assert_color(
        self, actual: np.ndarray, expected_rgb: Sequence[float], alpha: float = 1.0, places: int = 4
    ) -> None:
        """Assert that an RGBA value matches the expected color.

        Args:
            actual: RGBA value under test.
            expected_rgb: Expected red, green, and blue channels.
            alpha: Expected alpha channel.
            places: Decimal precision used for comparison.
        """
        for channel, value in enumerate((*expected_rgb, alpha)):
            self.assertAlmostEqual(float(actual[channel]), value, places=places, msg=f"channel {channel}")

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    async def test_linear_ramp_endpoints_and_stops(self) -> None:
        """Five evenly spaced scalars map onto the five ramp stops (blue -> cyan -> green -> yellow -> red)."""
        _, colors_buffer_size, colors = await self._compute_from_scalars([1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertEqual(colors_buffer_size, 5 * _COLOR_NBYTES)
        self.assertEqual(colors.shape, (5, _FLOATS_PER_COLOR))
        self._assert_color(colors[0], _RAMP_BLUE)
        self._assert_color(colors[1], _RAMP_CYAN)
        self._assert_color(colors[2], _RAMP_GREEN)
        self._assert_color(colors[3], _RAMP_YELLOW)
        self._assert_color(colors[4], _RAMP_RED)

    async def test_uniform_scalars_map_to_ramp_end(self) -> None:
        """A zero-width value range normalizes to 1.0, so every point gets the high-end (red) color."""
        _, _, colors = await self._compute_from_scalars([2.0, 2.0, 2.0])

        self.assertEqual(colors.shape, (3, _FLOATS_PER_COLOR))
        for color in colors:
            self._assert_color(color, _RAMP_RED)

    async def test_base_color_scales_ramp(self) -> None:
        """Verify that BaseColor scales RGB channels and directly supplies the alpha channel."""
        base_color = (0.5, 0.5, 0.5, 0.7)
        _, _, colors = await self._compute_from_scalars([1.0, 5.0], base_color=base_color)

        # min -> blue * 0.5, max -> red * 0.5, both with alpha 0.7
        self._assert_color(colors[0], (0.0, 0.0, 0.5), alpha=0.7)
        self._assert_color(colors[1], (0.5, 0.0, 0.0), alpha=0.7)

    async def test_non_finite_scalars_get_fallback_color(self) -> None:
        """NaN / Inf scalars receive the (0, 0, 0, 1) fallback while finite ones are colored; count is preserved."""
        _, colors_buffer_size, colors = await self._compute_from_scalars([1.0, float("nan"), 5.0, float("inf")])

        # Alignment is preserved: one color per input scalar, including the non-finite ones
        self.assertEqual(colors_buffer_size, 4 * _COLOR_NBYTES)
        self.assertEqual(colors.shape, (4, _FLOATS_PER_COLOR))
        # Range is taken over the finite values (1 -> blue, 5 -> red)
        self._assert_color(colors[0], _RAMP_BLUE)
        self._assert_color(colors[2], _RAMP_RED)
        # Non-finite scalars -> fallback color
        self._assert_color(colors[1], (0.0, 0.0, 0.0))
        self._assert_color(colors[3], (0.0, 0.0, 0.0))

    async def test_log_scaling_lifts_midrange(self) -> None:
        """Log scaling pushes mid-range scalars further up the ramp than linear scaling does."""
        scalars = [1.0, 3.0, 5.0]
        _, _, linear_colors = await self._compute_from_scalars(scalars, log_scale=False)
        _, _, log_colors = await self._compute_from_scalars(scalars, log_scale=True)

        # Linear puts the midpoint exactly at green; log scaling moves it toward yellow (more red).
        self._assert_color(linear_colors[1], _RAMP_GREEN)
        self.assertGreater(float(log_colors[1][0]), float(linear_colors[1][0]))
        self.assertAlmostEqual(float(log_colors[1][1]), 1.0, places=4)

    async def test_log_scaling_is_true_log_of_raw_values(self):
        """Log mode normalizes log(scalar) across [minPositive, max], not log of a linear-normalized value.

        For [1, 10, 100], log(10) sits exactly halfway between log(1) and log(100), so the middle
        value lands precisely on the green ramp stop; linear scaling would leave it near blue.
        """
        scalars = [1.0, 10.0, 100.0]
        _, _, linear_colors = await self._compute_from_scalars(scalars, log_scale=False)
        _, _, log_colors = await self._compute_from_scalars(scalars, log_scale=True)

        self._assert_color(log_colors[0], _RAMP_BLUE)
        self._assert_color(log_colors[1], _RAMP_GREEN)
        self._assert_color(log_colors[2], _RAMP_RED)
        # Under linear scaling the middle value is nowhere near green (well below the midpoint).
        self.assertLess(float(linear_colors[1][1]), 0.5)

    async def test_log_scaling_clamps_non_positive_to_low_end(self):
        """A logarithm is undefined for values <= 0, so they clamp to the ramp's low (blue) end."""
        scalars = [-5.0, 1.0, 10.0]
        _, _, log_colors = await self._compute_from_scalars(scalars, log_scale=True)

        # minPositive == 1 -> log(1) == 0 -> blue; the negative value clamps to the same low end.
        self._assert_color(log_colors[0], _RAMP_BLUE)
        self._assert_color(log_colors[1], _RAMP_BLUE)
        self._assert_color(log_colors[2], _RAMP_RED)

    async def test_log_scaling_without_positive_values_falls_back_to_linear(self):
        """With no positive values a log range is undefined, so the node falls back to linear scaling."""
        with _capture_errors():
            scalars = [-5.0, -3.0, -1.0]
            _, _, log_colors = await self._compute_from_scalars(scalars, log_scale=True)

        # Linear fallback spreads the range (blue -> green -> red) instead of collapsing everything.
        self._assert_color(log_colors[0], _RAMP_BLUE)
        self._assert_color(log_colors[1], _RAMP_GREEN)
        self._assert_color(log_colors[2], _RAMP_RED)

    async def test_empty_buffer_clears_outputs_without_error(self) -> None:
        """A zero-size scalar buffer is a no-op: outputs are cleared and no error is logged."""
        with _capture_errors() as messages:
            colors_ptr, colors_buffer_size = await self._compute(scalar_ptr=0, scalar_buffer_size=0)

        self.assertEqual(colors_ptr, 0)
        self.assertEqual(colors_buffer_size, 0)
        self.assertFalse(
            any("Scalar buffer pointer" in m for m in messages),
            msg=f"empty buffer should not log an error, got: {messages}",
        )

    async def test_null_pointer_with_nonzero_size_logs_error(self) -> None:
        """An inconsistent input (size > 0, pointer == 0) is an error and clears the outputs."""
        with _capture_errors() as messages:
            colors_ptr, colors_buffer_size = await self._compute(scalar_ptr=0, scalar_buffer_size=_COLOR_NBYTES)

        self.assertEqual(colors_ptr, 0)
        self.assertEqual(colors_buffer_size, 0)
        self.assertTrue(
            any("Scalar buffer pointer" in m for m in messages),
            msg=f"expected the inconsistent-input error, got: {messages}",
        )
