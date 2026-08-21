# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Python simulation-clock sender failure and rounding behavior."""

from types import SimpleNamespace
from unittest.mock import patch

import omni.graph.core as og
import omni.kit.test
from isaacsim.examples.ipc.nodes.OgnSimpleSendSimulationClockPy import (
    OgnSimpleSendSimulationClockPy,
    OgnSimpleSendSimulationClockPyInternalState,
    _seconds_to_nanoseconds,
)


class TestSimpleSendSimulationClockPyFailures(omni.kit.test.AsyncTestCase):
    """Verify socket failures remain recoverable and clock rounding matches C++."""

    async def test_socket_constructor_failure_returns_false(self) -> None:
        state = OgnSimpleSendSimulationClockPyInternalState()
        db = SimpleNamespace(
            per_instance_state=state,
            inputs=SimpleNamespace(uri="127.0.0.1:12345", simulationTime=0.0),
            outputs=SimpleNamespace(execOut=None),
        )

        with patch(
            "isaacsim.examples.ipc.nodes.OgnSimpleSendSimulationClockPy.socket.socket",
            side_effect=OSError("socket unavailable"),
        ):
            self.assertFalse(OgnSimpleSendSimulationClockPy.compute(db))

        self.assertIsNone(state.sock)
        self.assertEqual(state.uri, "")
        self.assertEqual(db.outputs.execOut, og.ExecutionAttributeState.ENABLED)

    async def test_nanosecond_rounding_matches_llround_half_away_from_zero(self) -> None:
        self.assertEqual(_seconds_to_nanoseconds(0.5e-9), 1)
        self.assertEqual(_seconds_to_nanoseconds(-0.5e-9), -1)
        self.assertEqual(_seconds_to_nanoseconds(2.5e-9), 3)
        self.assertEqual(_seconds_to_nanoseconds(-2.5e-9), -3)
