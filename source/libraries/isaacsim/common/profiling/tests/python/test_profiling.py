# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the public Python profiling facade without an application profiler."""

from __future__ import annotations

import asyncio
import sys

import isaacsim.common.profiling as profiling
import pytest


def test_standalone_nvtx_lifecycle() -> None:
    """Verify explicit startup, duplicate-start validation, and idempotent shutdown."""
    profiling.stop()
    assert not profiling.is_enabled()

    profiling.start()
    try:
        assert profiling.is_enabled()
        with profiling.zone("test/standalone") as zone:
            assert zone.active
        with pytest.raises(RuntimeError, match="already running"):
            profiling.start()
    finally:
        profiling.stop()

    assert not profiling.is_enabled()
    profiling.stop()


def test_no_application_profiler_is_a_lazy_noop() -> None:
    """Verify disabled zones do not evaluate dynamic names."""
    constructed = False

    def make_name() -> str:
        nonlocal constructed
        constructed = True
        return "disabled"

    assert not profiling.is_enabled()
    with profiling.zone(make_name) as zone:
        assert not zone.active
    assert not constructed


def test_static_zone_names_are_validated() -> None:
    """Verify that author errors are reported even when profiling is disabled."""
    with pytest.raises(ValueError, match="must not be empty"):
        profiling.zone("")
    with pytest.raises(TypeError, match="must be strings"):
        profiling.zone(1)  # type: ignore[arg-type]


def test_disabled_static_zone_does_not_capture_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify disabled static zones return before inspecting their caller frame.

    Args:
        monkeypatch: Pytest fixture used to reject source-frame inspection.
    """

    def fail_source_capture(*args: object) -> None:
        raise AssertionError("disabled zone captured its source")

    monkeypatch.setattr(sys, "_getframe", fail_source_capture)
    with profiling.zone("disabled") as zone:
        assert not zone.active


def test_decorators_preserve_sync_and_async_behavior() -> None:
    """Verify decorators do not change calls when no profiler is attached."""

    @profiling.profile
    def add(left: int, right: int) -> int:
        return left + right

    @profiling.profile(zone_name="example/async")
    async def multiply(left: int, right: int) -> int:
        return left * right

    assert add.__name__ == "add"
    assert add(2, 3) == 5
    assert multiply.__name__ == "multiply"
    assert asyncio.run(multiply(3, 4)) == 12


def test_event_helpers_are_noops_without_a_profiler() -> None:
    """Verify every explicit event helper is safe on the disabled path."""
    profiling.frame("example/frame")
    profiling.value("example/count", 3)
    profiling.value("example/duration_ms", 1.5)
    profiling.instant("example/ready", profiling.InstantType.PROCESS)
    profiling.flow_begin(7, "example/handoff")
    profiling.flow_end(7)
    profiling.set_thread_name("example-worker")
