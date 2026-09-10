# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide explicit Python profiling through Carbonite."""

from __future__ import annotations

import asyncio
import functools
import sys
from collections.abc import Callable
from typing import Any, TypeVar, overload

from isaacsim.common.profiling.bindings import _bindings
from isaacsim.common.profiling.bindings._bindings import InstantType as InstantType
from isaacsim.common.profiling.bindings._bindings import Zone as Zone
from isaacsim.common.profiling.bindings._bindings import flow_begin as flow_begin
from isaacsim.common.profiling.bindings._bindings import flow_end as flow_end
from isaacsim.common.profiling.bindings._bindings import frame as frame
from isaacsim.common.profiling.bindings._bindings import instant as instant
from isaacsim.common.profiling.bindings._bindings import is_enabled as is_enabled
from isaacsim.common.profiling.bindings._bindings import set_thread_name as set_thread_name
from isaacsim.common.profiling.bindings._bindings import value as value

from ._standalone import start as start
from ._standalone import stop as stop

_Callable = TypeVar("_Callable", bound=Callable[..., Any])
_ZoneName = str | Callable[[], str]


def zone(name: _ZoneName, *, mask: int = 0, capture_source: bool = True) -> Zone:
    """Create a profiling zone for use as a context manager.

    Args:
        name: Stable zone name or a factory evaluated only when profiling is enabled.
        mask: Carbonite capture-mask bits. Zero selects Carbonite's default mask.
        capture_source: Whether to include the Python source location.

    Returns:
        Active zone when a profiler admits the mask, otherwise an inactive context manager.

    Raises:
        TypeError: If the selected name is not a string.
        ValueError: If the selected name is empty.
    """
    enabled = is_enabled(mask)
    if callable(name):
        if not enabled:
            return _bindings.begin("", mask=mask, capture_source=False)
        selected_name = name()
    else:
        selected_name = name
    if not isinstance(selected_name, str):
        raise TypeError("Profiling zone names must be strings")
    if not selected_name:
        raise ValueError("Profiling zone names must not be empty")
    if not enabled:
        return _bindings.begin("", mask=mask, capture_source=False)
    if not capture_source:
        return _bindings.begin(selected_name, mask=mask, capture_source=False)
    frame = sys._getframe(1)
    return _bindings.begin(
        selected_name,
        mask=mask,
        capture_source=True,
        source_file=frame.f_code.co_filename,
        source_function=frame.f_code.co_name,
        source_line=frame.f_lineno,
    )


@overload
def profile(function: _Callable) -> _Callable: ...


@overload
def profile(
    function: None = None,
    *,
    mask: int = 0,
    zone_name: str | None = None,
    add_args: bool = False,
    capture_source: bool = True,
) -> Callable[[_Callable], _Callable]: ...


def profile(
    function: _Callable | None = None,
    *,
    mask: int = 0,
    zone_name: str | None = None,
    add_args: bool = False,
    capture_source: bool = True,
) -> _Callable | Callable[[_Callable], _Callable]:
    """Decorate a synchronous or asynchronous function with a profiling zone.

    Args:
        function: Function selected when the decorator is used without arguments.
        mask: Carbonite capture-mask bits. Zero selects Carbonite's default mask.
        zone_name: Explicit zone name. The function name is used when omitted.
        add_args: Whether to append positional and keyword arguments while profiling is enabled.
        capture_source: Whether to include the Python source location.

    Returns:
        Decorated function or a decorator configured with the selected options.
    """

    def decorate(selected: _Callable) -> _Callable:
        code = getattr(selected, "__code__", None)
        source_file = "" if code is None else code.co_filename
        source_function = "" if code is None else code.co_name
        source_line = 0 if code is None else code.co_firstlineno

        def get_name(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
            name = zone_name or selected.__name__
            if add_args:
                name += f"{args}{kwargs}"
            return name

        if asyncio.iscoroutinefunction(selected):

            @functools.wraps(selected)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not is_enabled(mask):
                    return await selected(*args, **kwargs)
                with _bindings.begin(
                    get_name(args, kwargs),
                    mask=mask,
                    capture_source=capture_source,
                    source_file=source_file,
                    source_function=source_function,
                    source_line=source_line,
                ):
                    return await selected(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(selected)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_enabled(mask):
                return selected(*args, **kwargs)
            with _bindings.begin(
                get_name(args, kwargs),
                mask=mask,
                capture_source=capture_source,
                source_file=source_file,
                source_function=source_function,
                source_line=source_line,
            ):
                return selected(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate(function) if function is not None else decorate


__all__ = [
    "InstantType",
    "Zone",
    "flow_begin",
    "flow_end",
    "frame",
    "instant",
    "is_enabled",
    "profile",
    "set_thread_name",
    "start",
    "stop",
    "value",
    "zone",
]
