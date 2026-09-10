# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manage the package-owned standalone Carbonite NVTX profiler."""

from __future__ import annotations

import atexit
import ctypes
import os
import sys
import threading
from pathlib import Path

_LOCK = threading.RLock()
_LIBRARY: ctypes.CDLL | None = None
_LIBRARY_DIRECTORY: Path | None = None
_STARTED = False


def _library_filename() -> str:
    """Get the platform-specific standalone profiling host filename.

    Returns:
        Platform-specific shared-library filename.
    """
    if sys.platform == "win32":
        return "isaacsim-common-profiling-carbonite-host.dll"
    if sys.platform == "darwin":
        return "libisaacsim-common-profiling-carbonite-host.dylib"
    return "libisaacsim-common-profiling-carbonite-host.so"


def _library_candidates() -> tuple[Path, ...]:
    """Get the supported package locations for the standalone profiling host.

    Returns:
        Candidate paths in lookup order.
    """
    module_directory = Path(__file__).absolute().parents[1]
    shared_library_directory = module_directory.parents[1] / "lib"
    binding_directory = module_directory / "bindings"
    filename = _library_filename()
    if sys.platform == "win32":
        return binding_directory / filename, shared_library_directory / filename
    return shared_library_directory / filename, binding_directory / filename


def _load_library() -> tuple[ctypes.CDLL, Path]:
    """Load and configure the standalone profiling host library.

    Returns:
        Loaded library and the directory containing it.

    Raises:
        RuntimeError: If the profiling host cannot be found or loaded.
    """
    global _LIBRARY
    global _LIBRARY_DIRECTORY

    if _LIBRARY is not None and _LIBRARY_DIRECTORY is not None:
        return _LIBRARY, _LIBRARY_DIRECTORY

    candidates = _library_candidates()
    library_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if library_path is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise RuntimeError(f"The standalone profiling host is not installed; searched {searched}")

    try:
        if sys.platform == "win32":
            with os.add_dll_directory(str(library_path.parent)):
                library = ctypes.CDLL(str(library_path))
        else:
            library = ctypes.CDLL(str(library_path))
    except OSError as error:
        raise RuntimeError(f"Could not load the standalone profiling host: {error}") from error

    library.isaacsimCommonProfilingCarboniteHostStart.argtypes = [ctypes.c_char_p]
    library.isaacsimCommonProfilingCarboniteHostStart.restype = ctypes.c_int32
    library.isaacsimCommonProfilingCarboniteHostStop.argtypes = []
    library.isaacsimCommonProfilingCarboniteHostStop.restype = ctypes.c_int32
    library.isaacsimCommonProfilingCarboniteHostGetLastError.argtypes = []
    library.isaacsimCommonProfilingCarboniteHostGetLastError.restype = ctypes.c_char_p
    _LIBRARY = library
    _LIBRARY_DIRECTORY = library_path.parent
    return library, library_path.parent


def _get_last_error(library: ctypes.CDLL) -> str:
    """Get the most recent native host error message.

    Args:
        library: Loaded standalone profiling host library.

    Returns:
        Most recent native error, or a fallback message when none is available.
    """
    message = library.isaacsimCommonProfilingCarboniteHostGetLastError()
    if not message:
        return "The standalone profiling host failed"
    return message.decode("utf-8", errors="replace")


def start() -> None:
    """Start the package-owned Carbonite NVTX profiler.

    Raises:
        RuntimeError: If standalone profiling is already running, another profiling host is attached, or the native
            host cannot start.

    Example:

        .. code-block:: python

            import isaacsim.common.profiling as profiling

            profiling.start()
            try:
                with profiling.zone("example/work"):
                    run_work()
            finally:
                profiling.stop()
    """
    global _STARTED

    with _LOCK:
        if _STARTED:
            raise RuntimeError("Standalone profiling is already running")
        library, plugin_directory = _load_library()
        result = library.isaacsimCommonProfilingCarboniteHostStart(os.fsencode(plugin_directory))
        if result != 0:
            raise RuntimeError(_get_last_error(library))
        _STARTED = True


def stop() -> None:
    """Stop the package-owned Carbonite NVTX profiler.

    This function has no effect when :func:`start` has not successfully started a session. It never detaches a
    profiler supplied by Kit or another application host.

    Raises:
        RuntimeError: If the native host cannot stop an active standalone session.

    Example:

        .. code-block:: python

            profiling.start()
            try:
                run_profiled_work()
            finally:
                profiling.stop()
    """
    global _STARTED

    with _LOCK:
        if not _STARTED:
            return
        assert _LIBRARY is not None
        result = _LIBRARY.isaacsimCommonProfilingCarboniteHostStop()
        if result != 0:
            raise RuntimeError(_get_last_error(_LIBRARY))
        _STARTED = False


def _stop_at_exit() -> None:
    """Stop an active standalone profiler during interpreter shutdown."""
    try:
        stop()
    except Exception:
        pass


atexit.register(_stop_at_exit)
