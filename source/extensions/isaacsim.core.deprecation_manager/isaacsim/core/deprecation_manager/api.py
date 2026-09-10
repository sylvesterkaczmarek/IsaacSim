# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Functions for importing Python packages and modules with enhanced error handling and deprecation notices."""

from __future__ import annotations

import functools
import importlib
import sys
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any, TypeVar, cast

import carb
import omni.kit.app

_F = TypeVar("_F", bound=Callable[..., Any])


class _StubModule:
    """No-op mock module returned during stub generation when a dependency is unavailable.

    Attribute access returns another ``_StubModule``, and calling an instance
    acts as an identity decorator so that patterns like ``@torch.jit.script``
    do not crash.
    """

    def __getattr__(self, _name: str) -> _StubModule:
        return _StubModule()

    def __call__(self, *args, **kwargs) -> object:  # noqa: ANN002, ANN003
        if args and callable(args[0]):
            return args[0]
        return _StubModule()


def import_module(name: str) -> ModuleType:
    """Try to import a Python package/module and return it.

    If a package or module is not found, an error message will be logged and the application will exit.
    A notice with further instructions will also be logged for the following packages:

    - ``torch``

    Args:
        name: The name of the package/module to import.

    Returns:
        The imported module.

    Example:

    .. code-block:: python

        >>> from isaacsim.core.deprecation_manager import import_module
        >>>
        >>> numpy = import_module("numpy")
    """

    def exit_app() -> None:
        if carb.settings.get_settings().get_as_bool("/app/stubgen/enabled"):
            return
        if carb.settings.get_settings().get_as_bool("/exts/omni.kit.test/runTestsAndQuit"):
            sys.exit(1)
        omni.kit.app.get_app().shutdown()

    # PyTorch
    if name == "torch":
        try:
            return importlib.import_module(name)
        except (ModuleNotFoundError, ImportError) as e:
            if carb.settings.get_settings().get_as_bool("/app/stubgen/enabled"):
                return _StubModule()
            msg = """
============================================================================
========================== IMPLEMENTATION NOTICE ===========================
============================================================================

PyTorch (torch) dependency is not installed/enabled by default in Isaac Sim.
Please, follow the instructions below to install and use it.

For a specific PyTorch version, see: https://pytorch.org/get-started/locally
----------------------------------------------------------------------------

 * Isaac Sim - Binary installation (Linux: ./python.sh, Windows: python.bat):
  
    ./python.sh -m pip install torch
  
 * Isaac Sim - Python Packages (pip) installation:
  
    pip install torch

============================================================================
"""
            carb.log_error(f"Import error: {str(e)}")
            carb.log_warn(msg)
            exit_app()
    # any other module
    else:
        try:
            return importlib.import_module(name)
        except (ModuleNotFoundError, ImportError) as e:
            if carb.settings.get_settings().get_as_bool("/app/stubgen/enabled"):
                return _StubModule()
            carb.log_error(f"Import error: {str(e)}")
            exit_app()


def deprecated(
    obj: _F | None = None,
    reason: str | None = None,
    replacement: str | None = None,
    removal_version: str | None = None,
) -> _F | Callable[[_F], _F]:
    """Mark a public function or class as deprecated.

    Args:
        obj: Function or class to decorate when used as ``@deprecated``.
        reason: Optional explanation for the deprecation.
        replacement: Optional replacement API name or import path.
        removal_version: Optional version when the symbol is expected to be removed.

    Returns:
        A decorated function/class, or a decorator when called with keyword arguments.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.deprecation_manager as deprecation_manager
        >>>
        >>> @deprecation_manager.deprecated(replacement="new_function", removal_version="6.0")
        ... def old_function():
        ...     return new_function()
    """

    def build_message(name: str) -> str:
        parts = [f"{name} is deprecated."]
        if replacement:
            parts.append(f"Use {replacement} instead.")
        if removal_version:
            parts.append(f"It will be removed in {removal_version}.")
        if reason:
            parts.append(reason)
        return " ".join(parts)

    def decorate(deprecated_obj: _F) -> _F:
        if isinstance(deprecated_obj, type):
            original_init = deprecated_obj.__init__

            @functools.wraps(original_init)
            def wrapped_init(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN001
                warnings.warn(build_message(deprecated_obj.__qualname__), DeprecationWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            deprecated_obj.__init__ = wrapped_init
            return deprecated_obj

        @functools.wraps(deprecated_obj)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(build_message(deprecated_obj.__qualname__), DeprecationWarning, stacklevel=2)
            return deprecated_obj(*args, **kwargs)

        return cast(_F, wrapped)

    if obj is not None:
        return decorate(obj)
    return decorate
