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

"""Optional module access for the Kit-independent core library.

Some Omniverse runtime modules (for example ``usdrt`` / Fabric and the ``pxr.PhysxSchema`` PhysX schema) are not part
of the Kit-free runtime, so importing them at module load would make this library unimportable in standalone
(pure-OpenUSD) environments. This module provides :func:`optional_import`, which returns the real module when it is
installed (for example inside Kit) and a lightweight placeholder otherwise, and exposes ``usdrt`` resolved that way.

The placeholder resolves any attribute chain (``usdrt.Usd.Prim``, ``PhysxSchema.PhysxCollisionAPI``, ...) to a
unique, never-instantiated type. This keeps ``isinstance(obj, (Usd.Prim, usdrt.Usd.Prim))`` valid on the pure-USD
path (no object is ever an instance of a placeholder type) while deferring any real runtime requirement to the code
paths that actually use the optional backend.
"""

from __future__ import annotations

import importlib

__all__ = ["optional_import", "usdrt", "HAS_USDRT"]


class _MissingMeta(type):
    """Metaclass whose classes resolve further attributes to nested placeholder classes."""

    def __getattr__(cls, name: str) -> _MissingMeta:
        return _make_placeholder(f"{cls.__qualname__}.{name}")


_placeholder_cache: dict[str, _MissingMeta] = {}


def _make_placeholder(qualname: str) -> _MissingMeta:
    placeholder = _placeholder_cache.get(qualname)
    if placeholder is None:
        placeholder = _MissingMeta(qualname, (), {})
        _placeholder_cache[qualname] = placeholder
    return placeholder


class _MissingModule:
    """Stand in for the ``usdrt`` module when Fabric is unavailable.

    Args:
        name: Fully qualified module name.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, name: str) -> _MissingMeta:
        return _make_placeholder(f"{self._name}.{name}")


def optional_import(name: str) -> object:
    """Import a module by name, returning a never-matching placeholder if it is unavailable.

    Args:
        name: The fully qualified module name (for example ``"usdrt"`` or ``"pxr.PhysxSchema"``).

    Returns:
        The imported module if available, otherwise a placeholder whose attribute chains resolve to unique,
        never-instantiated types.
    """
    try:
        return importlib.import_module(name)
    except ImportError:  # pragma: no cover - exercised only in pure-OpenUSD environments
        return _MissingModule(name)


usdrt = optional_import("usdrt")
HAS_USDRT = not isinstance(usdrt, _MissingModule)
