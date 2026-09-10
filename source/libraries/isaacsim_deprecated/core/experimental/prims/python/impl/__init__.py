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

"""Deprecated compatibility alias for the prim wrappers.

.. deprecated::
    This module preserves the ``isaacsim.core.experimental.prims`` import path for existing code.
    New code should import base wrappers from :mod:`isaacsim.foundation.objects` and physics-aware wrappers from
    :mod:`isaacsim.foundation.prims` instead.

``Prim`` aliases its Kit-independent foundation implementation. ``XformPrim`` and ``GeomPrim`` are thin compatibility
subclasses that preserve the experimental non-destructive construction defaults.
"""

# Bind the bundled Kit-free OpenUSD into the process before importing anything that uses ``pxr``.
import isaacsim.foundation.usd.openusd  # noqa: F401
from isaacsim.foundation.objects import Prim

from .geom_prim import GeomPrim
from .xform_prim import XformPrim

__all__ = [
    "GeomPrim",
    "Prim",
    "XformPrim",
]
