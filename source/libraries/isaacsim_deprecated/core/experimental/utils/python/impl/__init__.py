# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Kit-independent USD utility functions.

The public API of this module is exposed through its submodules (for example
``isaacsim.core.experimental.utils.stage`` and ``isaacsim.core.experimental.utils.ops``) rather than as package-level
names, so this package facade intentionally exports nothing.
"""

# Bind the bundled Kit-free OpenUSD into the process before importing anything that uses ``pxr``.
import isaacsim.foundation.usd.openusd  # noqa: F401

__all__ = []
