# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Deprecated legacy Python implementation of the Newton tensor API.

This implementation is retained for source compatibility but is no longer exported as
``isaacsim.physics.newton.tensors``. Use the API provided by the
``isaacsim.physics.newton.tensors`` extension instead.
"""

import warnings

__all__: list[str] = []

warnings.warn(
    "isaacsim.physics.newton.impl.tensors is deprecated; use the isaacsim.physics.newton.tensors extension instead",
    DeprecationWarning,
    stacklevel=2,
)
