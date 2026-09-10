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

"""Frozen policy specs for the bundled robot examples.

This package is example data: one frozen ``PolicySpec`` per shipped robot/task, imported by
interactive examples, standalone scripts, and tests. The generic machinery never reads it.
Every spec derives its observation/action interface from its hosted IO descriptor at bind time;
Franka additionally uses this package's explicit binding hook and task-state-provider factory
for its multi-articulation drawer task.
"""

from __future__ import annotations

from .anymal import get_anymal_spec
from .cartpole import get_cartpole_spec
from .franka import get_franka_spec, make_franka_task_state_provider
from .go2 import get_go2_spec
from .h1 import get_h1_spec
from .spot import get_spot_spec

__all__ = [
    "get_anymal_spec",
    "get_cartpole_spec",
    "get_franka_spec",
    "get_go2_spec",
    "get_h1_spec",
    "get_spot_spec",
    "make_franka_task_state_provider",
]
