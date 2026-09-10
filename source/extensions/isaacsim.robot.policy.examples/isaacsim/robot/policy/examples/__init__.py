# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Small public facade for deploying Isaac Lab policies in Isaac Sim.

Descriptor binding, model backends, articulation bridging, and actuator resources live in
their respective submodules. The bundled factories remain root entry points because interactive,
standalone, and cross-extension consumers all deploy those concrete examples.

Extension version 7.0.0 removed ``PolicyController`` and the per-robot policy classes. Their old
import paths resolve to migration shims that raise actionable errors when constructed.
"""

from . import robots as robots
from ._legacy import AnymalFlatTerrainPolicy as AnymalFlatTerrainPolicy
from ._legacy import FrankaOpenDrawerPolicy as FrankaOpenDrawerPolicy
from ._legacy import Go2FlatTerrainPolicy as Go2FlatTerrainPolicy
from ._legacy import H1FlatTerrainPolicy as H1FlatTerrainPolicy
from ._legacy import SpotFlatTerrainPolicy as SpotFlatTerrainPolicy
from .bundled import get_anymal_spec as get_anymal_spec
from .bundled import get_cartpole_spec as get_cartpole_spec
from .bundled import get_franka_spec as get_franka_spec
from .bundled import get_go2_spec as get_go2_spec
from .bundled import get_h1_spec as get_h1_spec
from .bundled import get_spot_spec as get_spot_spec
from .bundled import make_franka_task_state_provider as make_franka_task_state_provider
from .controller import IsaacLabPolicyController as IsaacLabPolicyController
from .env_config import PolicyEnvConfig as PolicyEnvConfig
from .runtime import RobotPolicyRunner as RobotPolicyRunner
from .spec import PolicyArtifact as PolicyArtifact
from .spec import PolicySpec as PolicySpec

__all__ = [
    "AnymalFlatTerrainPolicy",
    "FrankaOpenDrawerPolicy",
    "Go2FlatTerrainPolicy",
    "H1FlatTerrainPolicy",
    "IsaacLabPolicyController",
    "PolicyArtifact",
    "PolicyEnvConfig",
    "RobotPolicyRunner",
    "PolicySpec",
    "SpotFlatTerrainPolicy",
    "get_anymal_spec",
    "get_cartpole_spec",
    "get_franka_spec",
    "get_go2_spec",
    "get_h1_spec",
    "get_spot_spec",
    "make_franka_task_state_provider",
]
