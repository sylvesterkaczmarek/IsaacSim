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

# setup threading
import carb

threadCount = carb.settings.get_settings().get("/plugins/carb.tasking.plugin/threadCount")

if threadCount:
    # set threading related environment variables
    import os

    os.environ["OMP_NUM_THREADS"] = str(threadCount)
    os.environ["KMP_NUM_THREADS"] = str(threadCount)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threadCount)
    os.environ["MKL_NUM_THREADS"] = str(threadCount)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threadCount)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threadCount)

from .bindings import _simulation_manager  # noqa: F401
from .impl.extension import Extension as Extension
from .impl.isaac_events import IsaacEvents as IsaacEvents
from .impl.mjc_scene import NewtonMjcScene as NewtonMjcScene
from .impl.physics_scene import PhysicsScene as PhysicsScene
from .impl.physx_scene import PhysxGpuCfg as PhysxGpuCfg
from .impl.physx_scene import PhysxScene as PhysxScene

# from .impl.vbd_scene import NewtonVbdScene as NewtonVbdScene
from .impl.simulation_event import SimulationEvent as SimulationEvent
from .impl.simulation_manager import SimulationManager as SimulationManager
from .impl.vbd_scene import NewtonVbdScene as NewtonVbdScene
from .impl.xpbd_scene import NewtonXpbdScene as NewtonXpbdScene

__all__ = [
    "IsaacEvents",
    "PhysicsScene",
    "PhysxGpuCfg",
    "PhysxScene",
    "NewtonMjcScene",
    "NewtonXpbdScene",
    # disabled until VBD schema release
    # "NewtonVbdScene",
    "SimulationEvent",
    "SimulationManager",
]
