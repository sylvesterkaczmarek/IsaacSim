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

newton_solver_to_api_schema = {
    "mujoco": "MjcSceneAPI",
    "xpbd": "NewtonXpbdSceneAPI",
    "vbd": "NewtonVbdSceneAPI",
}


# This is only called from simulation manager.
# However, I still want to leave it here, so that all solver related dictionaries to be at the same place
def get_newton_solver_to_physics_scene_object() -> dict:
    # import inside to lazy import
    from isaacsim.core.simulation_manager.impl.mjc_scene import NewtonMjcScene
    from isaacsim.core.simulation_manager.impl.vbd_scene import NewtonVbdScene
    from isaacsim.core.simulation_manager.impl.xpbd_scene import NewtonXpbdScene

    newton_solver_to_physics_scene_object = {
        "mujoco": NewtonMjcScene,
        "xpbd": NewtonXpbdScene,
        "vbd": NewtonVbdScene,
    }
    return newton_solver_to_physics_scene_object


from .solver_config import MuJoCoSolverConfig, VBDSolverConfig, XPBDSolverConfig

newton_solver_to_solver_config = {
    "mujoco": MuJoCoSolverConfig,
    "xpbd": XPBDSolverConfig,
    "vbd": VBDSolverConfig,
}

from newton.solvers import SolverMuJoCo, SolverVBD, SolverXPBD

newton_solver_to_solver_object = {
    "mujoco": SolverMuJoCo,
    "xpbd": SolverXPBD,
    "vbd": SolverVBD,
}

from pxr import Sdf, Usd


# public API for switching solvers on a newton physics scene prim
# We need this API because the UI switches "newton:solver" attribute,
# and the swapping of the API schema is through a UI callback.
# We need a UI free API that changes the API schema based on given solver
def switch_newton_solver(newton_solver: str, prim: Usd.Prim) -> bool:
    """API for switching newton solver on a physics scene prim

    Args:
        newton_solver: solver to switch to
        prim: the physics scene prim
    Returns:
        If the operation is successful
    """
    if not newton_solver in newton_solver_to_api_schema:
        return False

    if not prim.GetAttribute("newton:solver"):
        # newton:solver is created by newton ui extension, so it may not exist
        prim.CreateAttribute("newton:solver", Sdf.ValueTypeNames.Token).Set(newton_solver)

    # If we already have the API do nothing.
    # User should always call switch_newton_solver or use UI to switch the solver
    # so API schema should always match the attribute
    if prim.HasAPI(newton_solver_to_api_schema[newton_solver]):
        return True

    # Set the attribute
    prim.CreateAttribute("newton:solver", Sdf.ValueTypeNames.Token).Set(newton_solver)
    for solver in newton_solver_to_api_schema:
        api = newton_solver_to_api_schema[solver]
        if newton_solver == solver:
            if not prim.HasAPI(api):
                prim.ApplyAPI(api)
        else:
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
    return True


def get_newton_solver(prim: Usd.Prim) -> str:
    attr = prim.GetAttribute("newton:solver")
    # return the default solver, if attribute does not exist.
    # old name was mjcwarp. adding it for backward compatibility
    if not attr or not attr.Get() or attr.Get() == "mjcwarp":
        return "mujoco"
    return attr.Get()
