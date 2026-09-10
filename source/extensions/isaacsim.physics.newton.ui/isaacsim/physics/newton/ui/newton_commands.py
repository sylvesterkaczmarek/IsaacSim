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

"""Commands that are related newton simulations."""

import inspect
from typing import Any

import carb
import omni.usd
from omni.kit.commands import Command, execute, register_all_commands_in_module
from omni.usd.commands.usd_commands import DeletePrimsCommand
from pxr import Sdf, Usd


# This is defined in omni.physx.utils, but we can't depends on physx
def is_defined(stage: Usd.Stage, path: Sdf.Path) -> bool:
    """Report whether a prim already exists at the given path.

    Args:
        stage: USD stage to query.
        path: Prim path to test.

    Returns:
        ``True`` when a valid prim already exists at ``path``.
    """
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        carb.log_warn(f"Prim at path {path} is already defined")
        return True
    return False


class CreateNewtonActuatorCommand(Command):
    """Create a ``NewtonActuator`` prim under the given parent path.

    Args:
        stage: USD stage receiving the new prim.
        parent_path: Path of the prim the actuator is created under.
    """

    @classmethod
    def execute(cls, *args: Any, **kwargs: Any) -> Any:
        """Bind the constructor signature and dispatch through the command registry.

        Args:
            *args: Positional arguments forwarded to the constructor.
            **kwargs: Keyword arguments forwarded to the constructor.

        Returns:
            The result of the registered command execution.
        """
        sig = inspect.signature(cls.__init__)
        bound = sig.bind(0, *args, **kwargs)
        bound.arguments.pop("self")
        return execute(cls.__name__, **bound.kwargs)

    def __init__(self, stage: Usd.Stage, parent_path: str) -> None:
        self._parent_path = parent_path
        self._prim_path = ""
        self._stage = stage

    def do(self) -> Usd.Prim | None:
        """Create the actuator prim.

        Returns:
            The created prim, or ``None`` when the path was taken or creation failed.
        """
        path = Sdf.Path(self._parent_path).AppendPath("newton_actuator")
        if is_defined(self._stage, path):
            return None
        prim = self._stage.DefinePrim(path, "NewtonActuator")
        if prim and prim.IsValid():
            self._prim_path = prim.GetPath().pathString
            return prim
        return None

    def undo(self) -> None:
        """Delete the prim the command created."""
        DeletePrimsCommand([self._prim_path]).do()


class CreateMujocoPrimCommand(Command):
    """Create a concrete MuJoCo typed prim (e.g. MjcActuator, MjcKeyframe, MjcTendon).

    Unlike applied API schemas, these MuJoCo schemas are standalone prim types and must be
    instantiated as their own prims rather than layered onto an existing prim.

    Args:
        stage: USD stage receiving the new prim.
        parent_path: Path of the prim the new prim is created under.
        type_name: Concrete MuJoCo schema type to instantiate.
        base_name: Base prim name, made unique when the path is already taken.
    """

    @classmethod
    def execute(cls, *args: Any, **kwargs: Any) -> Any:
        """Bind the constructor signature and dispatch through the command registry.

        Args:
            *args: Positional arguments forwarded to the constructor.
            **kwargs: Keyword arguments forwarded to the constructor.

        Returns:
            The result of the registered command execution.
        """
        sig = inspect.signature(cls.__init__)
        bound = sig.bind(0, *args, **kwargs)
        bound.arguments.pop("self")
        return execute(cls.__name__, **bound.kwargs)

    def __init__(self, stage: Usd.Stage, parent_path: str, type_name: str, base_name: str) -> None:
        self._parent_path = parent_path
        self._type_name = type_name
        self._base_name = base_name
        self._prim_path = ""
        self._stage = stage

    def do(self) -> Usd.Prim | None:
        """Create the typed MuJoCo prim.

        Returns:
            The created prim, or ``None`` when creation failed.
        """
        parent = Sdf.Path(self._parent_path)
        # Ensure a unique, unused path so repeated creations don't collide.
        path = omni.usd.get_stage_next_free_path(self._stage, parent.AppendChild(self._base_name).pathString, False)
        prim = self._stage.DefinePrim(path, self._type_name)
        if prim and prim.IsValid():
            self._prim_path = prim.GetPath().pathString
            return prim
        return None

    def undo(self) -> None:
        """Delete the prim the command created."""
        # Reverse the creation on the same stage the prim was authored on. Using the stage
        # directly (rather than DeletePrimsCommand) keeps undo independent of the USD context.
        if self._prim_path and self._stage:
            self._stage.RemovePrim(self._prim_path)
            self._prim_path = ""


register_all_commands_in_module(__name__)
