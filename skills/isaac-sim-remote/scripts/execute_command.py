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

"""Dispatch a named, already-registered omni.kit.commands entry.

Lists the Kit command registry or runs one registered name with JSON kwargs.
Intended for live stage iteration through the localhost python_server control
plane (see SKILL.md Security). Names must match the registry; unknown names
raise ValueError before any execute call.

Injected globals (via isaacsim_send.py --arg):
    action: str — "run" (default) or "list".
    command_name: str — Registered command name for "run" (e.g. "CreateMeshPrimWithDefaultXform").
    kwargs: str — JSON string of keyword arguments (e.g. '{"prim_type":"Cube"}').
    filter: str — Filter string for "list" action (case-insensitive substring match).
    undo_last: str — If "true", undo the last command instead of running one.
"""

if "action" not in dir():
    action = "run"
if "command_name" not in dir():
    command_name = None
if "kwargs" not in dir():
    kwargs = None
if "filter" not in dir():
    filter = None  # noqa: A001
if "undo_last" not in dir():
    undo_last = "false"

import re

import omni.kit.commands

_COMMAND_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _resolve_registered_command(name: str) -> str:
    """Return a registered command key matching ``name``, or raise ValueError."""
    if not isinstance(name, str) or not _COMMAND_NAME_RE.match(name):
        raise ValueError(f"invalid command_name {name!r}; expected a registered Kit command identifier")
    registered = omni.kit.commands.get_commands()
    if name in registered:
        return name
    with_suffix = f"{name}Command"
    if with_suffix in registered:
        return with_suffix
    raise ValueError(f"command_name {name!r} is not in the registered omni.kit.commands set")


if action == "list":
    cmds = sorted(omni.kit.commands.get_commands().keys())
    # Deduplicate (many commands register with and without "Command" suffix)
    unique = sorted(set(c.removesuffix("Command") for c in cmds))
    if filter:
        unique = [c for c in unique if filter.lower() in c.lower()]
    print(f"Commands ({len(unique)}):")
    for c in unique:
        print(f"  {c}")

elif action == "run":
    if not command_name:
        raise ValueError("command_name required (e.g. --arg command_name=CreateMeshPrimWithDefaultXform)")

    resolved_name = _resolve_registered_command(command_name)

    cmd_kwargs = {}
    if kwargs:
        if isinstance(kwargs, dict):
            cmd_kwargs = kwargs
        else:
            import json

            cmd_kwargs = json.loads(kwargs)

    import isaacsim.core.experimental.utils.app as app_utils

    print(f"Dispatching registered command: {resolved_name} args={cmd_kwargs}")
    result = omni.kit.commands.execute(resolved_name, **cmd_kwargs)
    app_utils.update_app(steps=5)

    print(f"Executed: {resolved_name}")
    if cmd_kwargs:
        print(f"Args: {cmd_kwargs}")
    if result is not None:
        print(f"Result: {result}")

elif undo_last.lower() == "true":
    omni.kit.commands.undo()
    print("Undo: last command undone")

else:
    print(f"ERROR: Unknown action '{action}'. Use: run, list")
