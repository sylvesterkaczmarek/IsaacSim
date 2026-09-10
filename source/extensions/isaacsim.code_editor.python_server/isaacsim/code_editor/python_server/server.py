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

"""Small public runtime API for the Isaac Sim Python server."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class ServerState(Enum):
    """Observable listener lifecycle states."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    RESTARTING = "restarting"
    ERROR = "error"


@dataclass(frozen=True)
class BoundEndpoint:
    """One active listening address."""

    host: str
    port: int


@dataclass(frozen=True)
class ServerStatus:
    """Immutable listener status snapshot."""

    state: ServerState
    configured_host: str
    configured_port: int
    bound_endpoints: tuple[BoundEndpoint, ...]
    active_connections: int
    authentication_required: bool
    last_error: str | None = None


StatusCallback = Callable[[ServerStatus], None]


class _ServerRuntime(Protocol):
    @property
    def status(self) -> ServerStatus: ...

    def subscribe(self, callback: StatusCallback) -> Callable[[], None]: ...

    async def start(self) -> bool: ...

    async def stop(self) -> bool: ...

    async def restart(self, host: str, port: int) -> bool: ...


_active_server: _ServerRuntime | None = None


def get_server_status() -> ServerStatus | None:
    """Return current runtime status, or None when unavailable."""
    return _active_server.status if _active_server is not None else None


def get_server_endpoint() -> tuple[str, int] | None:
    """Return the active client endpoint."""
    status = get_server_status()
    if status is None or status.state != ServerState.RUNNING or not status.bound_endpoints:
        return None
    endpoint = status.bound_endpoints[0]
    host = "127.0.0.1" if endpoint.host == "0.0.0.0" else endpoint.host
    return host, endpoint.port


def subscribe_server_status(callback: StatusCallback) -> Callable[[], None]:
    """Subscribe to status changes and return an unsubscribe callback."""
    if _active_server is None:
        return lambda: None
    return _active_server.subscribe(callback)


async def start_server() -> bool:
    """Start the active listener."""
    return bool(_active_server and await asyncio.shield(_active_server.start()))


async def stop_server() -> bool:
    """Stop the active listener."""
    return bool(_active_server and await asyncio.shield(_active_server.stop()))


async def restart_server(host: str, port: int) -> bool:
    """Restart the active listener at host:port."""
    return bool(_active_server and await asyncio.shield(_active_server.restart(host, port)))


def _set_active_server(server: _ServerRuntime | None) -> None:
    global _active_server
    _active_server = server
