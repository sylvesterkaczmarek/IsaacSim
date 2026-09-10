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

"""Library-owned logging configuration for motion generation."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaacsim.common.logging import Logger

_DEFAULT_LOGGING_CHANNEL = "isaacsim.robot_motion.motion_gen"


class _LoggingState:
    """Manage the logging channel and lazily create its logger."""

    def __init__(self) -> None:
        self._channel = _DEFAULT_LOGGING_CHANNEL
        self._configured = False
        self._logger: Logger | None = None
        self._lock = threading.Lock()

    def configure_channel(self, channel: str) -> None:
        """Configure the channel before the logger is first used."""
        if not channel:
            raise ValueError("The logging channel must not be empty")

        with self._lock:
            if self._logger is not None and self._logger.channel != channel:
                raise RuntimeError("The logging channel cannot be changed after the logger has been created")
            if self._configured and self._channel != channel:
                raise RuntimeError("The logging channel has already been configured")
            self._channel = channel
            self._configured = True

    def get_logger(self) -> Logger:
        """Return the library logger, creating it on first use."""
        with self._lock:
            if self._logger is None:
                from isaacsim.common.logging import Logger

                self._logger = Logger(self._channel)
            return self._logger


class _LazyLogger:
    """Forward attribute access to the library logger without eagerly creating it."""

    def __getattr__(self, name: str) -> object:
        return getattr(_LOGGING_STATE.get_logger(), name)


_LOGGING_STATE = _LoggingState()

if TYPE_CHECKING:
    log: Logger
else:
    log = _LazyLogger()


def configure_logging_channel(channel: str) -> None:
    """Configure the motion generation logging channel before its first use."""
    _LOGGING_STATE.configure_channel(channel)


def get_logger() -> Logger:
    """Return the lazily initialized motion generation logger."""
    return _LOGGING_STATE.get_logger()
