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

"""Implement USD stage utilities."""

__all__ = [
    "get_active_stage",
    "get_default_stage",
    "set_default_stage",
    "use_stage",
]

import contextlib
from collections.abc import Generator

from isaacsim.foundation.objects import Stage

from ..bindings._bindings import (
    _StageGuard,
    get_active_stage,
    get_default_stage,
    set_default_stage,
)


@contextlib.contextmanager
def use_stage(stage: Stage) -> Generator[None, None, None]:
    """Context manager that temporarily overrides the thread-local active stage.

    Args:
        stage: The stage to activate for the duration of the context.

    Example:

    .. code-block:: python

        >>> import isaacsim.foundation.utils.stage as stage_utils
        >>>
        >>> with stage_utils.use_stage(some_stage):
        ...    # active stage is `some_stage` on this thread
        ...    active_stage = stage_utils.get_active_stage()
        ...
        >>> # previous active stage is restored
    """
    guard = _StageGuard(stage)
    try:
        yield
    finally:
        guard.close()
