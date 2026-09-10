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

"""Implement backend-selection utilities."""

__all__ = [
    "get_current_backend",
    "is_backend_set",
    "should_raise_on_fallback",
    "should_raise_on_unsupported",
    "use_backend",
]

import contextlib
from collections.abc import Generator

from ..bindings._bindings import (
    _BackendGuard,
)
from ..bindings._bindings import get_current_backend as _get_current_backend
from ..bindings._bindings import (
    is_backend_set,
    should_raise_on_fallback,
    should_raise_on_unsupported,
)


@contextlib.contextmanager
def use_backend(
    backend: str,
    *,
    raise_on_unsupported: bool = False,
    raise_on_fallback: bool = False,
) -> Generator[None, None, None]:
    """Context manager that sets a thread-local backend value.

    Args:
        backend: The value to set in the context.
        raise_on_unsupported: Whether to raise an exception if the backend is not supported when requested.
        raise_on_fallback: Whether to raise an exception if the backend is supported,
            but a fallback is being used at a particular point in time when requested.

    Example:

    .. code-block:: python

        >>> import isaacsim.foundation.utils.backend as backend_utils
        >>>
        >>> with backend_utils.use_backend("usd"):
        ...    # operate on the specified backend
        ...    pass
        >>> # operate on the default backend
    """
    guard = _BackendGuard(backend, raise_on_unsupported, raise_on_fallback)
    try:
        yield
    finally:
        guard.close()


def get_current_backend(supported_backends: list[str], *, raise_on_unsupported: bool | None = None) -> str:
    """Get the current backend value if it exists.

    Args:
        supported_backends: The list of supported backends.
        raise_on_unsupported: Whether to raise an error if the backend is not supported when requested.
            If set to a value other than ``None``, this parameter has precedence over the context value.

    Returns:
        The current backend value or the default value (first supported backend) if no backend is active.
    """
    return _get_current_backend(supported_backends, raise_on_unsupported)
