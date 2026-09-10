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

"""Python warning logging helpers for Newton physics."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import carb


@contextmanager
def _log_python_warnings() -> Iterator[None]:
    """Route Python warnings through Carb at warning severity."""
    captured_warnings = []
    try:
        with warnings.catch_warnings(record=True) as captured_warnings:
            yield
    finally:
        for warning in captured_warnings:
            message = warnings.formatwarning(
                warning.message,
                warning.category,
                warning.filename,
                warning.lineno,
                line=warning.line,
            ).rstrip()
            carb.log_warn(f"[Newton] {message}")
