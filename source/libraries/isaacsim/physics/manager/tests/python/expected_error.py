# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Provide a context manager for tests that intentionally emit diagnostics."""

from __future__ import annotations

import sys
from types import TracebackType

_preface_printed = False


class ExpectedError:
    """Prefix diagnostic output that a test expects to emit."""

    def __enter__(self) -> None:
        """Print the expected-diagnostic prefix before entering the context."""
        global _preface_printed
        if not _preface_printed:
            print(
                "[warn] Test(s) are running that expect errors and/or warnings",
                file=sys.stderr,
                flush=True,
            )
            _preface_printed = True

        # Preflush any output, otherwise it may be appended to the next statement.
        print("", flush=True)
        print("[Ignore this error/warning] ", end="", flush=False)

    def __exit__(
        self,
        exit_type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Terminate the prefixed diagnostic line when leaving the context.

        Args:
            exit_type: Exception type raised in the context, if any.
            value: Exception raised in the context, if any.
            traceback: Traceback associated with the exception, if any.

        """
        del exit_type, value, traceback
        print("", flush=True)
