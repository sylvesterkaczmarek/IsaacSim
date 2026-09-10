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

"""Verify Python warnings use Carb warning severity during Newton USD parsing."""

from __future__ import annotations

import io
import warnings
from contextlib import redirect_stderr
from unittest import mock

import omni.kit.test
from isaacsim.physics.newton.impl._warning_logging import _log_python_warnings


class TestWarningLogging(omni.kit.test.AsyncTestCase):
    """Test Newton Python warning routing."""

    async def test_log_python_warnings_uses_carb_warning(self) -> None:
        """Route a Python warning to Carb without writing to standard error."""
        standard_error = io.StringIO()

        with (
            warnings.catch_warnings(),
            mock.patch("isaacsim.physics.newton.impl._warning_logging.carb.log_warn") as log_warn,
            redirect_stderr(standard_error),
        ):
            warnings.simplefilter("always")
            with _log_python_warnings():
                warnings.warn("Eigenvalues below threshold detected", UserWarning)

        log_warn.assert_called_once()
        logged_message = log_warn.call_args.args[0]
        self.assertIn("UserWarning: Eigenvalues below threshold detected", logged_message)
        self.assertEqual(standard_error.getvalue(), "")

    async def test_log_python_warnings_logs_before_propagating_exception(self) -> None:
        """Log captured warnings when the protected operation raises an exception."""
        with (
            warnings.catch_warnings(),
            mock.patch("isaacsim.physics.newton.impl._warning_logging.carb.log_warn") as log_warn,
        ):
            warnings.simplefilter("always")
            with self.assertRaisesRegex(RuntimeError, "Parsing failed"):
                with _log_python_warnings():
                    warnings.warn("Parser warning", UserWarning)
                    raise RuntimeError("Parsing failed")

        log_warn.assert_called_once()
        self.assertIn("UserWarning: Parser warning", log_warn.call_args.args[0])
