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

"""Test the Python logging facade."""

from __future__ import annotations

import inspect
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import isaacsim.common.logging as logging
import pytest


def test_public_api_exposes_loggers_and_application_configuration() -> None:
    """Verify the public API inventory and severity members."""
    assert logging.__all__ == [
        "ChannelLoggingConfig",
        "ChannelSettingBehavior",
        "ConfigureResult",
        "ElapsedTimeUnit",
        "GlobalLoggingConfig",
        "LogLevel",
        "Logger",
        "OutputStream",
        "configure_global",
        "flush",
    ]
    assert set(logging.LogLevel.__members__) == {"VERBOSE", "INFO", "WARNING", "ERROR"}


def test_logger_requires_and_exposes_an_immutable_channel() -> None:
    """Verify that logger channels are non-empty and immutable."""
    logger = logging.Logger("isaacsim.python.test")
    assert logger.channel == "isaacsim.python.test"
    assert logger.is_enabled(logging.LogLevel.WARNING)

    with pytest.raises(AttributeError):
        setattr(logger, "channel", "replacement")
    with pytest.raises(ValueError):
        logging.Logger("")


def test_log_and_convenience_methods_submit_records() -> None:
    """Verify that each Python logging method submits a record."""
    logger = logging.Logger("isaacsim.python.methods")
    logger.log(logging.LogLevel.INFO, "Generic")
    logger.verbose("Verbose")
    logger.info("Info")
    logger.warning("Warning")
    logger.warn("Warning alias")
    logger.error("Error")
    logger.warning("Temporary Unicode message: " + "π" * 8)
    logging.flush()


def test_report_writes_and_flushes_native_standard_output() -> None:
    """Verify that reports write to and flush native standard output."""
    script = "\n".join(
        [
            "import isaacsim.common.logging as logging",
            'logger = logging.Logger("isaacsim.python.report")',
            'logger.report("Calculation result: 42")',
        ]
    )
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    assert "[isaacsim.python.report] Calculation result: 42" in result.stdout.splitlines()


def test_python_source_capture_is_disabled_by_default_and_can_be_enabled(tmp_path: Path) -> None:
    """Verify that Python caller metadata is attached only when explicitly requested.

    Args:
        tmp_path: Temporary directory supplied by pytest.
    """
    log_path = tmp_path / "source-capture.log"
    config = logging.GlobalLoggingConfig()
    config.file_path = str(log_path)
    config.file_append = False
    config.minimum_level = logging.LogLevel.INFO
    config.file_level = logging.LogLevel.INFO
    config.file_flush_level = logging.LogLevel.WARNING
    config.filename_included = True
    config.function_name_included = True
    config.line_number_included = True
    config.color_included = False
    assert logging.configure_global(config) is logging.ConfigureResult.SUCCESS

    logger = logging.Logger("isaacsim.python.source_capture")
    default_marker = "Python source capture disabled"
    captured_marker = "Python source capture enabled"
    report_marker = "Python report source capture enabled"
    logger.warning(default_marker)
    frame = inspect.currentframe()
    assert frame is not None
    expected_line = frame.f_lineno + 1
    logger.warning(captured_marker, capture_source=True)
    frame = inspect.currentframe()
    assert frame is not None
    expected_report_line = frame.f_lineno + 1
    logger.report(report_marker, capture_source=True)
    logging.flush()

    restore = logging.GlobalLoggingConfig()
    restore.file_path = ""
    restore.minimum_level = logging.LogLevel.WARNING
    restore.filename_included = False
    restore.function_name_included = False
    restore.line_number_included = False
    assert logging.configure_global(restore) is logging.ConfigureResult.SUCCESS

    lines = log_path.read_text(encoding="utf-8").splitlines()
    default_line = next(line for line in lines if default_marker in line)
    captured_line = next(line for line in lines if captured_marker in line)
    captured_report_line = next(line for line in lines if report_marker in line)
    assert "test_logging.py" not in default_line
    assert "test_python_source_capture_is_disabled_by_default_and_can_be_enabled()" not in default_line
    assert "test_logging.py" in captured_line
    assert "test_python_source_capture_is_disabled_by_default_and_can_be_enabled()" in captured_line
    assert f":{expected_line}:" in captured_line
    assert "test_logging.py" in captured_report_line
    assert "test_python_source_capture_is_disabled_by_default_and_can_be_enabled()" in captured_report_line
    assert f":{expected_report_line}:" in captured_report_line


def test_removed_dispatch_configuration_is_not_exposed() -> None:
    """Verify that removed callback and logger policy APIs stay hidden."""
    logger = logging.Logger("isaacsim.python.surface")
    for name in ("destroy", "flush", "set_log_callback", "set_log_level"):
        assert not hasattr(logger, name)
    for name in ("DEFAULT", "NONE"):
        assert not hasattr(logging.LogLevel, name)


def test_application_configuration_writes_and_filters_a_log_file(tmp_path: Path) -> None:
    """Verify file output and process-wide channel filtering.

    Args:
        tmp_path: Temporary directory supplied by pytest.
    """
    log_path = tmp_path / "application.log"
    global_config = logging.GlobalLoggingConfig()
    global_config.minimum_level = logging.LogLevel.VERBOSE
    global_config.file_path = str(log_path)
    global_config.file_append = False
    global_config.file_level = logging.LogLevel.VERBOSE
    global_config.file_flush_level = logging.LogLevel.VERBOSE
    global_config.color_included = False

    threshold_channel = logging.ChannelLoggingConfig()
    threshold_channel.channel = "isaacsim.python.file.threshold"
    threshold_channel.minimum_level_behavior = logging.ChannelSettingBehavior.OVERRIDE
    threshold_channel.minimum_level = logging.LogLevel.ERROR

    disabled_channel = logging.ChannelLoggingConfig()
    disabled_channel.channel = "isaacsim.python.file.disabled"
    disabled_channel.enabled_behavior = logging.ChannelSettingBehavior.OVERRIDE
    disabled_channel.enabled = False
    global_config.channels = [threshold_channel, disabled_channel]
    assert logging.configure_global(global_config) is logging.ConfigureResult.SUCCESS

    threshold_logger = logging.Logger(threshold_channel.channel)
    threshold_logger.warning("Filtered by Python channel threshold")
    threshold_logger.error("Included by Python channel threshold")
    disabled_logger = logging.Logger(disabled_channel.channel)
    disabled_logger.error("Filtered by disabled Python channel")
    inherited_logger = logging.Logger("isaacsim.python.file.inherited")
    inherited_logger.info("Included by Python global policy")
    logging.flush()

    threshold_channel.minimum_level_behavior = logging.ChannelSettingBehavior.INHERIT
    disabled_channel.enabled_behavior = logging.ChannelSettingBehavior.INHERIT
    restore_channels = logging.GlobalLoggingConfig()
    restore_channels.channels = [threshold_channel, disabled_channel]
    assert logging.configure_global(restore_channels) is logging.ConfigureResult.SUCCESS
    threshold_logger.warning("Included after Python threshold inheritance")
    disabled_logger.error("Included after Python enablement inheritance")
    logging.flush()

    disable_file = logging.GlobalLoggingConfig()
    disable_file.file_path = ""
    disable_file.minimum_level = logging.LogLevel.WARNING
    assert logging.configure_global(disable_file) is logging.ConfigureResult.SUCCESS
    contents = log_path.read_text(encoding="utf-8")
    assert "Included by Python channel threshold" in contents
    assert "Included by Python global policy" in contents
    assert "Included after Python threshold inheritance" in contents
    assert "Included after Python enablement inheritance" in contents
    assert "Filtered by Python channel threshold" not in contents
    assert "Filtered by disabled Python channel" not in contents


def test_global_configuration_is_a_patch() -> None:
    """Verify that unset configuration fields remain unchanged."""
    config = logging.GlobalLoggingConfig()
    assert config.enabled is None
    assert config.file_path is None
    assert config.channels == []
    config.elapsed_time = logging.ElapsedTimeUnit.MILLISECONDS
    assert logging.configure_global(config) is logging.ConfigureResult.SUCCESS


def test_invalid_channel_configuration_is_rejected() -> None:
    """Verify that an empty overridden channel is rejected."""
    channel = logging.ChannelLoggingConfig()
    channel.enabled_behavior = logging.ChannelSettingBehavior.OVERRIDE
    channel.enabled = False
    config = logging.GlobalLoggingConfig()
    config.channels = [channel]
    assert logging.configure_global(config) is logging.ConfigureResult.INVALID_ARGUMENT


def test_concurrent_configuration_snapshots_mutable_python_input() -> None:
    """Verify that concurrent configuration snapshots mutable input safely."""
    config = logging.GlobalLoggingConfig()

    def apply(index: int) -> logging.ConfigureResult:
        config.elapsed_time = (
            logging.ElapsedTimeUnit.MILLISECONDS if index % 2 == 0 else logging.ElapsedTimeUnit.MICROSECONDS
        )
        return logging.configure_global(config)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(apply, range(100)))
    assert all(result is logging.ConfigureResult.SUCCESS for result in results)

    restore = logging.GlobalLoggingConfig()
    restore.elapsed_time = logging.ElapsedTimeUnit.MILLISECONDS
    assert logging.configure_global(restore) is logging.ConfigureResult.SUCCESS


def test_concurrent_submission_is_safe() -> None:
    """Verify that multiple Python threads can submit records safely."""
    logger = logging.Logger("isaacsim.python.threading")

    def produce(thread_index: int) -> None:
        for record_index in range(100):
            logger.info(f"Thread {thread_index} record {record_index}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(produce, range(4)))
    logging.flush()
