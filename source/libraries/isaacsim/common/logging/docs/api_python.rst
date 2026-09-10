..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. _isaacsim-common-logging-api-python:

==============================
Python guide and API reference
==============================

The Python API exposes the native logger and process-wide application configuration through nanobind. Logger methods
return ``None`` and delegate filtering, rendering, destinations, and asynchronous delivery to Carbonite.

.. _isaacsim-common-logging-python-summary:

Summary
========

.. currentmodule:: isaacsim.common.logging

.. autosummary::
    :nosignatures:

    LogLevel
    Logger
    ChannelLoggingConfig
    ChannelSettingBehavior
    GlobalLoggingConfig
    ConfigureResult
    OutputStream
    ElapsedTimeUnit
    configure_global
    flush

.. _isaacsim-common-logging-python-severity:

Severity
========

.. autoclass:: isaacsim.common.logging.LogLevel
    :members:

.. _isaacsim-common-logging-python-logger:

Logger
======

Construct ``Logger`` with one non-empty channel, then submit already formatted messages with ``log``, ``verbose``,
``info``, ``warning``, or ``error``. ``warn`` is a compatibility alias for ``warning``. Python source capture is
disabled by default. Construction caches the channel without selecting a backend, allowing a native host to attach
after feature modules are imported. Pass the keyword-only argument ``capture_source=True`` to attach the immediate
Python caller's filename, function name, and line number to an individual submission. Calls through a Python wrapper
identify the wrapper until a stack-level API is added.

.. code-block:: python

    logger.warning("Using the fallback backend", capture_source=True)

Use ``report`` to synchronously print and flush ``[channel] message`` to standard output and also submit the unprefixed
message to Carbonite at ``INFO``. The standard-output write ignores Carbonite's level, while the Carbonite record
remains subject to its configured policy and may cause duplicate console output. The channel is immutable and the
logger needs no explicit destruction. Use ``is_enabled`` before constructing an expensive message. Filtered calls
skip source capture even when requested.

.. autoclass:: isaacsim.common.logging.Logger
    :members:
    :undoc-members:

.. _isaacsim-common-logging-python-configuration:

Application-level configuration
================================

Configure global logging only from the application entry point. Every property defaults to ``None``, which leaves the
corresponding Carbonite setting unchanged. Set ``file_path`` to an empty string to disable file logging. The
``channels`` list is empty by default.

.. code-block:: python

    from isaacsim.common.logging import GlobalLoggingConfig, LogLevel, configure_global

    config = GlobalLoggingConfig()
    config.file_path = "logs/isaac-sim-${pid}.log"
    config.file_append = True
    config.file_level = LogLevel.INFO
    result = configure_global(config)

To disable one channel while leaving the process-wide policy enabled, add a channel patch:

.. code-block:: python

    from isaacsim.common.logging import ChannelLoggingConfig, ChannelSettingBehavior

    channel = ChannelLoggingConfig()
    channel.channel = "isaacsim.sensors.camera"
    channel.enabled_behavior = ChannelSettingBehavior.OVERRIDE
    channel.enabled = False

    config = GlobalLoggingConfig()
    config.channels = [channel]
    configure_global(config)

Set ``enabled_behavior`` or ``minimum_level_behavior`` to ``INHERIT`` to clear that override and resume using the
process-wide setting. ``UNCHANGED`` leaves the corresponding Carbonite setting untouched. Carbonite performs the
filtering.

Create parent directories before configuring a file destination. Carbonite resolves relative file paths from the
working directory at the time of the call and does not report file-open failures through its configuration interface.

.. autoclass:: isaacsim.common.logging.GlobalLoggingConfig
    :members:
    :undoc-members:

.. autoclass:: isaacsim.common.logging.ChannelLoggingConfig
    :members:
    :undoc-members:

.. autoclass:: isaacsim.common.logging.ChannelSettingBehavior
    :members:

.. autoclass:: isaacsim.common.logging.ConfigureResult
    :members:

.. autoclass:: isaacsim.common.logging.OutputStream
    :members:

.. autoclass:: isaacsim.common.logging.ElapsedTimeUnit
    :members:

.. autofunction:: isaacsim.common.logging.configure_global

.. _isaacsim-common-logging-python-flush:

Flush
=====

.. autofunction:: isaacsim.common.logging.flush

``flush`` drains Carbonite's process-wide backend; it is not scoped to one logger.
