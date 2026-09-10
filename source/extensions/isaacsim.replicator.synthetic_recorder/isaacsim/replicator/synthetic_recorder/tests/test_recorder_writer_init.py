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

"""Tests for SyntheticRecorder writer initialize failures and failed-start cleanup."""

from __future__ import annotations

import os
from unittest.mock import patch

import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
import omni.replicator.core as rep
from isaacsim.replicator.synthetic_recorder.synthetic_recorder import RecorderState, SyntheticRecorder


class SyntheticRecorderTestOutputDirWriter(rep.Writer):
    """Registered test writer that requires output_dir and rejects backend."""

    initialized: bool = False

    def __init__(self, output_dir: str) -> None:
        SyntheticRecorderTestOutputDirWriter.initialized = True
        self.annotators = []

    def write(self, data: dict) -> None:
        """Ignore captured annotator data."""
        return


class SyntheticRecorderTestInitializeNoBackendWriter(rep.Writer):
    """Registered test writer whose initialize() signature rejects backend."""

    initialized: bool = False

    def initialize(self, output_dir: str) -> None:
        SyntheticRecorderTestInitializeNoBackendWriter.initialized = True
        self.annotators = []

    def write(self, data: dict) -> None:
        """Ignore captured annotator data."""
        return


def _printed_messages(mock_print: object) -> list[str]:
    """Collect print() messages from a mock.

    Args:
        mock_print: Mock that replaced the builtin print in the recorder module.

    Returns:
        List of printed string arguments.
    """
    messages = []
    for call in mock_print.call_args_list:
        if call.args:
            messages.append(str(call.args[0]))
    return messages


class TestRecorderWriterInit(omni.kit.test.AsyncTestCase):
    """Test writer initialize kwargs and failed-start behavior."""

    async def setUp(self) -> None:
        """Create a clean stage before each test."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()
        SyntheticRecorderTestOutputDirWriter.initialized = False
        SyntheticRecorderTestInitializeNoBackendWriter.initialized = False

    async def tearDown(self) -> None:
        """Close the test stage and wait for assets to finish loading."""
        stage_utils.close_stage()
        await app_utils.update_app_async()
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

    async def test_start_stop_async_fails_closed_for_unknown_writer(self) -> None:
        """Failed start returns False and does not report a finished write."""
        recorder = SyntheticRecorder()
        recorder.verbose = True
        recorder.writer_name = "NotARegisteredWriter"
        recorder.backend_type = "DiskBackend"
        recorder.backend_params = {"output_dir": os.path.join(os.getcwd(), "_out_sdrec_missing_writer")}
        recorder.rp_data = [["/OmniverseKit_Persp", 64, 64, ""]]

        with patch("isaacsim.replicator.synthetic_recorder.synthetic_recorder.print") as mock_print:
            started = await recorder.start_stop_async()

        self.assertFalse(started)
        self.assertEqual(recorder.get_state(), RecorderState.STOPPED)
        messages = _printed_messages(mock_print)
        joined = "\n".join(messages)
        self.assertTrue(any("Failed to start recording" in message for message in messages), joined)
        self.assertFalse(any("Finished" in message for message in messages), joined)

    async def test_failed_start_restores_capture_on_play(self) -> None:
        """Failed start restores capture-on-play after a partial recorder init."""
        settings = carb.settings.get_settings()
        original = settings.get("/omni/replicator/captureOnPlay")
        settings.set("/omni/replicator/captureOnPlay", True)
        recorder = SyntheticRecorder()
        recorder.writer_name = "BasicWriter"
        recorder.backend_type = None
        recorder.rp_data = [["/OmniverseKit_Persp", 64, 64, ""]]
        try:
            started = await recorder.start_stop_async()
            self.assertFalse(started)
            self.assertTrue(settings.get("/omni/replicator/captureOnPlay"))
        finally:
            settings.set("/omni/replicator/captureOnPlay", original)

    async def test_init_recorder_rejects_writer_without_backend(self) -> None:
        """Refuse writers that do not accept backend instead of passing an unsupported argument."""
        cases = (
            SyntheticRecorderTestOutputDirWriter,
            SyntheticRecorderTestInitializeNoBackendWriter,
        )
        for writer_cls in cases:
            writer_name = writer_cls.__name__
            writer_cls.initialized = False
            rep.WriterRegistry.register(writer_cls)
            recorder = SyntheticRecorder()
            recorder.verbose = True
            recorder.writer_name = writer_name
            recorder.backend_type = "DiskBackend"
            recorder.backend_params = {"output_dir": os.path.join(os.getcwd(), f"_out_sdrec_legacy_{writer_name}")}
            recorder.rp_data = [["/OmniverseKit_Persp", 64, 64, ""]]
            try:
                with patch("isaacsim.replicator.synthetic_recorder.synthetic_recorder.print") as mock_print:
                    started = await recorder.start_stop_async()
                self.assertFalse(started)
                self.assertFalse(writer_cls.initialized)
                self.assertEqual(recorder.get_state(), RecorderState.STOPPED)
                messages = _printed_messages(mock_print)
                joined = "\n".join(messages)
                self.assertTrue(any("must accept a 'backend' argument" in message for message in messages), joined)
                self.assertTrue(any("Failed to start recording" in message for message in messages), joined)
                self.assertFalse(any("Finished" in message for message in messages), joined)
            finally:
                recorder.clear_recorder()
                if writer_name in rep.WriterRegistry.get_writers():
                    rep.WriterRegistry.unregister(writer_name)
