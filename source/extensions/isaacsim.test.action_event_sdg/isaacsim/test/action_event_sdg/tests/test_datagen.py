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

import contextlib
import os
import pathlib
import shutil
import tempfile

import carb.tokens
import omni.kit.test
import yaml
from isaacsim.replicator.agent.core import api as IRA
from omni.metropolis.utils.unit_test.stage import TestStage
from pxr import Usd


class TestDataGen(omni.kit.test.AsyncTestCase):
    """Test suite for data generation functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        itoken = carb.tokens.get_tokens_interface()
        self.data_path = pathlib.Path(itoken.resolve("${isaacsim.replicator.agent.core}/data"))

    @contextlib.contextmanager
    def get_tmp_dir(self):
        # WAR for the fact that tempfile.TemporaryDirectory() may not be writable on CI
        if os.environ.get("CI"):
            tmp_dir = os.path.join(os.getcwd(), "_tmp_output")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)
            yield tmp_dir
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                yield tmp_dir

    async def test_data_gen_basic(self):
        """Sanity test for the data generation pipeline with a pre-setup stage."""
        test_stage_path = self.data_path / "test_scenes" / "minimal_cams.usda"
        async with TestStage(stage_path=test_stage_path.as_posix()):
            num_sensors = 0
            stage = omni.usd.get_context().get_stage()
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Cameras")):
                if prim.GetTypeName() == "Camera":
                    num_sensors += 1

            with self.get_tmp_dir() as tmp_dir:
                config_path = os.path.join(tmp_dir, "config.yaml")
                shutil.copyfile(
                    src=(self.data_path / "test_configs" / "test_data_gen.yaml").as_posix(), dst=config_path
                )
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                # Update output_dir in the writers dict (assuming IRABasicWriter is the first/only writer)
                writers = config["isaacsim.replicator.agent"]["replicator"]["writers"]
                # Get the first writer name and update its output_dir
                writer_name = next(iter(writers.keys()))
                config["isaacsim.replicator.agent"]["replicator"]["writers"][writer_name]["output_dir"] = tmp_dir
                with open(config_path, "w") as f:
                    yaml.dump(config, f)

                self.assertTrue(IRA.load_config_file(config_path))
                config = IRA.get_config_file()
                self.assertIsNotNone(config)
                replicator_config = config.isaacsim_replicator_agent.replicator
                self.assertIsNotNone(replicator_config)
                # Get the first writer's parameters from the writers dict
                writer_name = next(iter(replicator_config.writers.keys()))
                writer_entry = replicator_config.writers[writer_name]
                self.assertIsNotNone(writer_entry)
                writer_params = writer_entry.parameters
                self.assertIsNotNone(writer_params)
                output_dir = writer_params.output_dir
                self.assertIsNotNone(output_dir)
                output_dir = pathlib.Path(output_dir)

                # Calculate expected frame count from simulation_duration and start_frame/start_time, end_frame/end_time
                # Fixed FPS is 30 (from ReplicatorBridge._FIXED_FPS)
                FIXED_FPS = 30
                simulation_duration = config.isaacsim_replicator_agent.simulation_duration
                total_frames = int(simulation_duration * FIXED_FPS)

                # Get start_frame/start_time and end_frame/end_time from writer entry
                # Convert to frames using the fixed FPS
                start_frame = writer_entry.get_start_frame()
                end_frame = writer_entry.get_end_frame()
                if end_frame is not None:
                    expected_frame_num = max(0, end_frame - start_frame)
                else:
                    # If end_frame is None, writer continues until simulation ends
                    expected_frame_num = max(0, total_frames - start_frame)

                # we start data generation directly because the stage is pre-setup.
                await IRA.start_data_generation_async(will_wait_until_complete=True)

                # result verification
                output_dir = pathlib.Path(output_dir)
                self.assertTrue(output_dir.exists())
                self.assertTrue(output_dir.is_dir())

                # check sensor output
                def check_output(data_dir: pathlib.Path, ext: str):
                    total_frames = len(list(data_dir.glob(f"*.{ext}")))
                    self.assertEqual(
                        total_frames,
                        expected_frame_num,
                        f"Expected {expected_frame_num} frames, got {total_frames} inside {data_dir}",
                    )

                num_sensor_outputs = 0
                for file in output_dir.iterdir():
                    if file.is_dir():
                        check_output(file / "camera_params", "json")
                        check_output(file / "object_detection", "json")
                        check_output(file / "rgb", writer_params.image_output_format)
                        num_sensor_outputs += 1
                self.assertEqual(num_sensor_outputs, num_sensors)
