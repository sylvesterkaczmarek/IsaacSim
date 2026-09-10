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
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import carb.tokens
import omni.kit.test
import omni.usd
from isaacsim.replicator.agent.core.configuration.models.root import RootConfig
from isaacsim.replicator.agent.core.events import IRAEvents
from isaacsim.replicator.agent.core.simulation import SimulationManager


class TestSimulationManager(omni.kit.test.AsyncTestCase):
    """Test suite for SimulationManager functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        itoken = carb.tokens.get_tokens_interface()
        self.data_path = pathlib.Path(itoken.resolve("${isaacsim.replicator.agent.core}/data"))

    async def setUp(self):
        # Reset singleton before each test
        SimulationManager._instance = None
        self.manager = SimulationManager.get_instance()

    async def tearDown(self):
        self.manager.destroy()
        SimulationManager._instance = None

    def test_singleton(self):
        """Test singleton behavior of SimulationManager."""
        manager1 = SimulationManager()
        manager2 = SimulationManager.get_instance()
        self.assertIs(manager1, manager2)
        self.assertIs(manager1, self.manager)

    @patch("isaacsim.replicator.agent.core.simulation.pipeline_configuration.get_config_file_path")
    @patch("isaacsim.replicator.agent.core.simulation.pipeline_configuration.get_config")
    @patch("isaacsim.replicator.agent.core.simulation.pipeline_configuration.load_config_file")
    def test_load_config_file(self, mock_load_config_file, mock_get_config, mock_get_config_file_path):
        """Test loading configuration from file."""
        mock_config = MagicMock(spec=RootConfig)
        mock_load_config_file.return_value = True
        mock_get_config.return_value = mock_config
        mock_get_config_file_path.return_value = pathlib.Path("dummy/path/config.yaml")

        file_path = "dummy/path/config.yaml"
        result = self.manager.load_config_file(file_path)

        self.assertTrue(result)
        self.assertEqual(self.manager.get_config_file(), mock_config)
        mock_load_config_file.assert_called_once_with(file_path)

    def test_set_config(self):
        """Test setting configuration directly."""
        mock_config = MagicMock(spec=RootConfig)
        mock_config.model_validate.return_value = mock_config

        result = self.manager.set_config(mock_config)

        self.assertTrue(result)
        self.assertEqual(self.manager.get_config_file(), mock_config)

    @contextlib.asynccontextmanager
    async def _mock_stage(self):
        mock_stage = MagicMock()
        with patch("omni.usd.get_context") as mock_context:
            mock_context.return_value.get_stage.return_value = mock_stage
            yield mock_stage

    @patch("isaacsim.replicator.agent.core.simulation.EnvironmentLoader")
    @patch("isaacsim.replicator.agent.core.simulation.CharacterLoader")
    @patch("isaacsim.replicator.agent.core.simulation.RobotLoader")
    @patch("isaacsim.replicator.agent.core.simulation.SensorLoader")
    @patch("isaacsim.replicator.agent.core.simulation.TimelineController")
    @patch("isaacsim.replicator.agent.core.simulation.carb_util.dispatch_event")
    async def test_setup_simulation(
        self, mock_dispatch, mock_timeline, mock_sensor_loader, mock_robot_loader, mock_char_loader, mock_env_loader
    ):
        """Test setup_simulation flow."""
        # Setup mock config
        mock_config = MagicMock(spec=RootConfig)
        mock_config.simulation_duration = 10.0
        mock_config.seed = 42

        # Manually create mock configs since spec=RootConfig might be strict
        mock_env_config = MagicMock()
        mock_env_config.base_stage_asset_path = "dummy_stage.usd"
        mock_config.environment = mock_env_config

        mock_config.character = MagicMock()
        mock_config.robot = MagicMock()
        mock_config.sensor = MagicMock()

        self.manager._config = mock_config

        # Setup loaders mocks
        mock_env_loader_inst = mock_env_loader.return_value
        mock_env_loader_inst.load = AsyncMock(return_value=0)

        mock_char_loader_inst = mock_char_loader.return_value
        mock_char_loader_inst.load = AsyncMock(return_value=0)
        mock_char_loader_inst.get_asset_paths = MagicMock(return_value=set())

        mock_robot_loader_inst = mock_robot_loader.return_value
        mock_robot_loader_inst.load = AsyncMock(return_value=0)

        mock_sensor_loader_inst = mock_sensor_loader.return_value
        mock_sensor_loader_inst.load = AsyncMock(return_value=0)

        # Setup TimelineController mock
        mock_timeline_inst = mock_timeline.return_value
        # Ensure restore is awaitable for tearDown
        mock_timeline_inst.restore = AsyncMock()

        async with self._mock_stage():
            await self.manager.setup_simulation()

        # Verify calls
        mock_env_loader_inst.load.assert_called_once()
        mock_char_loader_inst.load.assert_called_once()
        mock_robot_loader_inst.load.assert_called_once()
        mock_sensor_loader_inst.load.assert_called_once()

        mock_timeline_inst.set_duration.assert_called_with(10.0)

        # Verify event dispatch
        mock_dispatch.assert_called_with(IRAEvents.SET_UP_SIMULATION_DONE_EVENT, payload={"success": True})

    @patch("isaacsim.replicator.agent.core.simulation.ReplicatorBridge")
    @patch("isaacsim.replicator.agent.core.simulation.carb_util.dispatch_event")
    async def test_start_data_generation(self, mock_dispatch, mock_bridge):
        """Test start_data_generation flow."""
        # Setup mock config
        mock_config = MagicMock(spec=RootConfig)
        mock_config.simulation_duration = 5.0
        # Ensure replicator config is present
        mock_config.replicator = MagicMock()
        mock_config.sensor = MagicMock()
        mock_config.sensor.root_prim_path = "/World/Cameras"
        mock_config.robot = MagicMock()  # Add robot mock

        self.manager._config = mock_config

        # Setup bridge mock
        mock_bridge_inst = mock_bridge.return_value
        mock_bridge_inst.run = AsyncMock()

        # Ensure timeline controller is initialized (needed for bridge)
        with patch("isaacsim.replicator.agent.core.simulation.TimelineController") as mock_timeline:
            mock_timeline_inst = mock_timeline.return_value
            # Ensure restore is awaitable for tearDown
            mock_timeline_inst.restore = AsyncMock()

            # Manually assign mock instance because setup_simulation isn't called here
            self.manager._timeline_controller = mock_timeline_inst

            async with self._mock_stage():
                await self.manager.start_data_generation(will_wait_until_complete=True)

        # Verify bridge initialization and run
        mock_bridge.assert_called_once()
        mock_bridge_inst.run.assert_called_with(5.0, True)

        # Verify event dispatch
        # Should verify start and done events
        # Note: We can't strictly check call order easily with just assert_called_with on mock_dispatch directly
        # if it's called multiple times, but we can check if it was called with specific args.
        calls = mock_dispatch.call_args_list
        start_call = ((IRAEvents.DATA_GENERATION_STARTED_EVENT,), {})
        done_call = ((IRAEvents.DATA_GENERATION_DONE_EVENT,), {"payload": {"success": True}})

        self.assertIn(start_call, calls)
        self.assertIn(done_call, calls)

    def test_destroy(self):
        """Test destroy cleans up resources."""
        self.manager._config = MagicMock()
        self.manager._config_path = "some/path"

        # Mock destroyable members
        self.manager._timeline_controller = MagicMock()
        # Ensure restore returns an awaitable (coroutine mock)
        self.manager._timeline_controller.restore = AsyncMock()

        self.manager.destroy()

        self.assertIsNone(self.manager._config)
        self.assertIsNone(self.manager._config_path)
        self.assertIsNone(self.manager._timeline_controller)
        self.assertFalse(SimulationManager.has_instance())
