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

from unittest.mock import patch

import numpy as np
import omni.kit.test
from isaacsim.replicator.agent.core.data_generation.writers.writer import IRABasicWriter
from isaacsim.replicator.agent.core.data_generation.writers.writer_utils import WriterSetting, WriterUtils


class TestWriterUtils(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_extract_prefix_and_remainder(self):
        prefix_list = ["prefix1_", "prefix2_"]

        # Test match
        prefix, remainder = WriterUtils.extract_prefix_and_remainder("prefix1_test", prefix_list)
        self.assertEqual(prefix, "prefix1_")
        self.assertEqual(remainder, "test")

        # Test no match
        prefix, remainder = WriterUtils.extract_prefix_and_remainder("other_test", prefix_list)
        self.assertIsNone(prefix)
        self.assertEqual(remainder, "other_test")

    async def test_select_elements_based_on_prefix(self):
        list_a = ["prefix1_a", "prefix2_b", "other_c"]
        list_b = ["prefix1_", "prefix2_"]

        selected = WriterUtils.select_elements_based_on_prefix(list_a, list_b)
        self.assertEqual(len(selected), 2)
        self.assertIn("prefix1_a", selected)
        self.assertIn("prefix2_b", selected)

    async def test_numpy_encoder(self):
        # Test numpy types
        self.assertEqual(WriterUtils.numpy_encoder(np.int32(1)), 1)

        # Verify list conversion
        arr_list = WriterUtils.numpy_encoder(np.array([1, 2, 3]))
        self.assertEqual(arr_list, [1, 2, 3])

        # Test void type (structured array element)
        dt = np.dtype([("f1", np.int32), ("f2", np.float64)])
        arr = np.array([(1, 2.5)], dtype=dt)
        encoded = WriterUtils.numpy_encoder(arr[0])
        self.assertIsInstance(encoded, dict)
        self.assertEqual(encoded["f1"], 1)
        self.assertEqual(encoded["f2"], 2.5)

    async def test_convert_to_serialized_dict(self):
        dt = np.dtype([("f1", np.int32), ("f2", np.float64)])
        arr = np.array([(10, 20.5)], dtype=dt)

        # Test the helper directly
        serialized = WriterUtils.convert_to_serialized_dict(arr[0])
        self.assertEqual(serialized["f1"], 10)
        self.assertEqual(serialized["f2"], 20.5)


class TestIRABasicWriter(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_extract_object_detection_annotator(self):
        # Test extraction using known prefixes from WriterSetting
        # WriterSetting.AnnotatorPrefix.ObjectDetection.GENERIC = "object_info"

        classification, name = IRABasicWriter.extract_object_detection_annotator("object_info_bounding_box_2d")
        self.assertEqual(classification, WriterSetting.AnnotatorPrefix.ObjectDetection.GENERIC)
        self.assertEqual(name, "bounding_box_2d")

        classification, name = IRABasicWriter.extract_object_detection_annotator("agent_info_skeleton")
        self.assertEqual(classification, WriterSetting.AnnotatorPrefix.ObjectDetection.AGENT_SPECIFIC)
        self.assertEqual(name, "skeleton")

        classification, name = IRABasicWriter.extract_object_detection_annotator("unknown_param")
        self.assertIsNone(classification)
        self.assertEqual(name, "unknown_param")

    async def test_get_all_write_functions(self):
        # Mock class to test get_all_write_functions logic
        class MockWriter(IRABasicWriter):
            def _write_test_annotator(self):
                pass

            def _write_another_one(self):
                pass

            def other_method(self):
                pass

        mapping = MockWriter.get_all_write_functions()
        self.assertIn("test_annotator", mapping)
        self.assertEqual(mapping["test_annotator"], "_write_test_annotator")
        self.assertIn("another_one", mapping)
        self.assertEqual(mapping["another_one"], "_write_another_one")
        self.assertNotIn("other_method", mapping)

    async def test_init_initialization(self):
        # Mock dependencies to test __init__
        with (
            patch(
                "isaacsim.replicator.agent.core.data_generation.writers.writer.BasicWriter.__init__"
            ) as mock_super_init,
            patch("isaacsim.replicator.agent.core.data_generation.writers.writer.ObjectInfoManager") as mock_manager,
            patch(
                "isaacsim.replicator.agent.core.data_generation.writers.writer.carb.settings.get_settings"
            ) as mock_settings,
            patch.object(IRABasicWriter, "initialize_writer") as mock_init_writer,
            patch.object(IRABasicWriter, "_set_up_actor_data_store") as mock_setup_store,
        ):

            mock_settings.return_value.get.return_value = 0
            mock_init_writer.return_value = {"output_dir": "/tmp"}

            writer = IRABasicWriter(output_dir="/tmp", custom_arg="value")

            # Verify super init called
            self.assertTrue(mock_super_init.called)

            # Verify ObjectInfoManager created
            self.assertTrue(mock_manager.called)

            # Verify attributes set
            self.assertEqual(writer.output_format, "JSON")
            self.assertEqual(writer.data_structure, "renderProduct")

            # Verify internal setup called
            self.assertTrue(mock_setup_store.called)
