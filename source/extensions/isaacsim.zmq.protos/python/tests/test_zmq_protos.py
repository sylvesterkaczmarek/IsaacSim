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

"""Unit tests for the generated protobuf Python modules shipped with isaacsim.zmq.protos.

Each ``.proto`` is compiled by protoc at build time into ``<name>_pb2.py`` alongside the
package (``isaacsim/zmq/protos/``). These tests assert every module imports, exposes its
message classes, and that the generated code is functional (construct + serialize round-trip).
"""

import importlib

import omni.kit.test

# (module suffix, [message class names]) for every proto in the extension.
PROTO_MODULES = [
    ("clock_pb2", ["Clock"]),
    ("image_pb2", ["Image", "GpuIpcImage", "GpuIpcArray"]),
    ("pose_pb2", ["Pose"]),
    ("joint_states_pb2", ["JointStates"]),
    ("joint_command_pb2", ["JointCommand"]),
    ("update_prim_attribute_pb2", ["UpdatePrimAttribute"]),
]

# Every message class name re-exported from the isaacsim.zmq.protos package root.
PROTO_MESSAGE_NAMES = [name for _, names in PROTO_MODULES for name in names]


class TestZmqProtosImports(omni.kit.test.AsyncTestCase):
    """Verify every generated proto module imports and exposes its message classes."""

    async def test_proto_modules_importable(self) -> None:
        """Each ``<name>_pb2`` module must import from the isaacsim.zmq.protos package."""
        for module_suffix, _ in PROTO_MODULES:
            qualified = f"isaacsim.zmq.protos.{module_suffix}"
            try:
                importlib.import_module(qualified)
            except ImportError as exc:
                self.fail(f"Could not import {qualified}: {exc}")

    async def test_proto_message_classes_present(self) -> None:
        """Each module must expose its declared message classes, and they must instantiate."""
        for module_suffix, message_names in PROTO_MODULES:
            module = importlib.import_module(f"isaacsim.zmq.protos.{module_suffix}")
            for name in message_names:
                self.assertTrue(
                    hasattr(module, name),
                    f"{module_suffix} is missing message class {name}",
                )
                # Generated message classes must be constructible with no arguments.
                getattr(module, name)()

    async def test_message_classes_reexported_from_package(self) -> None:
        """Every message class is importable directly from the package root.

        i.e. ``from isaacsim.zmq.protos import UpdatePrimAttribute`` works, not just
        ``from isaacsim.zmq.protos.update_prim_attribute_pb2 import UpdatePrimAttribute``.
        """
        import isaacsim.zmq.protos as zmq_protos

        for name in PROTO_MESSAGE_NAMES:
            self.assertTrue(
                hasattr(zmq_protos, name),
                f"isaacsim.zmq.protos does not re-export message class {name}",
            )
            self.assertIn(name, zmq_protos.__all__, f"{name} missing from isaacsim.zmq.protos.__all__")
            # The re-exported symbol must be the constructible message class.
            getattr(zmq_protos, name)()


class TestZmqProtosRoundTrip(omni.kit.test.AsyncTestCase):
    """Verify generated message classes serialize and parse, including new fields and oneofs."""

    async def test_update_prim_attribute_round_trip(self) -> None:
        """UpdatePrimAttribute fields survive a serialize/parse round-trip."""
        from isaacsim.zmq.protos import UpdatePrimAttribute

        cmd = UpdatePrimAttribute(
            timestamp=12.5, prim_path="/World/camera/y_link/Camera", attribute="focalLength", value=24.0
        )
        parsed = UpdatePrimAttribute()
        parsed.ParseFromString(cmd.SerializeToString())

        self.assertAlmostEqual(parsed.timestamp, 12.5)
        self.assertEqual(parsed.prim_path, "/World/camera/y_link/Camera")
        self.assertEqual(parsed.attribute, "focalLength")
        self.assertAlmostEqual(parsed.value, 24.0)

    async def test_image_pixel_transport_oneof(self) -> None:
        """Image.pixel_transport is a oneof: setting one arm clears the other."""
        from isaacsim.zmq.protos import Image

        img = Image()
        img.data = b"raw-pixels"
        self.assertEqual(img.WhichOneof("pixel_transport"), "data")

        img.gpu.ipc_event_handle = b"evt"
        self.assertEqual(img.WhichOneof("pixel_transport"), "gpu")
        self.assertEqual(img.gpu.ipc_event_handle, b"evt")
