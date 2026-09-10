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

"""Tests for ZMQSubscribeUpdatePrimAttribute and ZMQSubscribeEndEffectorCommand OGN nodes."""

import socket as _socket
import time

import omni
import omni.graph.core as og
import omni.kit.test


def find_available_port(start: int = 15800) -> int:
    for port in range(start, start + 100):
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found")


class TestZMQSubscribeUpdatePrimAttribute(omni.kit.test.AsyncTestCase):
    """Tests for OgnZMQSubscribeUpdatePrimAttribute node."""

    async def setUp(self):
        super().setUp()
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self.port = find_available_port()

    async def tearDown(self):
        omni.timeline.get_timeline_interface().stop()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        super().tearDown()

    async def test_node_loads(self):
        """ZMQSubscribeUpdatePrimAttribute can be created in an action graph."""
        og.Controller.edit(
            {"graph_path": "/TestGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                    ("SubscribeCC", "isaacsim.zmq.nodes.ZMQSubscribeUpdatePrimAttribute"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("SubscribeCC.inputs:ip", "localhost"),
                    ("SubscribeCC.inputs:port", self.port),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnImpulse.outputs:execOut", "SubscribeCC.inputs:execIn"),
                ],
            },
        )
        self.assertTrue(og.Controller.node("/TestGraph/SubscribeCC").is_valid())

    async def test_does_not_crash_without_sender(self):
        """Node should not crash when no PUB socket is sending on the configured port."""
        og.Controller.edit(
            {"graph_path": "/TestGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlayback", "omni.graph.action.OnPlaybackTick"),
                    ("SubscribeCC", "isaacsim.zmq.nodes.ZMQSubscribeUpdatePrimAttribute"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("SubscribeCC.inputs:ip", "localhost"),
                    ("SubscribeCC.inputs:port", self.port),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlayback.outputs:tick", "SubscribeCC.inputs:execIn"),
                ],
            },
        )
        omni.timeline.get_timeline_interface().play()
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        # No crash == pass

    async def test_receives_update_prim_attribute_proto(self):
        """Node receives an UpdatePrimAttribute from a ZMQ PUB socket and populates outputs."""
        try:
            import zmq
        except ImportError:
            self.skipTest("zmq Python package not available in test environment")

        try:
            from isaacsim.zmq.protos import update_prim_attribute_pb2
        except ImportError:
            self.skipTest("isaacsim.zmq.core proto modules not available")

        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.bind(f"tcp://127.0.0.1:{self.port}")

        try:
            og.Controller.edit(
                {"graph_path": "/TestGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlayback", "omni.graph.action.OnPlaybackTick"),
                        ("SubscribeCC", "isaacsim.zmq.nodes.ZMQSubscribeUpdatePrimAttribute"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("SubscribeCC.inputs:ip", "localhost"),
                        ("SubscribeCC.inputs:port", self.port),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlayback.outputs:tick", "SubscribeCC.inputs:execIn"),
                    ],
                },
            )

            omni.timeline.get_timeline_interface().play()

            # Tick a few frames so compute() is called and the SUB socket is created/connected
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()
            time.sleep(0.1)

            cmd = update_prim_attribute_pb2.UpdatePrimAttribute()
            cmd.timestamp = 1.0
            cmd.prim_path = "/World/camera/y_link/Camera"
            cmd.attribute = "focalLength"
            cmd.value = 75.0

            # Send repeatedly to overcome the ZMQ slow-joiner problem
            for _ in range(5):
                pub.send_multipart([b"update_prim_attribute", cmd.SerializeToString()])
                time.sleep(0.01)

            # Tick so the node polls and receives the queued message
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            self.assertEqual(og.Controller.attribute("/TestGraph/SubscribeCC.outputs:primPath").get(), cmd.prim_path)
            self.assertEqual(
                og.Controller.attribute("/TestGraph/SubscribeCC.outputs:attributeName").get(), cmd.attribute
            )
            self.assertAlmostEqual(
                og.Controller.attribute("/TestGraph/SubscribeCC.outputs:value").get(), 75.0, places=3
            )

        finally:
            pub.close()
            ctx.term()


class TestZMQSubscribeJointCommand(omni.kit.test.AsyncTestCase):
    """Tests for OgnZMQSubscribeJointCommand node."""

    async def setUp(self):
        super().setUp()
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self.port = find_available_port(start=15900)

    async def tearDown(self):
        omni.timeline.get_timeline_interface().stop()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        super().tearDown()

    async def test_node_loads(self):
        """ZMQSubscribeJointCommand can be created in an action graph."""
        og.Controller.edit(
            {"graph_path": "/TestGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                    ("SubscribeJC", "isaacsim.zmq.nodes.ZMQSubscribeJointCommand"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("SubscribeJC.inputs:ip", "localhost"),
                    ("SubscribeJC.inputs:port", self.port),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnImpulse.outputs:execOut", "SubscribeJC.inputs:execIn"),
                ],
            },
        )
        self.assertTrue(og.Controller.node("/TestGraph/SubscribeJC").is_valid())

    async def test_does_not_crash_without_sender(self):
        """Node should not crash when no PUB socket is sending on the configured port."""
        og.Controller.edit(
            {"graph_path": "/TestGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlayback", "omni.graph.action.OnPlaybackTick"),
                    ("SubscribeJC", "isaacsim.zmq.nodes.ZMQSubscribeJointCommand"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("SubscribeJC.inputs:ip", "localhost"),
                    ("SubscribeJC.inputs:port", self.port),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlayback.outputs:tick", "SubscribeJC.inputs:execIn"),
                ],
            },
        )
        omni.timeline.get_timeline_interface().play()
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        # No crash == pass

    async def test_receives_joint_command_proto(self):
        """Node receives a JointCommand from a ZMQ PUB socket and populates outputs."""
        try:
            import zmq
        except ImportError:
            self.skipTest("zmq Python package not available in test environment")

        try:
            from isaacsim.zmq.protos import joint_command_pb2
        except ImportError:
            self.skipTest("isaacsim.zmq.core proto modules not available")

        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.bind(f"tcp://127.0.0.1:{self.port}")

        try:
            og.Controller.edit(
                {"graph_path": "/TestGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlayback", "omni.graph.action.OnPlaybackTick"),
                        ("SubscribeJC", "isaacsim.zmq.nodes.ZMQSubscribeJointCommand"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("SubscribeJC.inputs:ip", "localhost"),
                        ("SubscribeJC.inputs:port", self.port),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlayback.outputs:tick", "SubscribeJC.inputs:execIn"),
                    ],
                },
            )

            omni.timeline.get_timeline_interface().play()

            # Tick a few frames so compute() is called and the SUB socket is created/connected
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()
            time.sleep(0.1)

            cmd = joint_command_pb2.JointCommand()
            cmd.joint_names.extend(["panda_joint1", "panda_joint2", "panda_joint3"])
            cmd.positions.extend([0.5, -0.3, 0.1])
            cmd.velocities.extend([0.05, -0.03, 0.01])

            # Send repeatedly to overcome the ZMQ slow-joiner problem
            for _ in range(5):
                pub.send_multipart([b"joint_command", cmd.SerializeToString()])
                time.sleep(0.01)

            # Tick so the node polls and receives the queued message
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            positions = og.Controller.attribute("/TestGraph/SubscribeJC.outputs:positions").get()
            self.assertIsNotNone(positions)
            self.assertAlmostEqual(positions[0], 0.5, places=4)
            self.assertAlmostEqual(positions[1], -0.3, places=4)
            self.assertAlmostEqual(positions[2], 0.1, places=4)
            velocities = og.Controller.attribute("/TestGraph/SubscribeJC.outputs:velocities").get()
            self.assertAlmostEqual(velocities[0], 0.05, places=4)
            self.assertAlmostEqual(velocities[1], -0.03, places=4)
            self.assertAlmostEqual(velocities[2], 0.01, places=4)

        finally:
            pub.close()
            ctx.term()
