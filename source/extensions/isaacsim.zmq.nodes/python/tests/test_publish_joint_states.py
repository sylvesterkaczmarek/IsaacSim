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

import socket

import omni
import omni.graph.core as og
import omni.kit.test


def find_available_port(start: int = 16100) -> int:
    """Find an available TCP port starting from start."""
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found")


class TestZMQPublishJointStates(omni.kit.test.AsyncTestCase):
    """Tests for OgnZMQPublishJointStates node."""

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

    async def test_joint_states_node_loads(self):
        """Verify OgnZMQPublishJointStates can be created in an action graph."""
        try:
            og.Controller.edit(
                {"graph_path": "/TestGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                        ("PublishJointStates", "isaacsim.zmq.nodes.ZMQPublishJointStates"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishJointStates.inputs:ip", "localhost"),
                        ("PublishJointStates.inputs:port", self.port),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnImpulse.outputs:execOut", "PublishJointStates.inputs:execIn"),
                    ],
                },
            )
        except Exception as exc:
            self.fail(f"Failed to create OgnZMQPublishJointStates graph: {exc}")

        await omni.kit.app.get_app().next_update_async()

    async def test_joint_states_node_does_not_crash_without_receiver(self):
        """OgnZMQPublishJointStates must not crash when no PULL socket is listening."""
        og.Controller.edit(
            {"graph_path": "/TestGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlayback", "omni.graph.action.OnPlaybackTick"),
                    ("PublishJointStates", "isaacsim.zmq.nodes.ZMQPublishJointStates"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishJointStates.inputs:ip", "localhost"),
                    ("PublishJointStates.inputs:port", self.port),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlayback.outputs:tick", "PublishJointStates.inputs:execIn"),
                ],
            },
        )

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        timeline.stop()

    async def test_joint_states_node_sends_proto(self):
        """OgnZMQPublishJointStates sends a valid JointStates protobuf message to a PULL socket."""
        try:
            import zmq
        except ImportError:
            self.skipTest("zmq Python package not available in test environment")

        try:
            from isaacsim.zmq.protos import joint_states_pb2
        except ImportError:
            self.skipTest("isaacsim.zmq.core proto modules not available")

        ctx = zmq.Context()
        pull = ctx.socket(zmq.PULL)
        pull.setsockopt(zmq.RCVTIMEO, 2000)
        pull.bind(f"tcp://127.0.0.1:{self.port}")

        joint_names = ["joint_wheel_left", "joint_wheel_right"]
        positions = [0.5, -0.5]
        velocities = [1.5, -1.5]

        try:
            og.Controller.edit(
                {"graph_path": "/TestGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                        ("PublishJointStates", "isaacsim.zmq.nodes.ZMQPublishJointStates"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishJointStates.inputs:ip", "localhost"),
                        ("PublishJointStates.inputs:port", self.port),
                        ("PublishJointStates.inputs:timestamp", 4.0),
                        ("PublishJointStates.inputs:jointNames", joint_names),
                        ("PublishJointStates.inputs:positions", positions),
                        ("PublishJointStates.inputs:velocities", velocities),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnImpulse.outputs:execOut", "PublishJointStates.inputs:execIn"),
                    ],
                },
            )

            timeline = omni.timeline.get_timeline_interface()
            timeline.play()

            og.Controller.attribute("/TestGraph/OnImpulse.state:enableImpulse").set(True)
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            try:
                frames = pull.recv_multipart()
            except zmq.Again:
                self.fail("Timed out waiting for ZMQ joint states message")

            self.assertEqual(len(frames), 2, "Expected 2-frame multipart [topic, payload]")
            self.assertEqual(frames[0].decode(), "joint_states", "First frame should be topic string")

            msg = joint_states_pb2.JointStates()
            msg.ParseFromString(frames[1])
            self.assertAlmostEqual(msg.timestamp, 4.0, places=3)
            self.assertEqual(list(msg.joint_names), joint_names)
            self.assertEqual(len(msg.positions), 2)
            self.assertAlmostEqual(msg.positions[0], 0.5, places=4)
            self.assertAlmostEqual(msg.positions[1], -0.5, places=4)
            self.assertEqual(len(msg.velocities), 2)
            self.assertAlmostEqual(msg.velocities[0], 1.5, places=4)
            self.assertAlmostEqual(msg.velocities[1], -1.5, places=4)

        finally:
            pull.close()
            ctx.term()
