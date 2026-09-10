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


class TestZMQPublishPose(omni.kit.test.AsyncTestCase):
    """Tests for OgnZMQPublishPose node."""

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

    async def test_pose_node_loads(self):
        """Verify OgnZMQPublishPose can be created in an action graph."""
        try:
            og.Controller.edit(
                {"graph_path": "/TestGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                        ("PublishPose", "isaacsim.zmq.nodes.ZMQPublishPose"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishPose.inputs:ip", "localhost"),
                        ("PublishPose.inputs:port", self.port),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnImpulse.outputs:execOut", "PublishPose.inputs:execIn"),
                    ],
                },
            )
        except Exception as exc:
            self.fail(f"Failed to create OgnZMQPublishPose graph: {exc}")

        await omni.kit.app.get_app().next_update_async()

    async def test_pose_node_does_not_crash_without_receiver(self):
        """OgnZMQPublishPose must not crash when no PULL socket is listening."""
        og.Controller.edit(
            {"graph_path": "/TestGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlayback", "omni.graph.action.OnPlaybackTick"),
                    ("PublishPose", "isaacsim.zmq.nodes.ZMQPublishPose"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishPose.inputs:ip", "localhost"),
                    ("PublishPose.inputs:port", self.port),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlayback.outputs:tick", "PublishPose.inputs:execIn"),
                ],
            },
        )

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        timeline.stop()

    async def test_pose_node_sends_proto(self):
        """OgnZMQPublishPose sends a valid Pose protobuf message to a PULL socket."""
        try:
            import zmq
        except ImportError:
            self.skipTest("zmq Python package not available in test environment")

        try:
            from isaacsim.zmq.protos import pose_pb2
        except ImportError:
            self.skipTest("isaacsim.zmq.core proto modules not available")

        ctx = zmq.Context()
        pull = ctx.socket(zmq.PULL)
        pull.setsockopt(zmq.RCVTIMEO, 2000)
        pull.bind(f"tcp://127.0.0.1:{self.port}")

        position = [1.0, 2.0, 3.0]
        # Distinct-per-component quaternion so the assertions below detect any i/j/k/w
        # transposition in the node (an identity quaternion could not — its imaginary
        # parts are all zero). Input order is IJKR (x, y, z, w).
        orientation = [0.1, 0.2, 0.3, 0.4]

        try:
            og.Controller.edit(
                {"graph_path": "/TestGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                        ("PublishPose", "isaacsim.zmq.nodes.ZMQPublishPose"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishPose.inputs:ip", "localhost"),
                        ("PublishPose.inputs:port", self.port),
                        ("PublishPose.inputs:timeStamp", 3.0),
                        ("PublishPose.inputs:position", position),
                        ("PublishPose.inputs:orientation", orientation),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnImpulse.outputs:execOut", "PublishPose.inputs:execIn"),
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
                self.fail("Timed out waiting for ZMQ pose message")

            self.assertEqual(len(frames), 2, "Expected 2-frame multipart [topic, payload]")
            self.assertEqual(frames[0].decode(), "pose", "First frame should be topic string")

            msg = pose_pb2.Pose()
            msg.ParseFromString(frames[1])
            self.assertAlmostEqual(msg.timestamp, 3.0, places=3)
            self.assertEqual(len(msg.position), 3)
            self.assertAlmostEqual(msg.position[0], 1.0, places=4)
            self.assertAlmostEqual(msg.position[1], 2.0, places=4)
            self.assertAlmostEqual(msg.position[2], 3.0, places=4)
            # Wire order is (x, y, z, w): imaginary i,j,k then real w. Assert each
            # component lands in its slot so a transposition would fail this test.
            self.assertEqual(len(msg.orientation), 4)
            self.assertAlmostEqual(msg.orientation[0], 0.1, places=4)  # x (imaginary i)
            self.assertAlmostEqual(msg.orientation[1], 0.2, places=4)  # y (imaginary j)
            self.assertAlmostEqual(msg.orientation[2], 0.3, places=4)  # z (imaginary k)
            self.assertAlmostEqual(msg.orientation[3], 0.4, places=4)  # w (real)

        finally:
            pull.close()
            ctx.term()
