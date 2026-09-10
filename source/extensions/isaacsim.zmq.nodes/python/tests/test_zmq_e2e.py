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

"""End-to-end test for the ZMQ bridge control loop used by the Carter example.

Exercises the full ZMQ round-trip a standalone client uses:

    Isaac Sim (ZMQPublishPose) ──────PUSH──► mock client PULL socket
                                              │
                                              │ (client computes commands)
                                              ▼
    Isaac Sim (ZMQSubscribeJointCommand) ◄──SUB── mock client PUB socket

No real robot or camera is required.  The mock client receives the published
pose and responds with a JointCommand (wheel velocity targets).
"""

import socket as _socket
import time

import omni
import omni.graph.core as og
import omni.kit.test


def find_available_port(start: int = 16100) -> int:
    for port in range(start, start + 200):
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found in range")


# Example configuration used across tests (a differential-drive base)
_POSITION = [1.0, -2.0, 0.1]
_ORIENTATION = [0.0, 0.0, 0.0, 1.0]
_JOINT_NAMES = ["joint_wheel_left", "joint_wheel_right"]
_JOINT_VELOCITIES = [0.5, -0.5]


class TestZmqE2E(omni.kit.test.AsyncTestCase):
    """End-to-end tests for the ZMQ bridge pose-out / joint-command-in loop."""

    async def setUp(self):
        super().setUp()
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self.pub_port = find_available_port(start=16100)  # Isaac Sim publishes poses here
        self.sub_port = find_available_port(start=self.pub_port + 1)  # Isaac Sim subscribes for joint commands here

    async def tearDown(self):
        omni.timeline.get_timeline_interface().stop()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        super().tearDown()

    async def test_full_round_trip(self):
        """Full loop: pose published by Isaac Sim, answered with a joint command.

        Simulates a standalone client's control loop:
          1. Isaac Sim publishes a prim pose via ZMQPublishPose.
          2. The mock client (this test) receives the pose.
          3. The mock client responds with a JointCommand via a PUB socket.
          4. Isaac Sim receives the joint command via ZMQSubscribeJointCommand.
        """
        try:
            import zmq
        except ImportError:
            self.skipTest("zmq Python package not available")

        try:
            from isaacsim.zmq.protos import joint_command_pb2, pose_pb2
        except ImportError:
            self.skipTest("isaacsim.zmq.core proto modules not available")

        ctx = zmq.Context()
        # PULL socket: receive Pose from Isaac Sim
        pull = ctx.socket(zmq.PULL)
        pull.setsockopt(zmq.RCVTIMEO, 3000)
        pull.bind(f"tcp://127.0.0.1:{self.pub_port}")
        # PUB socket: send JointCommand to Isaac Sim
        pub = ctx.socket(zmq.PUB)
        pub.bind(f"tcp://127.0.0.1:{self.sub_port}")

        try:
            og.Controller.edit(
                {"graph_path": "/TestGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlayback", "omni.graph.action.OnPlaybackTick"),
                        ("PublishPose", "isaacsim.zmq.nodes.ZMQPublishPose"),
                        ("SubscribeJC", "isaacsim.zmq.nodes.ZMQSubscribeJointCommand"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishPose.inputs:ip", "localhost"),
                        ("PublishPose.inputs:port", self.pub_port),
                        ("PublishPose.inputs:position", _POSITION),
                        ("PublishPose.inputs:orientation", _ORIENTATION),
                        ("SubscribeJC.inputs:ip", "localhost"),
                        ("SubscribeJC.inputs:port", self.sub_port),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlayback.outputs:tick", "PublishPose.inputs:execIn"),
                        ("OnPlayback.outputs:tick", "SubscribeJC.inputs:execIn"),
                    ],
                },
            )

            omni.timeline.get_timeline_interface().play()

            # Allow both sockets to connect before exchanging data
            for _ in range(20):
                await omni.kit.app.get_app().next_update_async()
            time.sleep(0.1)

            # Step 1: receive the pose published by Isaac Sim
            try:
                frames = pull.recv_multipart()
            except zmq.Again:
                self.fail("Timed out waiting for Pose from Isaac Sim")

            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].decode(), "pose")
            pose_msg = pose_pb2.Pose()
            pose_msg.ParseFromString(frames[1])
            for i, pos in enumerate(_POSITION):
                self.assertAlmostEqual(pose_msg.position[i], pos, places=4, msg=f"position[{i}] mismatch")

            # Step 2: respond with a joint command (mock client)
            cmd = joint_command_pb2.JointCommand()
            cmd.joint_names.extend(_JOINT_NAMES)
            cmd.velocities.extend(_JOINT_VELOCITIES)
            payload = cmd.SerializeToString()

            # Send several times to overcome the SUB slow-joiner race
            for _ in range(5):
                pub.send_multipart([b"joint_command", payload])
                await omni.kit.app.get_app().next_update_async()

            # Step 3: let the subscribe node receive the command
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            # Step 4: verify Isaac Sim received the joint command
            sub_node = og.Controller.node("/TestGraph/SubscribeJC")
            received_names = list(og.Controller.get(sub_node.get_attribute("outputs:jointNames")))
            received_velocities = list(og.Controller.get(sub_node.get_attribute("outputs:velocities")))

            self.assertEqual(received_names, _JOINT_NAMES)
            for i, vel in enumerate(_JOINT_VELOCITIES):
                self.assertAlmostEqual(received_velocities[i], vel, places=4, msg=f"round-trip velocity[{i}] mismatch")

        finally:
            pull.close()
            pub.close()
            ctx.term()
