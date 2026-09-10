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
import unittest

import omni
import omni.graph.core as og
import omni.kit.test


def find_available_port(start: int = 15600) -> int:
    """Find an available TCP port starting from start."""
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found")


class TestZMQPublishClock(omni.kit.test.AsyncTestCase):
    """Tests for OgnZMQPublishClock node."""

    async def setUp(self):
        super().setUp()
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self.port = find_available_port()

    async def tearDown(self):
        # Stop timeline and clean up graph
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        super().tearDown()

    async def test_clock_node_loads(self):
        """Verify OgnZMQPublishClock can be created in an action graph."""
        try:
            og.Controller.edit(
                {"graph_path": "/TestGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                        ("PublishClock", "isaacsim.zmq.nodes.ZMQPublishClock"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishClock.inputs:ip", "localhost"),
                        ("PublishClock.inputs:port", self.port),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnImpulse.outputs:execOut", "PublishClock.inputs:execIn"),
                    ],
                },
            )
        except Exception as exc:
            self.fail(f"Failed to create OgnZMQPublishClock graph: {exc}")

        await omni.kit.app.get_app().next_update_async()

    async def test_clock_node_does_not_crash_without_receiver(self):
        """OgnZMQPublishClock must not crash when no PULL socket is listening.

        ZMQ PUSH with HWM=1 and DONTWAIT silently drops frames — no exception.
        """
        og.Controller.edit(
            {"graph_path": "/TestGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlayback", "omni.graph.action.OnPlaybackTick"),
                    ("PublishClock", "isaacsim.zmq.nodes.ZMQPublishClock"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("ReadSysTime", "isaacsim.core.nodes.IsaacReadSystemTime"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishClock.inputs:ip", "localhost"),
                    ("PublishClock.inputs:port", self.port),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlayback.outputs:tick", "PublishClock.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:simulationTime"),
                    ("ReadSysTime.outputs:systemTime", "PublishClock.inputs:systemTime"),
                ],
            },
        )

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        # Run for several frames — must not raise
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()

        timeline.stop()

    async def test_clock_node_sends_proto(self):
        """OgnZMQPublishClock sends a valid Clock protobuf message to a PULL socket."""
        try:
            import zmq
        except ImportError:
            self.skipTest("zmq Python package not available in test environment")

        try:
            from isaacsim.zmq.protos import clock_pb2
        except ImportError:
            self.skipTest("isaacsim.zmq.core proto modules not available")

        ctx = zmq.Context()
        pull = ctx.socket(zmq.PULL)
        pull.setsockopt(zmq.RCVTIMEO, 2000)  # 2 s timeout
        pull.bind(f"tcp://127.0.0.1:{self.port}")

        try:
            og.Controller.edit(
                {"graph_path": "/TestGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                        ("PublishClock", "isaacsim.zmq.nodes.ZMQPublishClock"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("ReadSysTime", "isaacsim.core.nodes.IsaacReadSystemTime"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishClock.inputs:ip", "localhost"),
                        ("PublishClock.inputs:port", self.port),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnImpulse.outputs:execOut", "PublishClock.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:simulationTime"),
                        ("ReadSysTime.outputs:systemTime", "PublishClock.inputs:systemTime"),
                    ],
                },
            )

            timeline = omni.timeline.get_timeline_interface()
            timeline.play()

            # Trigger impulse
            og.Controller.attribute("/TestGraph/OnImpulse.state:enableImpulse").set(True)
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            try:
                frames = pull.recv_multipart()
            except zmq.Again:
                self.fail("Timed out waiting for ZMQ clock message")

            self.assertEqual(len(frames), 2, "Expected 2-frame multipart [topic, payload]")
            self.assertEqual(frames[0].decode(), "clock", "First frame should be topic string")

            msg = clock_pb2.Clock()
            msg.ParseFromString(frames[1])
            self.assertGreaterEqual(msg.sim_time, 0.0)
            self.assertGreaterEqual(msg.sys_time, 0.0)

        finally:
            pull.close()
            ctx.term()
