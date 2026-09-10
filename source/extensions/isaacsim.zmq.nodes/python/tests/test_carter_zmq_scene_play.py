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

"""Load-and-play smoke test for the published Carter warehouse ZMQ scene USD.

Opens the scene from the Isaac assets server
(``<assets_root>/Isaac/Samples/ZeroMQ/carter_warehouse_zmq.usd``), plays a few frames, and
asserts the bridge graphs + nodes are present and nothing raises. This catches authoring /
graph-topology regressions (node types resolve, graph builds, CameraHelper computes, timeline
advances) without needing the external teleop client.

Set ``ISAAC_ZMQ_SCENE_USD`` to a local .usd file or an ``omniverse://`` URL to test that
instead. The test is skipped if the scene cannot be reached (e.g. it is deployed to a dev
Nucleus but this environment's assets root points elsewhere).
"""

import os

import omni
import omni.graph.core as og
import omni.kit.test

_SEND_GRAPH = "/World/ZMQSendGraph"
_RECV_GRAPH = "/World/ZMQRecvGraph"

# Published location on the Isaac assets server (relative to the assets root).
_SCENE_ASSET_RELPATH = "/Isaac/Samples/ZeroMQ/carter_warehouse_zmq.usd"


def _resolve_scene_usd() -> str | None:
    """Resolve the scene USD: an explicit local override, else the Nucleus assets path."""
    env = os.environ.get("ISAAC_ZMQ_SCENE_USD")
    if env:  # explicit override: a local path or an omniverse:// URL
        return env
    try:
        from isaacsim.storage.native import get_assets_root_path  # noqa: PLC0415
    except ImportError:
        return None
    assets_root = get_assets_root_path()
    if not assets_root:
        return None
    return assets_root + _SCENE_ASSET_RELPATH


class TestCarterZmqScenePlay(omni.kit.test.AsyncTestCase):
    """Open + play the authored Carter warehouse ZMQ scene and verify it runs."""

    async def setUp(self):
        super().setUp()
        self._usd = _resolve_scene_usd()

    async def tearDown(self):
        omni.timeline.get_timeline_interface().stop()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        super().tearDown()

    async def test_scene_loads_and_plays(self):
        if not self._usd:
            self.skipTest(
                "scene USD unavailable; set ISAAC_ZMQ_SCENE_USD to a local file or omniverse:// URL, or ensure the "
                f"Isaac assets root is reachable (expected <assets_root>{_SCENE_ASSET_RELPATH})"
            )

        result, error = await omni.usd.get_context().open_stage_async(self._usd)
        if not result:
            # The scene is an external asset; if it is not deployed to this environment's
            # assets root (e.g. the public production server vs. a dev Nucleus), skip rather
            # than fail. It validates wherever the scene is reachable.
            self.skipTest(f"scene USD not reachable at '{self._usd}': {error}")
        await omni.kit.app.get_app().next_update_async()

        # The publish/subscribe graphs must have persisted in the saved USD.
        for graph_path, nodes in (
            (_SEND_GRAPH, ("CreateRP", "CameraHelper", "PubClock", "ReadPose", "PubPose")),
            (_RECV_GRAPH, ("SubJointCmd", "DriveCtrl")),
        ):
            self.assertIsNotNone(og.Controller.graph(graph_path), f"missing graph {graph_path}")
            for n in nodes:
                node = og.Controller.node(f"{graph_path}/{n}")
                self.assertTrue(node.is_valid(), f"{graph_path}/{n} missing/invalid after load")

        # Play several frames; the graphs (incl. CameraHelper bootstrapping its writers)
        # must advance without raising.
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()
        timeline.stop()
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()
