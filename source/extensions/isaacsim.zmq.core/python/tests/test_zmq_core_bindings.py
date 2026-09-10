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

"""Unit tests for the isaacsim.zmq.core pybind11 bindings (sockets, topics, proto registration)."""

import socket as _socket

import omni.kit.test


def find_available_port(start: int = 15800) -> int:
    """Return the first free TCP port at or above ``start``.

    Args:
        start: Lowest port number to try; a 100-port window is searched from here.

    Returns:
        The first bindable TCP port number.

    Raises:
        RuntimeError: If no free port is found in the search window.
    """
    for port in range(start, start + 100):
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found in range")


class TestZmqCoreImports(omni.kit.test.AsyncTestCase):
    """Verify all public symbols are importable from isaacsim.zmq.core."""

    async def test_zmq_socket_importable(self) -> None:
        """ZmqPublishSocket must be importable from the package root."""
        try:
            from isaacsim.zmq.core import ZmqPublishSocket  # noqa: F401
        except ImportError as exc:
            self.fail(f"Could not import ZmqPublishSocket: {exc}")

    async def test_zmq_subscribe_socket_importable(self) -> None:
        """ZmqSubscribeSocket must be importable from the package root."""
        try:
            from isaacsim.zmq.core import ZmqSubscribeSocket  # noqa: F401
        except ImportError as exc:
            self.fail(f"Could not import ZmqSubscribeSocket: {exc}")


class TestZmqPublishSocket(omni.kit.test.AsyncTestCase):
    """Verify ZmqPublishSocket constructs directly and exposes ip/port (one socket per node, no registry)."""

    async def setUp(self) -> None:
        """Open a fresh stage and reserve an available port for the test."""
        super().setUp()
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self.port = find_available_port()

    async def tearDown(self) -> None:
        """Tick once so the socket is destroyed before the next test."""
        await omni.kit.app.get_app().next_update_async()
        super().tearDown()

    async def test_socket_ip_property(self) -> None:
        """A constructed publish socket reports the ip it was given."""
        from isaacsim.zmq.core import ZmqPublishSocket

        sock = ZmqPublishSocket("localhost", self.port)
        self.assertEqual(sock.ip, "localhost")

    async def test_socket_port_property(self) -> None:
        """A constructed publish socket reports the port it was given."""
        from isaacsim.zmq.core import ZmqPublishSocket

        sock = ZmqPublishSocket("localhost", self.port)
        self.assertEqual(sock.port, self.port)


class TestZmqSubscribeSocket(omni.kit.test.AsyncTestCase):
    """Verify ZmqSubscribeSocket constructs, exposes ip/port/topic, and try_recv is non-blocking."""

    async def setUp(self) -> None:
        """Open a fresh stage and reserve an available port for the test."""
        super().setUp()
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self.port = find_available_port()

    async def tearDown(self) -> None:
        """Tick once so the socket is destroyed before the next test."""
        await omni.kit.app.get_app().next_update_async()
        super().tearDown()

    async def test_subscribe_socket_properties(self) -> None:
        """A constructed subscribe socket reports the ip/port/topic it was given."""
        from isaacsim.zmq.core import ZmqSubscribeSocket

        sock = ZmqSubscribeSocket("localhost", self.port, "update_prim_attribute")
        self.assertEqual(sock.ip, "localhost")
        self.assertEqual(sock.port, self.port)
        self.assertEqual(sock.topic, "update_prim_attribute")

    async def test_try_recv_returns_none_when_no_message(self) -> None:
        """try_recv is non-blocking: with no connected publisher it returns None, never blocks."""
        from isaacsim.zmq.core import ZmqSubscribeSocket

        sock = ZmqSubscribeSocket("localhost", self.port, "update_prim_attribute")
        self.assertIsNone(sock.try_recv())
