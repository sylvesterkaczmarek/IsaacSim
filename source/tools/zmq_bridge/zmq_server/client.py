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

from __future__ import annotations

import traceback

import zmq


class ZMQClient:
    """ZMQ client for Isaac Sim: a PULL socket to receive data and a PUB socket to send commands,
    both using two-frame ``[topic, payload]`` multipart messages."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "pub_sockets"):
            return
        self.pub_sockets = {}
        self.pull_sockets = {}
        self.context = None

    def create_context(self) -> None:
        self.context = zmq.Context()

    def create_pull_socket(self, port: int, high_water_mark: int = 1) -> zmq.Socket:
        """Bind a PULL socket on ``port`` to receive ``[topic, payload]`` messages from Isaac Sim
        (``high_water_mark`` caps buffered messages; default 1 keeps only the latest)."""
        socket = self.context.socket(zmq.PULL)
        socket.setsockopt(zmq.RCVHWM, high_water_mark)
        socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1-second timeout
        socket.bind(f"tcp://*:{port}")
        self.pull_sockets[port] = socket
        return socket

    def create_pub_socket(self, port: int) -> zmq.Socket:
        """Bind a PUB socket on ``port`` to send ``[topic, payload]`` commands to Isaac Sim's
        topic-filtered SUB nodes."""
        socket = self.context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1-second timeout
        socket.bind(f"tcp://*:{port}")
        self.pub_sockets[port] = socket
        return socket

    def send_multipart(self, port: int, topic: str, payload: bytes) -> bool:
        """Send ``[topic, payload]`` on the PUB socket. Returns False if the queue is full or errors."""
        try:
            self.pub_sockets[port].send_multipart([topic.encode(), payload], zmq.NOBLOCK)
            return True
        except zmq.error.Again:
            return False
        except Exception:
            traceback.print_exc()
            return False

    def receive_multipart(self, port: int) -> tuple[str, bytes]:
        """Receive ``[topic, payload]`` from the PULL socket; returns ``(topic, payload)`` or ``("", b"")`` on timeout/error."""
        try:
            frames = self.pull_sockets[port].recv_multipart()
            if len(frames) >= 2:
                return frames[0].decode(), frames[1]
            return "", b""
        except zmq.error.Again:
            return "", b""
        except Exception:
            traceback.print_exc()
            return "", b""

    def try_recv(self, port: int) -> tuple[str, bytes]:
        """Non-blocking receive from the PULL socket; returns ``(topic, payload)`` or ``("", b"")`` if none queued."""
        try:
            frames = self.pull_sockets[port].recv_multipart(zmq.NOBLOCK)
            if len(frames) >= 2:
                return frames[0].decode(), frames[1]
            return "", b""
        except zmq.error.Again:
            return "", b""
        except Exception:
            traceback.print_exc()
            return "", b""

    def cleanup(self) -> None:
        """Close all sockets and terminate the ZMQ context."""
        for socket in self.pub_sockets.values():
            socket.close()
        for socket in self.pull_sockets.values():
            socket.close()
        if self.context:
            self.context.term()
