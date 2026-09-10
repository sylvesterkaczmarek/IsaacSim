// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <isaacsim/zmq/core/ZmqPublishSocket.hpp>
#include <isaacsim/zmq/core/ZmqSubscribeSocket.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_isaacsim_zmq_core, m)
{
    m.doc() = "Isaac Sim ZMQ Core Python bindings";

    // Expose socket classes
    py::class_<isaacsim::zmq::core::ZmqPublishSocket, std::shared_ptr<isaacsim::zmq::core::ZmqPublishSocket>>(
        m, "ZmqPublishSocket")
        .def(py::init<const std::string&, uint16_t>(), py::arg("ip"), py::arg("port"),
             "Construct and connect a PUSH socket to tcp://ip:port")
        .def(
            "send_multipart",
            [](isaacsim::zmq::core::ZmqPublishSocket& self, const std::string& topic, const py::bytes& payload)
            {
                std::string payloadStr = payload;
                return self.sendMultipart(topic, payloadStr);
            },
            py::arg("topic"), py::arg("payload"), "Send [topic, payload] as a two-frame ZMQ multipart message")
        .def_property_readonly("ip", &isaacsim::zmq::core::ZmqPublishSocket::getIp)
        .def_property_readonly("port", &isaacsim::zmq::core::ZmqPublishSocket::getPort);

    py::class_<isaacsim::zmq::core::ZmqSubscribeSocket, std::shared_ptr<isaacsim::zmq::core::ZmqSubscribeSocket>>(
        m, "ZmqSubscribeSocket")
        .def(py::init<const std::string&, uint16_t, const std::string&>(), py::arg("ip"), py::arg("port"),
             py::arg("topic"), "Construct and connect a SUB socket to tcp://ip:port subscribed to topic")
        .def_property_readonly("ip", &isaacsim::zmq::core::ZmqSubscribeSocket::getIp)
        .def_property_readonly("port", &isaacsim::zmq::core::ZmqSubscribeSocket::getPort)
        .def_property_readonly("topic", &isaacsim::zmq::core::ZmqSubscribeSocket::getTopic)
        .def(
            "try_recv",
            [](isaacsim::zmq::core::ZmqSubscribeSocket& self) -> py::object
            {
                std::string payload;
                if (self.tryRecv(payload))
                {
                    return py::bytes(payload);
                }
                return py::none();
            },
            "Non-blocking receive; returns the payload as bytes, or None if no message is available");
}
