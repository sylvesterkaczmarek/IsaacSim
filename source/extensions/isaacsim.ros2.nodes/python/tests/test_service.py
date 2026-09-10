# SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Verifies generic ROS 2 service OmniGraph node behavior and service field type handling."""

import ctypes
import importlib
import json
import sys
from typing import Any

import omni.graph.core as og
import omni.kit.test
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.ros2.core.impl.ros2_test_case import ROS2TestCase

# Each test case exercises writeNodeAttributeFromMessage for specific primitive types
# by creating a service client/server pair and verifying that request fields are correctly
# written to the server's output attributes using the full prefixed path.
#
# Types covered:
#   eBool   - std_srvs/SetBool          (data: bool)
#   eInt    - nav_msgs/GetPlan           (start:header:stamp:sec: int32)
#   eUInt   - nav_msgs/GetPlan           (start:header:stamp:nanosec: uint32)
#   eInt64  - example_interfaces/AddTwoInts (a, b: int64)
#   eUInt64 - rcl_interfaces/ListParameters (depth: uint64)
#   eFloat  - nav_msgs/GetPlan           (tolerance: float32)
#   eDouble - nav_msgs/GetPlan           (start:pose:position:x: float64)
#   eToken   - nav_msgs/GetPlan          (start:header:frame_id: string)
#   eUnknown - isaac_ros2_messages/IsaacPose (poses: Pose[])
#
# Not covered (no standard service exposes these as flat request fields):
#   eUChar   - uint8 scalars/arrays
SERVICE_FIELD_TYPE_CASES = [
    # (label, package, subfolder, message, fields_to_test)
    # fields_to_test: list of (field_path, test_value, is_float)
    (
        "AddTwoInts",
        "example_interfaces",
        "srv",
        "AddTwoInts",
        [
            ("a", 42, False),
            ("b", -7, False),
        ],
    ),
    (
        "SetBool",
        "std_srvs",
        "srv",
        "SetBool",
        [
            ("data", True, False),
        ],
    ),
    (
        "GetPlan",
        "nav_msgs",
        "srv",
        "GetPlan",
        [
            ("tolerance", 3.14, True),
            ("start:header:stamp:sec", 42, False),
            ("start:header:stamp:nanosec", 123456, False),
            ("start:header:frame_id", "test_frame", False),
            ("start:pose:position:x", 1.5, True),
            ("start:pose:position:y", -2.7, True),
        ],
    ),
    (
        "ListParameters",
        "rcl_interfaces",
        "srv",
        "ListParameters",
        [
            ("depth", 5, False),
        ],
    ),
]


def _load_ros2_library(stem: str) -> Any:
    """Load a ROS 2 native support library if it is available to the test process."""
    if sys.platform == "win32":
        candidates = [stem + ".dll"]
    elif sys.platform == "darwin":
        candidates = ["lib" + stem + ".dylib", stem + ".dylib"]
    else:
        candidates = ["lib" + stem + ".so", stem + ".so"]

    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    return None


def _get_native_service_support_error(package: str, subfolder: str, message: str) -> str | None:
    """Return why the native C service support is unavailable, or None when usable."""
    try:
        importlib.import_module(f"{package}.{subfolder}")
    except (ImportError, ModuleNotFoundError) as exc:
        return f"{package}.{subfolder} is not available in the ROS 2 Python environment: {exc}"

    generator_library = _load_ros2_library(package + "__rosidl_generator_c")
    if generator_library is None:
        return f"{package} native generator support is not available"

    type_support_library = _load_ros2_library(package + "__rosidl_typesupport_c")
    if type_support_library is None:
        return f"{package} native type support is not available"

    for suffix in ("Request", "Response"):
        symbol = f"{package}__{subfolder}__{message}_{suffix}__create"
        if not hasattr(generator_library, symbol):
            return f"{package}/{subfolder}/{message} native generator symbol {symbol} is not available"

    type_support_symbol = f"rosidl_typesupport_c__get_service_type_support_handle__{package}__{subfolder}__{message}"
    if not hasattr(type_support_library, type_support_symbol):
        return f"{package}/{subfolder}/{message} native type support symbol {type_support_symbol} is not available"

    return None


class TestRos2Service(ROS2TestCase):
    """Verify generic ROS 2 service OmniGraph node request handling."""

    async def setUp(self) -> None:
        """Create a fresh stage for generic service graph tests."""
        await super().setUp()
        await stage_utils.create_new_stage_async()

    async def tearDown(self) -> None:
        """Run shared ROS 2 cleanup after generic service graph tests."""
        await super().tearDown()

    @staticmethod
    def _has_attributes(node: Any, attribute_names: list[str]) -> bool:
        """Return whether all expected dynamic service attributes were created."""
        return all(node.get_attribute_exists(attribute_name) for attribute_name in attribute_names)

    def _create_service_graph(
        self, graph_path: Any, service_name: Any, package: Any, subfolder: Any, message: Any
    ) -> Any:
        """Create a service client/server action graph.

        Args:
            graph_path: OmniGraph path to create.
            service_name: ROS 2 service name.
            package: Message package name.
            subfolder: Message package subfolder.
            message: Service message name.

        Returns:
            Created graph, server request node, and client node.
        """
        test_graph, new_nodes, _, _ = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ServerRequest", "isaacsim.ros2.bridge.OgnROS2ServiceServerRequest"),
                    ("ServerResponse", "isaacsim.ros2.bridge.OgnROS2ServiceServerResponse"),
                    ("Client", "isaacsim.ros2.bridge.OgnROS2ServiceClient"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ServerRequest.inputs:serviceName", service_name),
                    ("ServerRequest.inputs:messagePackage", package),
                    ("ServerRequest.inputs:messageSubfolder", subfolder),
                    ("ServerRequest.inputs:messageName", message),
                    ("ServerResponse.inputs:messagePackage", package),
                    ("ServerResponse.inputs:messageSubfolder", subfolder),
                    ("ServerResponse.inputs:messageName", message),
                    ("Client.inputs:serviceName", service_name),
                    ("Client.inputs:messagePackage", package),
                    ("Client.inputs:messageSubfolder", subfolder),
                    ("Client.inputs:messageName", message),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "Client.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ServerRequest.inputs:execIn"),
                    ("ServerRequest.outputs:onReceived", "ServerResponse.inputs:onReceived"),
                    ("ServerRequest.outputs:serverHandle", "ServerResponse.inputs:serverHandle"),
                ],
            },
        )
        server_req_node = new_nodes[1]
        server_res_node = new_nodes[2]
        client_node = new_nodes[3]
        return test_graph, server_req_node, server_res_node, client_node

    # ----------------------------------------------------------------------
    async def test_client_does_not_fire_without_response(self) -> None:
        """Do not signal a response when no service server exists."""
        native_support_error = _get_native_service_support_error("example_interfaces", "srv", "AddTwoInts")
        if native_support_error is not None:
            self.skipTest(native_support_error)

        graph_path = "/ServiceClientNoResponse"
        _, new_nodes, _, _ = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("Impulse", "omni.graph.action.OnImpulseEvent"),
                    ("Client", "isaacsim.ros2.bridge.OgnROS2ServiceClient"),
                    ("Counter", "omni.graph.action.Counter"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("Client.inputs:serviceName", "/test_no_service_server"),
                    ("Client.inputs:messagePackage", "example_interfaces"),
                    ("Client.inputs:messageSubfolder", "srv"),
                    ("Client.inputs:messageName", "AddTwoInts"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("Impulse.outputs:execOut", "Client.inputs:execIn"),
                    ("Client.outputs:execOut", "Counter.inputs:execIn"),
                ],
            },
        )
        impulse_node, _, counter_node = new_nodes

        self._timeline.play()
        try:
            og.Controller.attribute("state:enableImpulse", impulse_node).set(True)
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            before = og.Controller.attribute("outputs:count", counter_node).get()
            og.Controller.attribute("state:enableImpulse", impulse_node).set(True)
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            self.assertEqual(og.Controller.attribute("outputs:count", counter_node).get(), before)
        finally:
            self._timeline.stop()

    # ----------------------------------------------------------------------
    async def test_service(self) -> None:
        """Test service."""
        native_support_error = _get_native_service_support_error("example_interfaces", "srv", "AddTwoInts")
        if native_support_error is not None:
            self.skipTest(native_support_error)

        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()

        test_graph, server_req_node, server_res_node, client_node = self._create_service_graph(
            "/ActionGraph", "/custom_service", "example_interfaces", "srv", "AddTwoInts"
        )

        await og.Controller.evaluate(test_graph)
        await omni.kit.app.get_app().next_update_async()

        if (
            not self._has_attributes(client_node, ["inputs:Request:a", "inputs:Request:b", "outputs:Response:sum"])
            or not self._has_attributes(server_req_node, ["outputs:Request:a", "outputs:Request:b"])
            or not self._has_attributes(server_res_node, ["inputs:Response:sum"])
        ):
            self._timeline.stop()
            self.skipTest("example_interfaces/srv/AddTwoInts is not available in the ROS 2 environment")

        og.Controller.attribute("inputs:Request:a", client_node).set(11)
        og.Controller.attribute("inputs:Request:b", client_node).set(10)

        # wait for the client to executes and send the request
        await omni.kit.app.get_app().next_update_async()
        await og.Controller.evaluate(test_graph)
        a = og.Controller.attribute("outputs:Request:a", server_req_node).get()
        b = og.Controller.attribute("outputs:Request:b", server_req_node).get()

        await omni.kit.app.get_app().next_update_async()
        server_result = a + b
        og.Controller.attribute("inputs:Response:sum", server_res_node).set(server_result)

        # wait for the server to executes and send the response
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        client_result = og.Controller.attribute("outputs:Response:sum", client_node).get()
        print("server response = ", server_result)
        print("client response = ", client_result)
        self.assertEqual(client_result, 21)
        self._timeline.stop()

    # ----------------------------------------------------------------------
    async def test_nested_message_array(self) -> None:
        """Preserve nested message arrays in service requests."""
        package = "isaac_ros2_messages"
        message = "IsaacPose"
        native_support_error = _get_native_service_support_error(package, "srv", message)
        if native_support_error is not None:
            self.skipTest(native_support_error)

        graph, server_request, _, client = self._create_service_graph(
            "/NestedMessageArrayGraph", "/nested_message_array", package, "srv", message
        )
        self._timeline.play()
        try:
            await og.Controller.evaluate(graph)
            await omni.kit.app.get_app().next_update_async()

            pose = {
                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                "orientation": {"x": 0.0, "y": 0.7071, "z": 0.0, "w": 0.7071},
            }
            og.Controller.attribute("inputs:Request:poses", client).set([json.dumps(pose)])

            received = []
            for _ in range(20):
                await omni.kit.app.get_app().next_update_async()
                received = og.Controller.attribute("outputs:Request:poses", server_request).get()
                if received:
                    break

            self.assertTrue(received)
            self.assertEqual(json.loads(str(received[0])), pose)
        finally:
            self._timeline.stop()

    # ----------------------------------------------------------------------
    async def test_service_field_types(self) -> None:
        """Verify writeNodeAttributeFromMessage correctly prefixes attribute paths for all types.

        Uses multiple ROS 2 service types whose request fields collectively exercise every
        primitive data type in the writeNodeAttributeFromMessage switch statement.
        For each service, sets values on the client's request inputs, evaluates the graph,
        and checks that the server's request outputs received the correct values.
        """
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()

        ran_any = False
        for idx, (label, package, subfolder, message, fields) in enumerate(SERVICE_FIELD_TYPE_CASES):
            native_support_error = _get_native_service_support_error(package, subfolder, message)
            if native_support_error is not None:
                print(f"Skipping {label}: {native_support_error}")
                continue

            service_name = f"/test_field_types_{idx}"
            graph_path = f"/ActionGraph_types_{idx}"

            test_graph, server_req_node, _, client_node = self._create_service_graph(
                graph_path, service_name, package, subfolder, message
            )

            await og.Controller.evaluate(test_graph)
            await omni.kit.app.get_app().next_update_async()

            missing_attributes = [
                field_path
                for field_path, _, _ in fields
                if not client_node.get_attribute_exists(f"inputs:Request:{field_path}")
                or not server_req_node.get_attribute_exists(f"outputs:Request:{field_path}")
            ]
            if missing_attributes:
                print(
                    f"Skipping {label}: {package}/{subfolder}/{message} dynamic attributes not available: "
                    f"{missing_attributes}"
                )
                continue

            ran_any = True

            for field_path, value, _ in fields:
                og.Controller.attribute(f"inputs:Request:{field_path}", client_node).set(value)

            await omni.kit.app.get_app().next_update_async()
            await og.Controller.evaluate(test_graph)

            for field_path, expected, is_float in fields:
                actual = og.Controller.attribute(f"outputs:Request:{field_path}", server_req_node).get()
                if is_float:
                    self.assertAlmostEqual(
                        actual,
                        expected,
                        places=2,
                        msg=f"[{label}] field '{field_path}': expected {expected}, got {actual}",
                    )
                elif isinstance(expected, bool):
                    self.assertEqual(
                        actual,
                        expected,
                        msg=f"[{label}] field '{field_path}': expected {expected}, got {actual}",
                    )
                else:
                    self.assertEqual(
                        actual,
                        expected,
                        msg=f"[{label}] field '{field_path}': expected {expected}, got {actual}",
                    )

        self.assertTrue(ran_any, "No ROS 2 service packages available")
        self._timeline.stop()
