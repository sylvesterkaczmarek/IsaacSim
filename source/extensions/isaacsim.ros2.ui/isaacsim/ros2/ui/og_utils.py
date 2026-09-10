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

"""Utility functions for creating ROS 2 OmniGraph shortcuts."""

import isaacsim.core.experimental.utils.prim as prim_utils
import omni.ui as ui
import OmniGraphSchema
from isaacsim.gui.components import dropdown_builder
from isaacsim.gui.components.widgets import ParamWidget, SelectPrimWidget
from isaacsim.ros2.nodes import (
    Ros2ClockGraphConfig,
    Ros2GenericPublisherGraphConfig,
    Ros2JointStatesGraphConfig,
    Ros2OdometryGraphConfig,
    Ros2TfGraphConfig,
    create_ros2_clock_graph,
    create_ros2_generic_publisher_graph,
    create_ros2_joint_states_graph,
    create_ros2_odometry_graph,
    create_ros2_tf_graph,
)
from omni.kit.notification_manager import NotificationStatus, post_notification
from omni.kit.window.extensions import SimpleCheckBox

from .og_common import (
    Ros2GraphWindow,
    add_ok_cancel_buttons,
    add_script_docs_footer,
    validate_existing_graph_path,
    validate_new_graph_path,
)


class Ros2ClockGraph(Ros2GraphWindow):
    """A UI window for creating ROS2 Clock OmniGraph.

    This class provides a dialog interface that allows users to configure and generate an OmniGraph for publishing
    ROS2 clock messages. The graph includes nodes for reading simulation time and publishing it to the ROS2 ``/clock``
    topic. The graph is triggered on playback tick and can be configured with a custom graph path.

    The generated graph includes the following nodes:

    - ``OnPlaybackTick``: Triggers execution on each simulation tick.
    - ``ReadSimTime``: Reads the current simulation time.
    - ``PublishClock``: Publishes the time to the ROS2 ``/clock`` topic.
    - ``Context``: Provides the ROS2 context for communication.

    The UI provides fields for specifying the graph path and includes buttons to access documentation and view the
    Python script used for graph generation.
    """

    def __init__(self) -> None:
        super().__init__("ROS2 Clock Graph", width=300, height=150)
        self._og_path = "/Graph/ROS_Clock"

        # build UI
        self._build_ui()

    def make_graph(self) -> None:
        """Create and configures an OmniGraph for ROS2 clock publishing.

        Stops the timeline and constructs a graph containing nodes for playback tick, simulation time reading,
        clock publishing, and ROS2 context. The graph is set up to publish simulation time as ROS2 clock messages.
        """
        try:
            self._og_path = create_ros2_clock_graph(Ros2ClockGraphConfig(graph_path=self._og_path))
        except (ValueError, RuntimeError) as exc:
            post_notification(str(exc), status=NotificationStatus.WARNING)

    def _build_ui(self) -> None:
        """Construct the UI elements for the ROS2 clock graph configuration window.

        Creates input fields for graph path, OK and Cancel buttons, and documentation links for the ROS2 clock
        graph generation interface.
        """
        default_og_path = "/Graph/ROS_Clock"
        og_path_def = ParamWidget.FieldDef(
            name="og_path", label="Graph Path", type=ui.StringField, default=default_og_path
        )

        with self.frame:
            with ui.VStack(spacing=4):
                self.og_path_input = ParamWidget(field_def=og_path_def)
                add_ok_cancel_buttons(self._on_ok, self._on_cancel)
                add_script_docs_footer(
                    __file__,
                    "https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_clock.html#graph-shortcut",
                )

        return

    def _on_ok(self) -> None:
        """Handle the OK button click event.

        Validates user inputs, creates the ROS2 clock graph if parameters are valid, and closes the window.
        Displays a warning notification if parameter validation fails.
        """
        self._og_path = self.og_path_input.get_value()

        self._finish_on_ok()

    def _check_params(self) -> bool:
        """Validate the graph path parameter.

        Checks if a graph already exists at the specified path and displays a warning if it does.

        Returns:
            False if a graph already exists at the specified path, True otherwise.

        """
        return validate_new_graph_path(self._og_path, "clock graph")


class Ros2GenericPubGraph(Ros2GraphWindow):
    """A UI window for creating ROS2 generic publisher graphs in Isaac Sim.

    This window provides a user interface to generate OmniGraph graphs that publish various message types to ROS2 topics.
    Users can select from predefined publisher templates including:

    - Float32 publisher that publishes the real-time factor (RTF)
    - Bool publisher with a constant boolean value
    - Int64 publisher with a constant integer value
    - String publisher with a constant string message

    The window allows users to specify the graph path and choose the type of publisher to create. Each template creates
    a complete graph with the necessary nodes including tick, context, and publisher nodes, automatically configured
    with appropriate connections.

    The generated graphs use the ``isaacsim.ros2.bridge.ROS2Publisher`` node to publish messages of the selected type.
    For the RTF publisher, the graph includes an ``isaacsim.core.nodes.IsaacRealTimeFactor`` node that computes the
    real-time factor during simulation.

    The UI includes buttons to access the Python script used for graph generation and links to relevant documentation.
    """

    def __init__(self) -> None:
        super().__init__("ROS2 Generic Publisher Graph", width=350, height=180)
        self._og_path = "/Graph/ROS_GenericPub"
        self._dropdown_model = None
        self._dropdown_operations_list = [
            ("Publish RTF as Float32", self.make_rtf_graph),
            ("Publish Bool", self.make_bool_graph),
            ("Publish Int64", self.make_int64_graph),
            ("Publish String", self.make_string_graph),
        ]

        # build UI
        self._build_ui()

    def make_rtf_graph(self) -> None:
        """Create a ROS2 publisher graph that publishes the real-time factor (RTF) as a Float32 message.

        The graph includes nodes for playback tick, generic publisher configured for Float32 messages, RTF computation,
        and ROS2 context.
        """
        self._make_generic_graph("rtf_float32")

    def make_bool_graph(self) -> None:
        """Create a ROS2 publisher graph that publishes a boolean value.

        The graph includes nodes for playback tick, generic publisher configured for Bool messages, a constant boolean
        value, and ROS2 context.
        """
        self._make_generic_graph("bool")

    def make_int64_graph(self) -> None:
        """Create a ROS2 publisher graph that publishes an Int64 value.

        The graph includes nodes for playback tick, generic publisher configured for Int64 messages, a constant integer
        value set to 42, and ROS2 context.
        """
        self._make_generic_graph("int64")

    def make_string_graph(self) -> None:
        """Create a ROS2 publisher graph that publishes a string message.

        The graph includes nodes for playback tick, generic publisher configured for String messages, a constant string
        value, and ROS2 context.
        """
        self._make_generic_graph("string")

    def _make_generic_graph(self, publisher_kind: str) -> None:
        try:
            self._og_path = create_ros2_generic_publisher_graph(
                Ros2GenericPublisherGraphConfig(graph_path=self._og_path, publisher_kind=publisher_kind)
            )
        except (ValueError, RuntimeError) as exc:
            post_notification(str(exc), status=NotificationStatus.WARNING)

    def _build_ui(self) -> None:
        """Build the UI for the ROS2 Generic Publisher Graph window.

        The UI includes a graph path input field, a dropdown menu for selecting the publisher type, OK and Cancel
        buttons, and links to documentation and Python script.
        """
        default_og_path = "/Graph/ROS_GenericPub"
        og_path_def = ParamWidget.FieldDef(
            name="og_path", label="Graph Path", type=ui.StringField, default=default_og_path
        )

        with self.frame:
            with ui.VStack(spacing=4):
                self.og_path_input = ParamWidget(field_def=og_path_def)

                self._dropdown_model = dropdown_builder(
                    label="Generic Publisher Graph",
                    items=[names for (names, _) in self._dropdown_operations_list],
                    tooltip="Select an example generic publisher graph",
                )
                add_ok_cancel_buttons(self._on_ok, self._on_cancel)
                add_script_docs_footer(
                    __file__,
                    "https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_generic_publisher_subscriber.html",
                )
        return

    def _on_ok(self) -> None:
        """Handle the OK button click event.

        Validates parameters, creates the selected publisher graph, and closes the window if validation succeeds.
        """
        self._og_path = self.og_path_input.get_value()

        self._finish_on_ok(
            lambda: self._dropdown_operations_list[self._dropdown_model.get_item_value_model().as_int][1]()
        )

    def _check_params(self) -> bool:
        """Validate the graph parameters.

        Checks if a graph already exists at the specified path. If it does, displays a warning notification.

        Returns:
            True if parameters are valid, False otherwise.

        """
        return validate_new_graph_path(self._og_path, "generic publisher graph")


class Ros2JointStatesGraph(Ros2GraphWindow):
    """A UI window for generating OmniGraph graphs that publish and subscribe to ROS 2 joint state messages.

    This class provides a graphical interface to create OmniGraph configurations for ROS 2 joint state communication.
    It supports creating publisher nodes to send joint states, subscriber nodes to receive joint commands, and optional
    articulation controller nodes to apply received commands to robots in the scene.

    The generated graph can be created as a new standalone graph or added to an existing OmniGraph. When configured as a
    subscriber with robot movement enabled, it automatically connects the received joint commands to an articulation
    controller node that moves the robot accordingly.

    The UI allows users to specify:

    - Graph path where the nodes will be created
    - ROS 2 node namespace for topic organization
    - Articulation root path of the target robot
    - Publisher topic name for sending joint states
    - Subscriber topic name for receiving joint commands
    - Whether to add nodes to an existing graph
    - Whether to create publisher and/or subscriber nodes
    - Whether the subscriber should control robot movement

    The window provides links to documentation and the Python script used for graph generation, enabling users to
    understand and customize the generated OmniGraph configurations.
    """

    def __init__(self) -> None:
        super().__init__("ROS2 Joint States Graph", width=450, height=350)
        self._og_path = "/Graph/ROS_JointStates"
        self._node_namespace = ""
        self._art_root_path = ""
        self._pub_topic = "/joint_states"
        self._sub_topic = "/joint_command"
        self._add_to_existing_graph = False
        self._publisher = False
        self._subscriber = False
        self._sub_move_robot = True  # does subscriber feeds into an articulation node to move the robot

        # build UI
        self._build_ui()

    def make_graph(self) -> None:
        """Create or modifies the ROS2 joint states action graph with publisher and subscriber nodes.

        This method stops the timeline, then creates a new graph or modifies an existing one by adding ROS2 nodes for
        publishing and subscribing to joint states. When publishing is enabled, it adds a joint state publisher node.
        When subscribing is enabled, it adds a joint state subscriber node and optionally an articulation controller
        node to move the robot based on received joint commands.
        """
        try:
            self._og_path = create_ros2_joint_states_graph(
                Ros2JointStatesGraphConfig(
                    graph_path=self._og_path,
                    articulation_root=self._art_root_path,
                    node_namespace=self._node_namespace,
                    publish_topic=self._pub_topic,
                    subscribe_topic=self._sub_topic,
                    add_to_existing_graph=self._add_to_existing_graph,
                    publish_joint_states=self._publisher,
                    subscribe_joint_states=self._subscriber,
                    move_robot_on_subscribe=self._sub_move_robot,
                )
            )
        except (ValueError, RuntimeError) as exc:
            post_notification(str(exc), status=NotificationStatus.WARNING)

    def _build_ui(self) -> None:
        """Construct the user interface for the ROS2 Joint States Graph window.

        This method creates UI elements including graph path input, node namespace input, articulation root prim
        selector, publisher/subscriber checkboxes, topic input fields, and control buttons. It also adds links to the
        python script and documentation.
        """
        og_path_def = ParamWidget.FieldDef(
            name="og_path", label="Graph Path", type=ui.StringField, default=self._og_path
        )
        node_namespace_def = ParamWidget.FieldDef(
            name="node_namespace", label="Node Namespace", type=ui.StringField, default=self._node_namespace
        )
        pub_topic_def = ParamWidget.FieldDef(
            name="pub topic", label="Publisher Topic", type=ui.StringField, default=self._pub_topic
        )
        sub_topic_def = ParamWidget.FieldDef(
            name="sub topic", label="Subscriber Topic", type=ui.StringField, default=self._sub_topic
        )
        with self.frame:
            with ui.VStack(spacing=4):
                with ui.HStack():
                    ui.Label("Add to an existing graph?", width=ui.Percent(30))
                    cb = ui.SimpleBoolModel(default_value=self._add_to_existing_graph)
                    SimpleCheckBox(self._add_to_existing_graph, self._on_use_existing_graph, model=cb)
                self.og_path_input = ParamWidget(field_def=og_path_def)
                self.node_namespace_input = ParamWidget(field_def=node_namespace_def)
                self.art_root_input = SelectPrimWidget(label="Articulation Root", default=self._art_root_path)
                ui.Spacer(height=5)
                with ui.HStack():
                    ui.Label("Publisher", width=ui.Percent(15))
                    cb = ui.SimpleBoolModel(default_value=self._publisher)
                    SimpleCheckBox(self._publisher, self._on_pub_graph, model=cb)
                    ui.Spacer(width=ui.Percent(5))
                    self.pub_topic_input = ParamWidget(field_def=pub_topic_def)
                    ui.Spacer(width=ui.Percent(20))
                with ui.HStack():
                    ui.Label("Subscriber", width=ui.Percent(15))
                    cb = ui.SimpleBoolModel(default_value=self._subscriber)
                    SimpleCheckBox(self._subscriber, self._on_sub_graph, model=cb)
                    ui.Spacer(width=ui.Percent(5))
                    self.sub_topic_input = ParamWidget(field_def=sub_topic_def)
                    ui.Label("Move Robot?", width=ui.Percent(15))
                    cb = ui.SimpleBoolModel(default_value=self._sub_move_robot)
                    SimpleCheckBox(self._sub_move_robot, self._on_sub_move_robot, model=cb)
                add_ok_cancel_buttons(self._on_ok, self._on_cancel)
                add_script_docs_footer(
                    __file__,
                    "https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_manipulation.html#graph-shortcut",
                )
        return

    def _on_ok(self) -> None:
        """Handle the OK button click event.

        This method retrieves values from UI widgets, validates parameters, and creates the graph if validation
        passes. The window is hidden on success, or a warning notification is displayed on failure.
        """
        self._og_path = self.og_path_input.get_value()
        self._node_namespace = self.node_namespace_input.get_value()
        self._art_root_path = self.art_root_input.get_value()
        self._pub_topic = self.pub_topic_input.get_value()
        self._sub_topic = self.sub_topic_input.get_value()

        self._finish_on_ok()

    def _check_params(self) -> bool:
        """Validate the graph parameters.

        When adding to an existing graph, this method verifies that the specified graph path exists and is a valid
        OmniGraph. If validation fails, a warning notification is posted.

        Returns:
            True if all parameters are valid, False otherwise.

        """
        if self._add_to_existing_graph and not validate_existing_graph_path(self._og_path):
            return False

        return True

    def _on_pub_graph(self, check_state: bool) -> None:
        """Handle the checkbox state change for enabling the publisher.

        Args:
            check_state: Whether to enable the publisher.

        """
        self._publisher = check_state

    def _on_sub_graph(self, check_state: bool) -> None:
        """Handle the checkbox state change for enabling the subscriber.

        Args:
            check_state: Whether to enable the subscriber.

        """
        self._subscriber = check_state

    def _on_sub_move_robot(self, check_state: bool) -> None:
        """Handle the checkbox state change for enabling robot movement from subscriber.

        Args:
            check_state: Whether to enable robot movement from subscriber.

        """
        self._sub_move_robot = check_state


class Ros2TfPubGraph(Ros2GraphWindow):
    """A UI window for creating ROS2 transform (TF) publisher graphs.

    This class provides an interface for setting up OmniGraph graphs that publish coordinate frame transformations
    to ROS2. It enables users to configure TF publication by selecting target and parent prims, specifying graph
    paths, and optionally adding nodes to existing graphs. The window supports both creating new graphs with the
    necessary ROS2 context, timing, and TF publisher nodes, and extending existing graphs with additional TF
    publication capabilities.

    The generated graph publishes transform data between coordinate frames, allowing ROS2 nodes to track the
    spatial relationships between different parts of a robot or scene. Users can choose to append target prims to
    an existing TF publisher node (when multiple transforms share the same parent frame) or create separate
    publisher nodes for different transform hierarchies.
    """

    def __init__(self) -> None:
        super().__init__("ROS2 TF Publisher Graph", width=450, height=450)
        self._og_path = "/Graph/ROS_TF"
        self._node_namespace = ""
        self._existing_node_path = ""
        self._target_prim = ""
        self._parent_prim = ""
        self._pub_topic = "/tf"
        self._add_to_existing_graph = False
        self._add_to_existing_node = False
        self._has_existing_node = False

        # build UI
        self._build_ui()

    def make_graph(self) -> None:
        """Create or modifies an OmniGraph to publish ROS2 transform trees.

        Generates a new graph with tick, context, and simulation time nodes if starting from scratch, or adds
        transform tree publisher nodes to an existing graph. Configures connections between nodes based on the
        graph structure and user preferences.
        """
        try:
            self._og_path = create_ros2_tf_graph(
                Ros2TfGraphConfig(
                    graph_path=self._og_path,
                    target_prim=self._target_prim,
                    parent_prim=self._parent_prim,
                    topic=self._pub_topic,
                    node_namespace=self._node_namespace,
                    add_to_existing_graph=self._add_to_existing_graph,
                    append_to_existing_tf_node=self._add_to_existing_node,
                    existing_tf_node_path=self._existing_node_path,
                )
            )
        except (ValueError, RuntimeError) as exc:
            post_notification(str(exc), status=NotificationStatus.WARNING)

    def _build_ui(self) -> None:
        """Construct the UI elements for the ROS2 TF Publisher Graph window.

        Creates input fields for graph path, node namespace, node path, target prim, parent prim, and publisher
        topic. Includes checkboxes for existing graph and node options, along with action buttons and
        documentation links.
        """
        og_path_def = ParamWidget.FieldDef(
            name="og_path", label="Graph Path", type=ui.StringField, default=self._og_path
        )
        node_namespace_def = ParamWidget.FieldDef(
            name="node_namespace", label="Node Namespace", type=ui.StringField, default=self._node_namespace
        )
        pub_topic_def = ParamWidget.FieldDef(
            name="pub topic", label="Publisher Topic", type=ui.StringField, default=self._pub_topic
        )

        with self.frame:
            with ui.VStack(spacing=4):
                with ui.HStack():
                    ui.Label("Add to an existing graph?", width=ui.Percent(30))
                    cb = ui.SimpleBoolModel(default_value=self._add_to_existing_graph)
                    SimpleCheckBox(self._add_to_existing_graph, self._on_use_existing_graph, model=cb)
                self.og_path_input = ParamWidget(field_def=og_path_def)
                self.node_namespace_input = ParamWidget(field_def=node_namespace_def)
                with ui.HStack():
                    ui.Label("Add to an existing node (i.e. same Parent)?", width=ui.Percent(30))
                    cb = ui.SimpleBoolModel(default_value=self._add_to_existing_node)
                    SimpleCheckBox(self._add_to_existing_node, self._on_use_existing_node, model=cb)

                self.node_path_input = SelectPrimWidget(label="Node Path", default=self._existing_node_path)
                ui.Spacer(height=15)
                self.target_prim_input = SelectPrimWidget(label="Target Prim", default=self._target_prim)
                self.parent_prim_input = SelectPrimWidget(
                    label="Parent Prim (default to /World)", default=self._parent_prim
                )

                self.pub_topic_input = ParamWidget(field_def=pub_topic_def)
                ui.Spacer(width=ui.Percent(20))

                add_ok_cancel_buttons(self._on_ok, self._on_cancel)
                add_script_docs_footer(
                    __file__,
                    "https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_tf.html",
                )
        return

    def _on_ok(self) -> None:
        """Handle the OK button click event.

        Retrieves values from all input fields, validates parameters through _check_params, and generates the
        graph if validation passes. Closes the window on success or displays a warning notification if parameter
        check fails.
        """
        self._og_path = self.og_path_input.get_value()
        self._node_namespace = self.node_namespace_input.get_value()
        self._existing_node_path = self.node_path_input.get_value()
        self._parent_prim = self.parent_prim_input.get_value()
        self._target_prim = self.target_prim_input.get_value()
        self._pub_topic = self.pub_topic_input.get_value()

        self._finish_on_ok()

    def _check_params(self) -> bool:
        """Validate the user-provided parameters before graph creation.

        Verifies that the specified graph path exists when adding to an existing graph, and that the node path
        exists and is a valid OmniGraph node when adding to an existing node. Also ensures that adding to an
        existing node requires adding to an existing graph.

        Returns:
            True if all parameter checks pass, False otherwise.

        """
        if not self._target_prim:
            post_notification("Target prim is required", status=NotificationStatus.WARNING)
            return False

        if self._add_to_existing_graph and not validate_existing_graph_path(self._og_path):
            return False

        if self._add_to_existing_node and not self._add_to_existing_graph:
            msg = "Adding to an existing node requires adding to an existing graph, check the add to existing graph checkbox"
            post_notification(msg, status=NotificationStatus.WARNING)
            return False

        if self._add_to_existing_node:
            # make sure the "existing" node exist
            node_prim = prim_utils.get_prim_at_path(self._existing_node_path)
            if node_prim.IsValid() and node_prim.IsA(OmniGraphSchema.OmniGraphNode):
                pass
            else:
                msg = self._existing_node_path + " is not an existing node, check the node path"
                post_notification(msg, status=NotificationStatus.WARNING)
                return False

        return True

    def _on_use_existing_node(self, check_state: bool) -> None:
        """Update the state when the existing node checkbox is toggled.

        Args:
            check_state: Whether the checkbox is checked.

        """
        self._add_to_existing_node = check_state


class Ros2OdometryGraph(Ros2GraphWindow):
    """A UI window for creating ROS2 odometry computation and publishing graphs.

    This class provides an interactive interface to generate OmniGraph action graphs that compute and publish robot
    odometry data, including position, orientation, linear velocity, and angular velocity. The generated graph can
    optionally publish transform trees for the robot and odometry frames.

    The graph creates nodes for:

    - Computing odometry from an articulated robot's chassis
    - Publishing odometry messages to a ROS2 topic
    - Publishing transforms between world, odom, and robot frames
    - Optionally publishing the robot's complete transform tree

    The UI allows users to specify the robot's articulation root prim, chassis link prim, namespace, and whether to add
    the odometry nodes to an existing graph or create a new one.
    """

    def __init__(self) -> None:
        super().__init__("ROS2 Odometry Graph", width=450, height=350)
        self._og_path = "/Graph/ROS_Odometry"
        self._node_namespace = ""
        self._art_root_prim = ""
        self._odom_pub_topic = "/odom"
        self._tf_pub_topic = "/tf"
        self._add_to_existing_graph = False
        self._tf_robot_pub = True  # also publish TF tree of the robot
        self._chassis_prim = ""
        self._chassis_link_name = "base_link"

        # build UI
        self._build_ui()

    def make_graph(self) -> None:
        """Create or extends an OmniGraph for ROS2 odometry publishing.

        Sets up nodes for computing odometry from an articulated robot, publishing odometry messages, and publishing
        transform trees between world, odom, and robot frames. If configured, also publishes the robot's internal TF tree.
        """
        try:
            self._og_path = create_ros2_odometry_graph(
                Ros2OdometryGraphConfig(
                    graph_path=self._og_path,
                    articulation_root=self._art_root_prim,
                    chassis_prim=self._chassis_prim,
                    node_namespace=self._node_namespace,
                    odometry_topic=self._odom_pub_topic,
                    tf_topic=self._tf_pub_topic,
                    add_to_existing_graph=self._add_to_existing_graph,
                    publish_robot_tf=self._tf_robot_pub,
                )
            )
        except (ValueError, RuntimeError) as exc:
            post_notification(str(exc), status=NotificationStatus.WARNING)

    def _build_ui(self) -> None:
        """Build the user interface for configuring the ROS2 odometry graph.

        Creates input fields for graph path, node namespace, articulation root, and chassis link selection. Includes
        checkboxes for adding to an existing graph and publishing robot TF, along with OK and Cancel buttons.
        """
        og_path_def = ParamWidget.FieldDef(
            name="og_path", label="Graph Path", type=ui.StringField, default=self._og_path
        )
        node_namespace_def = ParamWidget.FieldDef(
            name="node_namespace", label="Node Namespace", type=ui.StringField, default=self._node_namespace
        )

        with self.frame:
            with ui.VStack(spacing=4):
                with ui.HStack():
                    ui.Label("Add to an existing graph?", width=ui.Percent(30))
                    cb = ui.SimpleBoolModel(default_value=self._add_to_existing_graph)
                    SimpleCheckBox(self._add_to_existing_graph, self._on_use_existing_graph, model=cb)
                self.og_path_input = ParamWidget(field_def=og_path_def)
                self.node_namespace_input = ParamWidget(field_def=node_namespace_def)
                with ui.HStack():
                    ui.Label("Publish Robot's TF?", width=ui.Percent(30))
                    cb = ui.SimpleBoolModel(default_value=self._tf_robot_pub)
                    SimpleCheckBox(self._tf_robot_pub, self._on_publish_robot_tf, model=cb)
                self.art_root_input = SelectPrimWidget(label="Robot Articulation Root", default=self._art_root_prim)
                self.chassis_prim_input = SelectPrimWidget(label="Chassis Link Prim", default=self._chassis_prim)

                ui.Spacer(height=5)

                add_ok_cancel_buttons(self._on_ok, self._on_cancel)
                add_script_docs_footer(
                    __file__,
                    "https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_tf.html#setting-up-odometry",
                )
        return

    def _on_ok(self) -> None:
        """Handle the OK button click.

        Validates user input parameters, generates the odometry graph if validation passes, and closes the dialog.
        """
        self._og_path = self.og_path_input.get_value()
        self._node_namespace = self.node_namespace_input.get_value()
        self._art_root_prim = self.art_root_input.get_value()
        self._chassis_prim = self.chassis_prim_input.get_value()
        self._chassis_link_name = self._chassis_prim.split("/")[-1]

        self._finish_on_ok()

    def _check_params(self) -> bool:
        """Validate the user-provided parameters.

        Returns:
            True if all parameters are valid, False otherwise.

        """
        if not self._art_root_prim:
            post_notification("Robot Articulation Root prim is required", status=NotificationStatus.WARNING)
            return False

        if self._add_to_existing_graph and not validate_existing_graph_path(self._og_path):
            return False

        return True

    def _on_publish_robot_tf(self, check_state: bool) -> None:
        """Handle the checkbox state change for publishing robot TF.

        Args:
            check_state: Whether the checkbox is checked.

        """
        self._tf_robot_pub = check_state
