# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Robot Motion examples extension startup."""

from __future__ import annotations

import omni.ext
from omni.behavior.tree.core import IBehaviorNodeLibrary, get_factory, register_node

# NodeLibraryVisibility is not part of omni.behavior_tree.__all__, so it is not re-exported by
# omni.behavior.tree.core; import it from the bindings module directly.
from omni.behavior_tree import NodeLibraryVisibility

from .manipulation._behavior_tree_franka import clear_franka_motion_adapters
from .manipulation.behavior_tree_nodes import (
    FrankaAcquireCube,
    FrankaCalibratePlaceTargets,
    FrankaCheckObjectAtTarget,
    FrankaCheckObjectLifted,
    FrankaCheckStackComplete,
    FrankaMarkCubePlaced,
    FrankaMoveToTarget,
    FrankaRecoverCube,
    FrankaReleaseCenter,
    FrankaReset,
    FrankaSetGripper,
)

BEHAVIOR_TREE_NODE_LIBRARY_NAME = "isaacsim.robot_motion.examples"
LEGACY_BEHAVIOR_TREE_NODE_LIBRARY_NAME = "isaacsim.robot.experimental.manipulators.examples"

# The legacy library only exists so behavior trees saved under the extension's former name still
# resolve their nodes; it is hidden so the same nodes are not listed twice in the Node Library panel.
_BEHAVIOR_TREE_NODE_LIBRARIES = (
    (BEHAVIOR_TREE_NODE_LIBRARY_NAME, NodeLibraryVisibility.PUBLIC),
    (LEGACY_BEHAVIOR_TREE_NODE_LIBRARY_NAME, NodeLibraryVisibility.HIDDEN),
)

_BEHAVIOR_TREE_NODE_CLASSES = (
    FrankaReset,
    FrankaAcquireCube,
    FrankaRecoverCube,
    FrankaCalibratePlaceTargets,
    FrankaMoveToTarget,
    FrankaSetGripper,
    FrankaCheckObjectLifted,
    FrankaCheckObjectAtTarget,
    FrankaReleaseCenter,
    FrankaMarkCubePlaced,
    FrankaCheckStackComplete,
)

_behavior_tree_node_libraries: tuple[IBehaviorNodeLibrary, ...] = ()


class Extension(omni.ext.IExt):
    """Register Robot Motion example behavior-tree nodes."""

    def on_startup(self, ext_id: str) -> None:
        """Called when the extension starts up.

        Args:
            ext_id: The extension ID.
        """
        global _behavior_tree_node_libraries
        libraries = []
        for library_name, visibility in _BEHAVIOR_TREE_NODE_LIBRARIES:
            library = get_factory().create_node_library(library_name, visibility)
            for node_class in _BEHAVIOR_TREE_NODE_CLASSES:
                register_node(library, node_class)
            libraries.append(library)
        _behavior_tree_node_libraries = tuple(libraries)

    def on_shutdown(self) -> None:
        """Called when the extension shuts down."""
        global _behavior_tree_node_libraries
        clear_franka_motion_adapters()
        _behavior_tree_node_libraries = ()
