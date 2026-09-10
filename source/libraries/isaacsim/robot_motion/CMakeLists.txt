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

isaacsim_add_group(
    NAME isaacsim_robot_motion
    MODULES
        isaacsim.robot_motion.experimental.motion_generation
        isaacsim.robot_motion.controllers
        isaacsim.robot_motion.cumotion
)

isaacsim_add_group_dependency(
    GROUP isaacsim_robot_motion
    DEPENDS_ON isaacsim_common
)

isaacsim_add_group_dependency(
    GROUP isaacsim_robot_motion
    DEPENDS_ON isaacsim_deprecated
)

isaacsim_add_group_dependency(
    GROUP isaacsim_robot_motion
    DEPENDS_ON isaacsim_robot_motion_schema
)

add_subdirectory(experimental/motion_generation)
add_subdirectory(controllers)
add_subdirectory(cumotion)
