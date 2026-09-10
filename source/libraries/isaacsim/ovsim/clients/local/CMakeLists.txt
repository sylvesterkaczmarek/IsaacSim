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

isaacsim_add_module(
    NAME isaacsim.ovsim.clients.local
    EXPORT_HEADER isaacsim/ovsim/clients/local/Export.h
    EXPORT_MACRO ISAACSIM_OVSIM_CLIENTS_LOCAL_API
    PUBLIC_DEPENDENCIES
        isaacsim::common-array
    PRIVATE_DEPENDENCIES
        isaacsim::common-logging
        isaacsim::foundation-ovsim
        isaacsim::physics-ovsim
    PUBLIC_CPP_HEADERS
        isaacsim/ovsim/clients/local/control/authoring/Authoring.hpp
        isaacsim/ovsim/clients/local/control/simulation/Simulation.hpp
        isaacsim/ovsim/clients/local/data/Data.hpp
    SOURCES
        src/Control.cpp
        src/Data.cpp
        src/VerifyInterfaces.cpp
)

target_include_directories(isaacsim-ovsim-clients-local PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ovsim/interfaces/cpp>
)

# The public wrappers include the canonical OV Sim interface headers,
# so ship those headers with the same development component.
get_target_property(isaacsim_ovsim_clients_local_group isaacsim-ovsim-clients-local ISAACSIM_GROUP_NAME)
_isaacsim_get_group_component(
    "${isaacsim_ovsim_clients_local_group}" development isaacsim_ovsim_clients_local_development_component
)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/ovsim/interfaces/cpp/ovsim"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT "${isaacsim_ovsim_clients_local_development_component}"
    FILES_MATCHING PATTERN "*.hpp"
)

isaacsim_add_cpp_tests(
    MODULE isaacsim.ovsim.clients.local
    RESOURCE_DIR resources/tests/isaacsim.ovsim.clients.local
    SOURCES
        tests/cpp/Control.cpp
        tests/cpp/Data.cpp
    DEPENDENCIES
        isaacsim::foundation-ovsim
        isaacsim::physics-ovsim
    LABELS
        unit
)
