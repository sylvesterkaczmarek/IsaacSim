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
    NAME isaacsim.ovsim.clients.grpc
    EXPORT_HEADER isaacsim/ovsim/clients/grpc/Export.h
    EXPORT_MACRO ISAACSIM_OVSIM_CLIENTS_GRPC_API
    PUBLIC_DEPENDENCIES
        isaacsim::common-array
    PUBLIC_CPP_HEADERS
        isaacsim/ovsim/clients/grpc/GrpcSession.hpp
    SOURCES
        src/GrpcSession.cpp
)

target_include_directories(isaacsim-ovsim-clients-grpc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ovsim/interfaces/cpp>
)

# The public wrapper includes the canonical OV Sim interface headers,
# so ship those headers with the same development component.
get_target_property(isaacsim_ovsim_clients_grpc_group isaacsim-ovsim-clients-grpc ISAACSIM_GROUP_NAME)
_isaacsim_get_group_component(
    "${isaacsim_ovsim_clients_grpc_group}" development isaacsim_ovsim_clients_grpc_development_component
)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/ovsim/interfaces/cpp/ovsim"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT "${isaacsim_ovsim_clients_grpc_development_component}"
    FILES_MATCHING PATTERN "*.hpp"
)

isaacsim_add_cpp_tests(
    MODULE isaacsim.ovsim.clients.grpc
    SOURCES
        tests/cpp/GrpcSession.cpp
    LABELS
        unit
)
