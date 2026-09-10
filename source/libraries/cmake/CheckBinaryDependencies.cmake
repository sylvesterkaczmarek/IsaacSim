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

if(NOT BINARY OR NOT EXISTS "${BINARY}")
    message(FATAL_ERROR "Binary dependency check requires an existing BINARY")
endif()

file(GET_RUNTIME_DEPENDENCIES
    LIBRARIES "${BINARY}"
    RESOLVED_DEPENDENCIES_VAR resolved_dependencies
    UNRESOLVED_DEPENDENCIES_VAR unresolved_dependencies
)
foreach(dependency IN LISTS resolved_dependencies unresolved_dependencies)
    string(TOLOWER "${dependency}" dependency_lower)
    if(dependency_lower MATCHES "(^|[/\\])(lib)?(carb|omni[.-](kit|ext|graph))")
        message(FATAL_ERROR "Module binary depends on forbidden framework library: ${dependency}")
    endif()
endforeach()
