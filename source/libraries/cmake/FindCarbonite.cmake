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

if(TARGET carb::static AND DEFINED ISAACSIM_CARBONITE_PLATFORM_DIR)
    set(Carbonite_FOUND TRUE)
    return()
endif()

set(ISAACSIM_CARBONITE_ROOT "${ISAACSIM_TARGET_DEPS_DIR}/carb_sdk_plugins" CACHE PATH
    "Pinned Carbonite SDK root used by private static module backends"
)
set(_carbonite_include_dir "${ISAACSIM_CARBONITE_ROOT}/include")
if(NOT IS_DIRECTORY "${_carbonite_include_dir}")
    message(FATAL_ERROR
        "Carbonite headers were not found at ${_carbonite_include_dir}. "
        "Pull deps/isaacsim-libraries.packman.xml before configuring this module."
    )
endif()

file(GLOB _carbonite_build_entries LIST_DIRECTORIES TRUE "${ISAACSIM_CARBONITE_ROOT}/_build/*")
set(_carbonite_platform_dirs)
foreach(_carbonite_build_entry IN LISTS _carbonite_build_entries)
    if(IS_DIRECTORY "${_carbonite_build_entry}")
        list(APPEND _carbonite_platform_dirs "${_carbonite_build_entry}")
    endif()
endforeach()
list(LENGTH _carbonite_platform_dirs _carbonite_platform_count)
if(NOT _carbonite_platform_count EQUAL 1)
    message(FATAL_ERROR
        "Expected one Carbonite platform directory under ${ISAACSIM_CARBONITE_ROOT}/_build, "
        "found ${_carbonite_platform_count}: ${_carbonite_platform_dirs}"
    )
endif()
list(GET _carbonite_platform_dirs 0 _carbonite_platform_dir)
set(ISAACSIM_CARBONITE_PLATFORM_DIR "${_carbonite_platform_dir}" CACHE INTERNAL
    "Pinned Carbonite platform build directory"
    FORCE
)

find_library(_carbonite_static_release NAMES carb.static
    PATHS "${_carbonite_platform_dir}/release" NO_DEFAULT_PATH NO_CACHE REQUIRED
)
find_library(_carbonite_static_debug NAMES carb.static
    PATHS "${_carbonite_platform_dir}/debug" NO_DEFAULT_PATH NO_CACHE REQUIRED
)
find_package(Threads REQUIRED)

if(NOT TARGET carb::static)
    add_library(carb::static STATIC IMPORTED)
    set_target_properties(carb::static PROPERTIES
        IMPORTED_CONFIGURATIONS "Debug;Release"
        IMPORTED_LOCATION_DEBUG "${_carbonite_static_debug}"
        IMPORTED_LOCATION_RELEASE "${_carbonite_static_release}"
        MAP_IMPORTED_CONFIG_MINSIZEREL Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
        INTERFACE_INCLUDE_DIRECTORIES "${_carbonite_include_dir}"
        INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS}"
    )
    if(WIN32)
        set_property(TARGET carb::static APPEND PROPERTY INTERFACE_LINK_LIBRARIES shlwapi pathcch winmm)
    endif()
endif()

set(Carbonite_FOUND TRUE)
