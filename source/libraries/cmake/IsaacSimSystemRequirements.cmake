# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(_isaacsim_read_system_requirement requirement_file key kind output_variable)
    file(STRINGS "${requirement_file}" requirement_lines REGEX "^${key}[ \t]*=")
    list(LENGTH requirement_lines requirement_line_count)
    if(NOT requirement_line_count EQUAL 1)
        message(FATAL_ERROR "${requirement_file} must contain exactly one '${key}' field")
    endif()
    list(GET requirement_lines 0 requirement_line)

    if(kind STREQUAL "STRING")
        set(requirement_pattern "^${key}[ \t]*=[ \t]*\"([^\"]+)\"[ \t]*$")
    elseif(kind STREQUAL "INTEGER")
        set(requirement_pattern "^${key}[ \t]*=[ \t]*([0-9]+)[ \t]*$")
    else()
        message(FATAL_ERROR "Unsupported system requirement kind: ${kind}")
    endif()
    if(NOT requirement_line MATCHES "${requirement_pattern}")
        message(FATAL_ERROR "${requirement_file} has an invalid '${key}' field")
    endif()
    set("${output_variable}" "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

function(isaacsim_load_system_requirements requirement_file)
    if(NOT IS_ABSOLUTE "${requirement_file}" OR NOT EXISTS "${requirement_file}")
        message(FATAL_ERROR "System requirements file does not exist: ${requirement_file}")
    endif()

    _isaacsim_read_system_requirement(
        "${requirement_file}"
        cmake_minimum_version
        STRING
        cmake_minimum_version
    )
    _isaacsim_read_system_requirement(
        "${requirement_file}"
        cmake_policy_maximum_version
        STRING
        cmake_policy_maximum_version
    )
    _isaacsim_read_system_requirement("${requirement_file}" c_standard INTEGER c_standard)
    _isaacsim_read_system_requirement("${requirement_file}" cpp_standard INTEGER cpp_standard)
    _isaacsim_read_system_requirement("${requirement_file}" python_version STRING python_version)

    set(ISAACSIM_CMAKE_MINIMUM_VERSION "${cmake_minimum_version}" PARENT_SCOPE)
    set(ISAACSIM_CMAKE_POLICY_MAXIMUM_VERSION "${cmake_policy_maximum_version}" PARENT_SCOPE)
    set(ISAACSIM_C_STANDARD "${c_standard}" PARENT_SCOPE)
    set(ISAACSIM_CXX_STANDARD "${cpp_standard}" PARENT_SCOPE)
    set(ISAACSIM_PYTHON_VERSION "${python_version}" PARENT_SCOPE)
endfunction()
