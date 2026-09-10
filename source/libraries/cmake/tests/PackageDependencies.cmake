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

cmake_minimum_required(VERSION 3.26...4.3)

include("${CMAKE_CURRENT_LIST_DIR}/../IsaacSimPackage.cmake")

function(set_up_groups)
    set_property(GLOBAL PROPERTY ISAACSIM_GROUP_NAMES "dependency;consumer")
    set_property(GLOBAL PROPERTY ISAACSIM_GROUP_dependency_VERSION "6.1.0")
    set_property(GLOBAL PROPERTY ISAACSIM_GROUP_consumer_VERSION "6.1.0")
    set_property(GLOBAL PROPERTY ISAACSIM_GROUP_consumer_DEPENDENCIES "")
endfunction()

if(TEST_CASE)
    set_up_groups()
    if(TEST_CASE STREQUAL self)
        isaacsim_add_group_dependency(GROUP consumer DEPENDS_ON consumer)
    elseif(TEST_CASE STREQUAL unknown_dependency)
        isaacsim_add_group_dependency(GROUP consumer DEPENDS_ON missing)
    elseif(TEST_CASE STREQUAL unknown_owner)
        isaacsim_add_group_dependency(GROUP missing DEPENDS_ON dependency)
    elseif(TEST_CASE STREQUAL explicit_version)
        isaacsim_add_group_dependency(GROUP consumer DEPENDS_ON dependency EXACT_VERSION 6.1.0)
    elseif(TEST_CASE STREQUAL mismatched_versions)
        set_property(GLOBAL PROPERTY ISAACSIM_GROUP_dependency_VERSION "6.0.0")
        isaacsim_add_group_dependency(GROUP consumer DEPENDS_ON dependency)
    elseif(TEST_CASE STREQUAL duplicate)
        isaacsim_add_group_dependency(GROUP consumer DEPENDS_ON dependency)
        isaacsim_add_group_dependency(GROUP consumer DEPENDS_ON dependency)
    else()
        message(FATAL_ERROR "Unknown test case: ${TEST_CASE}")
    endif()
    message(FATAL_ERROR "Negative test unexpectedly succeeded: ${TEST_CASE}")
endif()

function(expect_failure test_case expected_message)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" "-DTEST_CASE=${test_case}" -P "${CMAKE_CURRENT_LIST_FILE}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(result EQUAL 0)
        message(FATAL_ERROR "${test_case} unexpectedly succeeded")
    endif()
    string(CONCAT combined_output "${output}" "${error}")
    if(NOT combined_output MATCHES "${expected_message}")
        message(FATAL_ERROR "${test_case} failed for the wrong reason:\n${combined_output}")
    endif()
endfunction()

set_up_groups()
isaacsim_add_group_dependency(GROUP consumer DEPENDS_ON dependency)
get_property(dependencies GLOBAL PROPERTY ISAACSIM_GROUP_consumer_DEPENDENCIES)
get_property(version GLOBAL PROPERTY ISAACSIM_GROUP_consumer_DEPENDENCY_dependency_VERSION)
get_property(specifier GLOBAL PROPERTY ISAACSIM_GROUP_consumer_DEPENDENCY_dependency_SPECIFIER)
if(NOT dependencies STREQUAL dependency OR NOT version STREQUAL "6.1.0" OR NOT specifier STREQUAL "==6.1.0")
    message(FATAL_ERROR "Exact dependency registration produced ${dependencies} ${version} ${specifier}")
endif()

expect_failure(self "cannot depend on itself")
expect_failure(unknown_dependency "must be registered")
expect_failure(unknown_owner "owner group is not registered")
expect_failure(explicit_version "unknown arguments")
expect_failure(mismatched_versions "must use the same")
expect_failure(duplicate "more than once")
