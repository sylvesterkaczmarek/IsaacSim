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

include("${CMAKE_CURRENT_LIST_DIR}/../IsaacSimPython.cmake")

if(TEST_CASE)
    if(TEST_CASE STREQUAL duplicate)
        _isaacsim_register_python_import("isaacsim.compat.first" "isaacsim.core.legacy" "isaacsim_compat")
        _isaacsim_register_python_import("isaacsim.compat.second" "isaacsim.core.legacy" "isaacsim_compat")
    elseif(TEST_CASE STREQUAL invalid)
        _isaacsim_register_python_import("isaacsim.compat.invalid" "unscoped.legacy" "isaacsim_compat")
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

_isaacsim_register_python_import(
    "isaacsim.compat.legacy"
    "isaacsim.core.legacy"
    "isaacsim_compat"
)
get_property(python_imports GLOBAL PROPERTY ISAACSIM_PYTHON_IMPORTS)
get_property(python_import_owners GLOBAL PROPERTY ISAACSIM_PYTHON_IMPORT_OWNERS)
get_property(group_python_imports GLOBAL PROPERTY ISAACSIM_GROUP_isaacsim_compat_PYTHON_IMPORTS)
if(NOT python_imports STREQUAL "isaacsim.core.legacy"
   OR NOT python_import_owners STREQUAL "isaacsim.compat.legacy"
   OR NOT group_python_imports STREQUAL "isaacsim.core.legacy")
    message(FATAL_ERROR
        "Python import registration produced ${python_imports}, ${python_import_owners}, ${group_python_imports}"
    )
endif()

expect_failure(duplicate "already provided")
expect_failure(invalid "Invalid Python import name")
