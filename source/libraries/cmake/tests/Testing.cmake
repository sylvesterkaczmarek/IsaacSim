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

set(ISAACSIM_LIBRARIES_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
include("${CMAKE_CURRENT_LIST_DIR}/../IsaacSimModuleCore.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../IsaacSimTesting.cmake")

if(TEST_CASE)
    if(TEST_CASE STREQUAL c)
        _isaacsim_validate_test_sources("isaacsim.test.module" c cpp "tests/c/Main.cpp")
    elseif(TEST_CASE STREQUAL cpp)
        _isaacsim_validate_test_sources("isaacsim.test.module" cpp cpp "tests/cpp/main.cpp")
    else()
        message(FATAL_ERROR "Unknown test case: ${TEST_CASE}")
    endif()
    message(FATAL_ERROR "Negative test unexpectedly succeeded: ${TEST_CASE}")
endif()

function(expect_main_rejected test_case)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" "-DTEST_CASE=${test_case}" -P "${CMAKE_CURRENT_LIST_FILE}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(result EQUAL 0)
        message(FATAL_ERROR "${test_case} Main.cpp unexpectedly succeeded")
    endif()
    string(CONCAT combined_output "${output}" "${error}")
    if(NOT combined_output MATCHES "must not provide Main.cpp")
        message(FATAL_ERROR "${test_case} Main.cpp failed for the wrong reason:\n${combined_output}")
    endif()
endfunction()

expect_main_rejected(c)
expect_main_rejected(cpp)
