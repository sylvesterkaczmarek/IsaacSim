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

if(NOT BINARY OR NOT EXISTS "${BINARY}" OR NOT SYMBOLS_FILE OR NOT SYMBOL_PREFIX)
    message(FATAL_ERROR "Exported-symbol check requires BINARY, SYMBOLS_FILE, and SYMBOL_PREFIX")
endif()
if(NOT UPDATE_BASELINE AND NOT EXISTS "${SYMBOLS_FILE}")
    message(FATAL_ERROR "Exported-symbol baseline does not exist: ${SYMBOLS_FILE}")
endif()

if(WIN32)
    find_program(symbol_tool dumpbin REQUIRED)
    execute_process(
        COMMAND "${symbol_tool}" /EXPORTS "${BINARY}"
        OUTPUT_VARIABLE exported_symbols
        RESULT_VARIABLE result
    )
elseif(APPLE)
    find_program(symbol_tool nm REQUIRED)
    execute_process(
        COMMAND "${symbol_tool}" -gU "${BINARY}"
        OUTPUT_VARIABLE exported_symbols
        RESULT_VARIABLE result
    )
else()
    find_program(symbol_tool nm REQUIRED)
    execute_process(
        COMMAND "${symbol_tool}" -D --defined-only "${BINARY}"
        OUTPUT_VARIABLE exported_symbols
        RESULT_VARIABLE result
    )
endif()
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Could not inspect exported symbols for ${BINARY}")
endif()

string(REPLACE "\r\n" "\n" exported_symbols "${exported_symbols}")
string(REPLACE "\n" ";" exported_lines "${exported_symbols}")
set(actual_symbols)
foreach(line IN LISTS exported_lines)
    string(STRIP "${line}" line)
    if(WIN32 AND line MATCHES
        "^[0-9]+[ \t]+[0-9A-Fa-f]+[ \t]+[0-9A-Fa-f]+[ \t]+(${SYMBOL_PREFIX}[A-Za-z0-9_]*)([ \t=]|$)"
    )
        list(APPEND actual_symbols "${CMAKE_MATCH_1}")
    elseif(APPLE AND line MATCHES "(^|[ \t])_(${SYMBOL_PREFIX}[A-Za-z0-9_]*)$")
        list(APPEND actual_symbols "${CMAKE_MATCH_2}")
    elseif(line MATCHES "(^|[ \t])(${SYMBOL_PREFIX}[A-Za-z0-9_]*)$")
        list(APPEND actual_symbols "${CMAKE_MATCH_2}")
    endif()
endforeach()
list(REMOVE_DUPLICATES actual_symbols)
list(SORT actual_symbols)

if(UPDATE_BASELINE)
    list(JOIN actual_symbols "\n" baseline_contents)
    if(baseline_contents)
        string(APPEND baseline_contents "\n")
    endif()
    file(WRITE "${SYMBOLS_FILE}" "${baseline_contents}")
    message(STATUS "Updated ABI symbol baseline: ${SYMBOLS_FILE}")
    return()
endif()

file(STRINGS "${SYMBOLS_FILE}" baseline_lines)
set(expected_symbols)
foreach(symbol IN LISTS baseline_lines)
    string(STRIP "${symbol}" symbol)
    if(symbol STREQUAL "" OR symbol MATCHES "^#")
        continue()
    endif()
    if(NOT symbol MATCHES "^${SYMBOL_PREFIX}[A-Za-z0-9_]*$")
        message(FATAL_ERROR "ABI baseline contains a symbol outside prefix '${SYMBOL_PREFIX}': ${symbol}")
    endif()
    list(APPEND expected_symbols "${symbol}")
endforeach()
list(REMOVE_DUPLICATES expected_symbols)
list(SORT expected_symbols)

if(NOT "${expected_symbols}" STREQUAL "${actual_symbols}")
    list(JOIN expected_symbols "\n  " expected_text)
    list(JOIN actual_symbols "\n  " actual_text)
    message(FATAL_ERROR
        "Exported C ABI does not match ${SYMBOLS_FILE}\n"
        "Expected:\n  ${expected_text}\n"
        "Actual:\n  ${actual_text}\n"
        "Run the module's update-abi-baseline target only after approving the ABI change."
    )
endif()
