# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26...4.3)

include("${CMAKE_CURRENT_LIST_DIR}/../IsaacSimPackagingFilters.cmake")

set(runtime_files
    "/source/runtime.txt"
    "/source/__pycache__/runtime.pyc"
    "/source/.pytest_cache/state"
    "/source/.mypy_cache/state"
    "/source/.ruff_cache/state"
    "/source/runtime.pyo"
)
list(FILTER runtime_files EXCLUDE REGEX "${ISAACSIM_PACKAGING_EXCLUDE_REGEX}")
if(NOT runtime_files STREQUAL "/source/runtime.txt")
    message(FATAL_ERROR "Packaging dependency filter retained transient paths: ${runtime_files}")
endif()

set(test_root "${CMAKE_CURRENT_BINARY_DIR}/copy-runtime-directory-test")
set(source_dir "${test_root}/source")
set(destination_dir "${test_root}/destination")
file(REMOVE_RECURSE "${test_root}")
file(MAKE_DIRECTORY
    "${source_dir}/nested"
    "${source_dir}/nested/__pycache__"
    "${source_dir}/.pytest_cache"
    "${source_dir}/.mypy_cache"
    "${source_dir}/.ruff_cache"
)
file(WRITE "${source_dir}/nested/runtime.txt" "runtime\n")
file(WRITE "${source_dir}/nested/generatedSchema.usda" "#usda 1.0\n")
file(WRITE "${source_dir}/nested/__pycache__/runtime.cpython-312.pyc" "cache\n")
file(WRITE "${source_dir}/nested/runtime.pyc" "cache\n")
file(WRITE "${source_dir}/nested/runtime.pyo" "cache\n")
file(WRITE "${source_dir}/.pytest_cache/state" "cache\n")
file(WRITE "${source_dir}/.mypy_cache/state" "cache\n")
file(WRITE "${source_dir}/.ruff_cache/state" "cache\n")

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        "-DSOURCE=${source_dir}"
        "-DDESTINATION=${destination_dir}"
        -P "${CMAKE_CURRENT_LIST_DIR}/../CopyRuntimeDirectory.cmake"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "CopyRuntimeDirectory failed:\n${output}${error}")
endif()

foreach(required_file nested/runtime.txt nested/generatedSchema.usda)
    if(NOT EXISTS "${destination_dir}/${required_file}")
        message(FATAL_ERROR "CopyRuntimeDirectory omitted required file: ${required_file}")
    endif()
endforeach()
foreach(prohibited_path
    nested/__pycache__
    nested/runtime.pyc
    nested/runtime.pyo
    .pytest_cache
    .mypy_cache
    .ruff_cache
)
    if(EXISTS "${destination_dir}/${prohibited_path}")
        message(FATAL_ERROR "CopyRuntimeDirectory retained cache path: ${prohibited_path}")
    endif()
endforeach()

file(REMOVE_RECURSE "${test_root}")
