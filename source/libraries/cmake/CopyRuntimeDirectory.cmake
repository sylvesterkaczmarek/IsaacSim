# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/IsaacSimPackagingFilters.cmake")

if(NOT DEFINED SOURCE OR NOT IS_DIRECTORY "${SOURCE}")
    message(FATAL_ERROR "CopyRuntimeDirectory requires an existing SOURCE directory: ${SOURCE}")
endif()
if(NOT DEFINED DESTINATION OR DESTINATION STREQUAL "")
    message(FATAL_ERROR "CopyRuntimeDirectory requires DESTINATION")
endif()

file(REMOVE_RECURSE "${DESTINATION}")
file(MAKE_DIRECTORY "${DESTINATION}")
file(COPY "${SOURCE}/"
    DESTINATION "${DESTINATION}"
    ${ISAACSIM_PACKAGING_EXCLUDE_ARGS}
)
