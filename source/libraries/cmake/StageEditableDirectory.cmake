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

include("${CMAKE_CURRENT_LIST_DIR}/IsaacSimPackagingFilters.cmake")

if(NOT DEFINED SOURCE OR NOT IS_DIRECTORY "${SOURCE}")
    message(FATAL_ERROR "StageEditableDirectory requires an existing SOURCE directory: ${SOURCE}")
endif()
if(NOT DEFINED DESTINATION OR DESTINATION STREQUAL "")
    message(FATAL_ERROR "StageEditableDirectory requires DESTINATION")
endif()

cmake_path(ABSOLUTE_PATH SOURCE NORMALIZE OUTPUT_VARIABLE normalized_source)
cmake_path(ABSOLUTE_PATH DESTINATION NORMALIZE OUTPUT_VARIABLE normalized_destination)
if(normalized_source STREQUAL normalized_destination)
    message(FATAL_ERROR "StageEditableDirectory SOURCE and DESTINATION must differ")
endif()

cmake_path(GET normalized_destination PARENT_PATH destination_parent)
file(MAKE_DIRECTORY "${destination_parent}")
file(REMOVE_RECURSE "${normalized_destination}")
if(WIN32)
    # GitLab restores build artifacts before job setup can recreate stable drive mappings. Keep Windows stages
    # self-contained so restored Python packages and tests never depend on source-tree directory symlinks.
    file(MAKE_DIRECTORY "${normalized_destination}")
    file(COPY "${normalized_source}/"
        DESTINATION "${normalized_destination}"
        ${ISAACSIM_PACKAGING_EXCLUDE_ARGS}
    )
else()
    file(CREATE_LINK "${normalized_source}" "${normalized_destination}" SYMBOLIC RESULT link_result)
    if(link_result)
        message(STATUS "Directory symlink unavailable; copying editable stage: ${normalized_source}")
        file(MAKE_DIRECTORY "${normalized_destination}")
        file(COPY "${normalized_source}/"
            DESTINATION "${normalized_destination}"
            ${ISAACSIM_PACKAGING_EXCLUDE_ARGS}
        )
    endif()
endif()
