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

include_guard(GLOBAL)

function(_isaacsim_json_quote output value)
    string(REPLACE "\\" "\\\\" escaped "${value}")
    string(REPLACE "\"" "\\\"" escaped "${escaped}")
    string(REPLACE "\r" "\\r" escaped "${escaped}")
    string(REPLACE "\n" "\\n" escaped "${escaped}")
    string(REPLACE "\t" "\\t" escaped "${escaped}")
    set(${output} "\"${escaped}\"" PARENT_SCOPE)
endfunction()

function(_isaacsim_json_array output)
    set(result "[")
    set(separator "")
    foreach(value IN LISTS ARGN)
        _isaacsim_json_quote(quoted_value "${value}")
        string(APPEND result "${separator}${quoted_value}")
        set(separator ", ")
    endforeach()
    string(APPEND result "]")
    set(${output} "${result}" PARENT_SCOPE)
endfunction()

function(_isaacsim_documentation_relative_path output path)
    if(NOT path)
        set(${output} "" PARENT_SCOPE)
        return()
    endif()
    file(RELATIVE_PATH relative_path "${ISAACSIM_REPO_ROOT}" "${path}")
    string(REPLACE "\\" "/" relative_path "${relative_path}")
    set(${output} "${relative_path}" PARENT_SCOPE)
endfunction()

function(_isaacsim_generate_documentation_catalog)
    get_property(group_names GLOBAL PROPERTY ISAACSIM_GROUP_NAMES)
    list(REMOVE_DUPLICATES group_names)
    list(SORT group_names)
    get_property(complete_group_names GLOBAL PROPERTY ISAACSIM_COMPLETE_GROUP_NAMES)

    set(distributions_json "[")
    set(distribution_separator "")
    foreach(group_name IN LISTS group_names)
        get_property(group_dependencies GLOBAL PROPERTY "ISAACSIM_GROUP_${group_name}_DEPENDENCIES")
        list(REMOVE_DUPLICATES group_dependencies)
        list(SORT group_dependencies)
        _isaacsim_json_array(dependencies_json ${group_dependencies})

        get_property(group_docs_dir GLOBAL PROPERTY "ISAACSIM_GROUP_${group_name}_DOCS_DIR")
        _isaacsim_documentation_relative_path(group_docs_root "${group_docs_dir}")
        _isaacsim_json_quote(group_name_json "${group_name}")
        _isaacsim_json_quote(group_docs_json "${group_docs_root}")
        if(group_name IN_LIST complete_group_names)
            set(group_complete_json true)
        else()
            set(group_complete_json false)
        endif()

        get_property(module_names GLOBAL PROPERTY "ISAACSIM_GROUP_${group_name}_MODULES")
        list(REMOVE_DUPLICATES module_names)
        list(SORT module_names)
        set(modules_json "[")
        set(module_separator "")
        foreach(module_name IN LISTS module_names)
            string(MAKE_C_IDENTIFIER "${module_name}" module_key)
            get_property(module_root GLOBAL PROPERTY "ISAACSIM_MODULE_${module_key}_ROOT")
            get_property(module_docs_dir GLOBAL PROPERTY "ISAACSIM_MODULE_${module_key}_DOCS_DIR")
            get_property(c_headers GLOBAL PROPERTY "ISAACSIM_MODULE_${module_key}_PUBLIC_C_HEADERS")
            get_property(cpp_headers GLOBAL PROPERTY "ISAACSIM_MODULE_${module_key}_PUBLIC_CPP_HEADERS")
            get_property(python_imports GLOBAL PROPERTY "ISAACSIM_MODULE_${module_key}_PYTHON_IMPORTS")
            list(REMOVE_DUPLICATES c_headers)
            list(REMOVE_DUPLICATES cpp_headers)
            list(REMOVE_DUPLICATES python_imports)
            list(SORT c_headers)
            list(SORT cpp_headers)
            list(SORT python_imports)

            set(public_c_headers)
            foreach(header IN LISTS c_headers)
                _isaacsim_documentation_relative_path(header_path "${module_root}/include/${header}")
                list(APPEND public_c_headers "${header_path}")
            endforeach()
            set(public_cpp_headers)
            foreach(header IN LISTS cpp_headers)
                _isaacsim_documentation_relative_path(header_path "${module_root}/include/${header}")
                list(APPEND public_cpp_headers "${header_path}")
            endforeach()
            _isaacsim_json_array(c_headers_json ${public_c_headers})
            _isaacsim_json_array(cpp_headers_json ${public_cpp_headers})
            _isaacsim_json_array(python_imports_json ${python_imports})

            set(api_languages)
            if(public_c_headers)
                list(APPEND api_languages c)
            endif()
            if(public_cpp_headers)
                list(APPEND api_languages cpp)
            endif()
            if(python_imports)
                list(APPEND api_languages python)
            endif()
            _isaacsim_json_array(api_languages_json ${api_languages})
            _isaacsim_documentation_relative_path(module_docs_root "${module_docs_dir}")
            _isaacsim_json_quote(module_name_json "${module_name}")
            _isaacsim_json_quote(module_docs_json "${module_docs_root}")
            string(APPEND modules_json
                "${module_separator}{\"name\": ${module_name_json}, \"documentation_root\": ${module_docs_json}, "
                "\"public_headers\": {\"c\": ${c_headers_json}, \"cpp\": ${cpp_headers_json}}, "
                "\"python_imports\": ${python_imports_json}, \"api_languages\": ${api_languages_json}}"
            )
            set(module_separator ",")
        endforeach()
        string(APPEND modules_json "]")
        string(APPEND distributions_json
            "${distribution_separator}{\"name\": ${group_name_json}, \"complete\": ${group_complete_json}, "
            "\"dependencies\": ${dependencies_json}, \"documentation_root\": ${group_docs_json}, "
            "\"modules\": ${modules_json}}"
        )
        set(distribution_separator ",")
    endforeach()
    string(APPEND distributions_json "]")

    get_filename_component(catalog_directory "${ISAACSIM_DOCUMENTATION_CATALOG}" DIRECTORY)
    file(MAKE_DIRECTORY "${catalog_directory}")
    _isaacsim_json_quote(version_json "${ISAACSIM_LIBRARIES_VERSION}")
    file(WRITE "${ISAACSIM_DOCUMENTATION_CATALOG}"
        "{\n  \"schema_version\": 1,\n  \"release_version\": ${version_json},\n"
        "  \"distributions\": ${distributions_json}\n}\n"
    )
    message(STATUS "Generated library documentation catalog: ${ISAACSIM_DOCUMENTATION_CATALOG}")
endfunction()
