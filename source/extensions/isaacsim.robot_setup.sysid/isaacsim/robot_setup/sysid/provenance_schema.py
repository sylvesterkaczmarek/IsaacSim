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

"""USD custom attribute names for SysId provenance."""

from __future__ import annotations

SYSID_PROVENANCE_SCHEMA_VERSION = "1.0"

# Searchable attributes written to the articulation root prim.
ATTR_SCHEMA_VERSION = "isaac:sysid:schemaVersion"
ATTR_DATASET_SOURCE_TYPE = "isaac:sysid:datasetSourceType"
ATTR_DATASET_SOURCE_PATH = "isaac:sysid:datasetSourcePath"
ATTR_OPTIMIZED_AT = "isaac:sysid:optimizedAt"
ATTR_OPTIMIZER_BACKEND = "isaac:sysid:optimizerBackend"
ATTR_FINAL_COST = "isaac:sysid:finalCost"
ATTR_ITERATIONS = "isaac:sysid:iterations"
ATTR_ROBOT_PRIM_PATH = "isaac:sysid:robotPrimPath"
ATTR_PARAMETER_NAMES = "isaac:sysid:parameterNames"
ATTR_PARAMETER_VALUES = "isaac:sysid:parameterValues"
ATTR_PARAMETER_CATEGORIES = "isaac:sysid:parameterCategories"
ATTR_PARAMETER_TYPES = "isaac:sysid:parameterTypes"
ATTR_SYSID_PARAMETER_NAMES = "isaac:sysid:sysidParameterNames"
ATTR_SYSID_PARAMETER_VALUES = "isaac:sysid:sysidParameterValues"
ATTR_CALIBRATION_PARAMETER_NAMES = "isaac:sysid:calibrationParameterNames"
ATTR_CALIBRATION_PARAMETER_VALUES = "isaac:sysid:calibrationParameterValues"
ATTR_PROVENANCE_JSON = "isaac:sysid:provenanceJson"

USD_ATTRIBUTE_NAMES: tuple[str, ...] = (
    ATTR_SCHEMA_VERSION,
    ATTR_DATASET_SOURCE_TYPE,
    ATTR_DATASET_SOURCE_PATH,
    ATTR_OPTIMIZED_AT,
    ATTR_OPTIMIZER_BACKEND,
    ATTR_FINAL_COST,
    ATTR_ITERATIONS,
    ATTR_ROBOT_PRIM_PATH,
    ATTR_PARAMETER_NAMES,
    ATTR_PARAMETER_VALUES,
    ATTR_PARAMETER_CATEGORIES,
    ATTR_PARAMETER_TYPES,
    ATTR_SYSID_PARAMETER_NAMES,
    ATTR_SYSID_PARAMETER_VALUES,
    ATTR_CALIBRATION_PARAMETER_NAMES,
    ATTR_CALIBRATION_PARAMETER_VALUES,
    ATTR_PROVENANCE_JSON,
)
