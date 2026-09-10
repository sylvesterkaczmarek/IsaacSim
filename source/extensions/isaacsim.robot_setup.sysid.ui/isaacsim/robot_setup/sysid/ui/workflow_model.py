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

"""Pure workflow-navigation helpers for the System Identification UI."""

from __future__ import annotations

WORKFLOW_STAGES = ("sim", "data", "params", "check", "solve", "results")
WORKFLOW_STAGE_LABELS = {
    "sim": "Robot",
    "data": "Data",
    "params": "Parameters",
    "check": "Check",
    "solve": "Solve",
    "results": "Results",
}


def normalize_stage(stage: str) -> str:
    """Return a valid workflow stage, falling back to the first stage.

    Args:
        stage: Candidate workflow stage identifier.

    Returns:
        A valid workflow stage identifier.
    """
    return stage if stage in WORKFLOW_STAGES else WORKFLOW_STAGES[0]


def adjacent_stage(stage: str, offset: int) -> str:
    """Return the adjacent workflow stage, clamped to workflow bounds.

    Args:
        stage: Current workflow stage identifier.
        offset: Signed number of stages to move.

    Returns:
        The target workflow stage identifier.
    """
    current = WORKFLOW_STAGES.index(normalize_stage(stage))
    target = max(0, min(len(WORKFLOW_STAGES) - 1, current + int(offset)))
    return WORKFLOW_STAGES[target]


__all__ = ["WORKFLOW_STAGE_LABELS", "WORKFLOW_STAGES", "adjacent_stage", "normalize_stage"]
