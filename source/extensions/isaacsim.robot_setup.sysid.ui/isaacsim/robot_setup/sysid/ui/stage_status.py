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

"""Per-stage readiness annotations for the workflow navigation.

Pure Python (no omni.ui dependency) so the title logic is testable headless.
The UI builder assigns the returned strings to the persistent stage summary,
giving a glanceable pipeline status while only one stage is visible at a time.
"""

from __future__ import annotations

CHECK_STATE_NOT_RUN = "not_run"
CHECK_STATE_OK = "ok"
CHECK_STATE_WARNINGS = "warnings"
CHECK_STATE_STALE = "stale"

_CHECK_STATE_TEXT = {
    CHECK_STATE_NOT_RUN: "not run",
    CHECK_STATE_OK: "ok",
    CHECK_STATE_WARNINGS: "completed with warnings",
    CHECK_STATE_STALE: "stale - re-run",
}


def format_stage_titles(
    *,
    engine_label: str,
    sample_count: int | None,
    num_joints: int | None,
    chunk_summary: str = "",
    selected_count: int = 0,
    check_state: str = CHECK_STATE_NOT_RUN,
    run_ready: bool = False,
) -> dict[str, str]:
    """Return stage-frame titles annotated with the current pipeline state.

    Keys: ``sim``, ``data``, ``params``, ``check``, ``solve``, and ``results``.
    ``sample_count`` is ``None`` while no trajectory is loaded.

    Args:
        engine_label: Engine label value.
        sample_count: Sample count value.
        num_joints: Num joints value.
        chunk_summary: Chunk summary value.
        selected_count: Selected count value.
        check_state: Check state value.
        run_ready: Run ready value.

    Returns:
        The resulting value.
    """
    if sample_count is None:
        data_detail = "not loaded"
    else:
        data_detail = f"{int(sample_count):,} samples, {int(num_joints or 0)} joints"
        if chunk_summary:
            data_detail += f", {chunk_summary}"
    params_detail = f"{int(selected_count)} selected" if selected_count else "none selected"
    check_detail = _CHECK_STATE_TEXT.get(check_state, check_state)
    solve_detail = "ready" if run_ready else "not ready"
    return {
        "sim": f"1. Robot — {engine_label}" if engine_label else "1. Robot",
        "data": f"2. Data — {data_detail}",
        "params": f"3. Parameters — {params_detail}",
        "check": f"4. Check — {check_detail}",
        "solve": f"5. Solve — {solve_detail}",
        "results": "6. Results",
    }


__all__ = [
    "CHECK_STATE_NOT_RUN",
    "CHECK_STATE_OK",
    "CHECK_STATE_STALE",
    "CHECK_STATE_WARNINGS",
    "format_stage_titles",
]
