#!/usr/bin/env python3
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

"""Minimal static-scene SDG pipeline using Isaac Sim Replicator.

Captures annotated RGB, depth, segmentation, and bounding-box frames to disk
via BasicWriter.  Run with $ISAAC_SIM_DIR/python.sh.
"""

import tempfile
from pathlib import Path

# DLSS execMode 2 == "Quality" preset (see rtx/post/dlss/execMode setting docs).
_DLSS_EXEC_MODE_QUALITY = 2
# Render product resolution (width, height) for the capture camera.
_RENDER_RESOLUTION = (1280, 720)
# Static-scene capture freezes the timeline: no time advances between frames.
_STATIC_DELTA_TIME = 0.0
_DEFAULT_OUTPUT_DIR = str(Path(tempfile.gettempdir()) / "sdg_output")


def run_minimal_sdg_pipeline(
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    num_frames: int = 10,
    rt_subframes: int = 32,
) -> None:
    """Run a minimal static-scene SDG capture loop.

    Args:
        output_dir: Directory to write captured frames.
        num_frames: Number of frames to capture.
        rt_subframes: Number of RT subframes per capture step.

    Raises:
        ValueError: If ``num_frames`` or ``rt_subframes`` is not a positive int,
            or ``output_dir`` is empty.
        RuntimeError: If the Isaac Sim assets root cannot be resolved.
    """
    # Validate inputs before touching the simulator so failures are cheap and explicit.
    if not output_dir:
        raise ValueError("output_dir must be a non-empty path")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if rt_subframes < 1:
        raise ValueError(f"rt_subframes must be >= 1, got {rt_subframes}")

    import carb.settings
    import isaacsim.core.experimental.utils.stage as stage_utils
    import omni.replicator.core as rep
    from isaacsim.core.experimental.utils.semantics import add_labels
    from isaacsim.storage.native import get_assets_root_path

    # 1. Scene
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", _DLSS_EXEC_MODE_QUALITY)
    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError(
            "Could not resolve Isaac Sim assets root. Check Nucleus/S3 asset-root "
            "configuration (see profile-isaac-sim prerequisites)."
        )
    stage_utils.open_stage(assets_root + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd")

    # 2. Semantic labels — add a referenced prop and tag it for annotators.
    stage_utils.add_reference_to_stage(
        usd_path=assets_root + "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
        path="/World/MyObj",
    )
    prim = stage_utils.get_current_stage().GetPrimAtPath("/World/MyObj")
    # Either API writes the UsdSemantics LabelsAPI schema on the prim:
    add_labels(prim, labels=["cracker_box"], taxonomy="class")
    # or, equivalently:
    # rep.functional.modify.semantics(prim, {"class": "cracker_box"}, mode="add")

    # 3. Camera + render product
    cam = rep.functional.create.camera(position=(3, 3, 2), look_at=(0, 0, 0), name="DataCam")
    rp = rep.create.render_product(cam, _RENDER_RESOLUTION, name="main_view")

    # 4. Writer (explicit backend form)
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=output_dir)
    writer = rep.writers.get("BasicWriter")
    writer.initialize(
        backend=backend,
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        distance_to_image_plane=True,
    )
    writer.attach(rp)

    # 5. Capture loop with randomization (static scene: delta_time=0.0 freezes timeline).
    #    Cleanup in finally so the writer/render product are released even if a
    #    capture step raises (avoids leaking backend handles across retries).
    try:
        for _ in range(num_frames):
            # Randomize object pose, camera, lights here
            rep.orchestrator.step(delta_time=_STATIC_DELTA_TIME, rt_subframes=rt_subframes)
        rep.orchestrator.wait_until_complete()
    finally:
        # 6. Cleanup
        writer.detach()
        rp.destroy()
