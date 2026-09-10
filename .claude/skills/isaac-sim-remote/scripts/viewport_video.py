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

"""Record the viewport to an MP4 video and save it to disk.

Video counterpart to viewport_screenshot.py. Uses Kit's native Movie Capture
pipeline (omni.kit.capture.viewport + omni.videoencoding) — no external ffmpeg.
Records the viewport render over timeline frames [start_frame, end_frame] as the
USD timeline advances, encoding directly to MP4.

This is timeline-driven: it advances the timeline itself and captures each frame.
It does NOT capture inside a manual world.step() control loop — for in-loop
physics/contact evidence, capture frames with viewport_screenshot.py inside the
loop and encode them with omni.videoencoding's encode_image_file_sequence().

Requires both extensions enabled at launch (or auto-enabled below):
    --enable omni.kit.capture.viewport --enable omni.videoencoding

Injected globals (via isaacsim_send.py --arg, all strings — cast below):
    output_path: str — Output .mp4 path (default: under tempfile.gettempdir()).
    start_frame: int — First timeline frame (default: 1).
    end_frame:   int — Last timeline frame (default: 48).
    fps:         int — Frames per second of the output video (default: 24).
    width:       int — Capture width (default: 1280).
    height:      int — Capture height (default: 720).
    camera:      str — Camera prim path (default: "" = active viewport camera).
"""

import os
import tempfile
from pathlib import Path

# Defaults (overridden by injected globals)
if "output_path" not in dir():
    output_path = str(Path(tempfile.gettempdir()) / "viewport_video.mp4")  # noqa: F841
if "start_frame" not in dir():
    start_frame = 1  # noqa: F841
if "end_frame" not in dir():
    end_frame = 48  # noqa: F841
if "fps" not in dir():
    fps = 24  # noqa: F841
if "width" not in dir():
    width = 1280  # noqa: F841
if "height" not in dir():
    height = 720  # noqa: F841
if "camera" not in dir():
    camera = ""  # noqa: F841


async def _record():
    import isaacsim.core.experimental.utils.app as app_utils

    # Native Kit capture + encoder (no external ffmpeg).
    app_utils.enable_extension("omni.videoencoding")
    app_utils.enable_extension("omni.kit.capture.viewport")
    await app_utils.update_app_async(steps=5)

    from omni.kit.capture.viewport import (
        CaptureExtension,
        CaptureMovieType,
        CaptureOptions,
        CaptureRangeType,
        CaptureRenderPreset,
    )
    from omni.kit.capture.viewport import extension as capture_ext

    folder = os.path.dirname(output_path) or tempfile.gettempdir()
    file_name, file_type = os.path.splitext(os.path.basename(output_path))
    if file_type.lower() != ".mp4":
        file_type = ".mp4"
    os.makedirs(folder, exist_ok=True)

    # Use the live instance created on extension startup; build one if absent.
    capture = getattr(capture_ext, "capture_instance", None)
    if capture is None:
        capture = CaptureExtension()
        capture.on_startup()

    opts = CaptureOptions()
    opts.range_type = CaptureRangeType.FRAMES
    opts.start_frame = int(float(start_frame))
    opts.end_frame = int(float(end_frame))
    _fps = int(float(fps))
    opts.fps = _fps  # output (encode/playback) frame rate
    opts.animation_fps = _fps  # timeline sampling rate (keeps playback real-time)
    opts.res_width = int(float(width))
    opts.res_height = int(float(height))
    opts.output_folder = folder
    opts.file_name = file_name
    opts.file_type = file_type
    opts.movie_type = CaptureMovieType.SEQUENCE
    opts.render_preset = CaptureRenderPreset.RAY_TRACE  # real-time RTX (RaytracedLighting)
    opts.overwrite_existing_frames = True
    if camera:
        opts.camera = camera

    capture.options = opts
    capture.show_default_progress_window = False
    if not capture.start():
        raise RuntimeError("CaptureExtension.start() failed — check capture options / active viewport.")

    # Pump the app until render + encode finish.
    while not capture.done:
        await app_utils.update_app_async()

    print("VIEWPORT_VIDEO=" + os.path.join(folder, file_name + file_type))


await _record()
