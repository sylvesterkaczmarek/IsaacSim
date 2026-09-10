# Shared helpers for validation-diff-gifs entry scripts.
# Sourced by generate_diff_gifs.sh and sdg_tolerance_diff_gifs.sh.
# Not executable on its own (no shebang).
#
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

diff_gif_require_imagemagick() {
    if ! command -v composite &>/dev/null || ! command -v convert &>/dev/null; then
        echo "ERROR: ImageMagick (composite, convert) is required but not found." >&2
        exit 1
    fi
}

# Build one amplified difference frame: composite then multiply.
# Args: <captured_png> <golden_png> <diff_tmp_path> <amplify>
diff_gif_make_frame() {
    local cap_png="$1"
    local golden_png="$2"
    local diff_tmp="$3"
    local amplify="$4"
    composite "$cap_png" "$golden_png" -compose difference "$diff_tmp"
    convert "$diff_tmp" -evaluate multiply "$amplify" -clamp "$diff_tmp"
}

# Write a looping GIF from space-separated frame paths, then delete temps.
# Args: <delay_cs> <gif_out> <frame_paths...>
diff_gif_write_animation() {
    local delay="$1"
    local gif_out="$2"
    shift 2
    local frames=("$@")
    convert -delay "$delay" -loop 0 "${frames[@]}" "$gif_out"
    rm -f "${frames[@]}"
}
