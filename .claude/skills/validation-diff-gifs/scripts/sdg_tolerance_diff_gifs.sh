#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# sdg_tolerance_diff_gifs.sh — flat SDG layout + optional QA filter. Full Apache-2.0 grant text:
# skills/validation-diff-gifs/scripts/diff_gif_common.inc

# Generate animated difference GIFs for SDG golden-set validation failures.
#
# Usage:
#   ./sdg_tolerance_diff_gifs.sh <captured_dir> <golden_dir> [qa_report] [amplify] [fps]
#
#   captured_dir   SDG output directory (contains rgb/, distance_to_image_plane/, etc.)
#   golden_dir     Golden reference directory (same flat layout)
#   qa_report      Path to qa_report.json from production_capture_qa.py (optional).
#                  When provided, only frames that failed threshold checks are diffed.
#                  Pass "-" to skip (diff all frames).
#   amplify        Multiply difference pixel values by this factor (default: 10)
#   fps            Frames per second for the output GIF (default: 5)
#
# Output:
#   <captured_dir>/rgb/diff_animation.gif           — all (or failed-only) RGB diffs
#   <captured_dir>/rgb/diff_failed_frames.txt       — list of diffed frame indices (when qa_report used)
#   <captured_dir>/distance_to_image_plane/diff_animation.gif — depth diffs (if golden exists)
#
# Requires: ImageMagick (composite, convert), jq (only if qa_report is provided)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=diff_gif_common.inc
source "${SCRIPT_DIR}/diff_gif_common.inc"

CAPTURED_DIR="${1:?Usage: $0 <captured_dir> <golden_dir> [qa_report] [amplify] [fps]}"
GOLDEN_DIR="${2:?Usage: $0 <captured_dir> <golden_dir> [qa_report] [amplify] [fps]}"
QA_REPORT="${3:-}"
AMPLIFY="${4:-10}"
FPS="${5:-5}"
DELAY=$((100 / FPS))

diff_gif_require_imagemagick

# Build list of failed frame indices from qa_report.json if provided.
declare -A FAILED_FRAMES=()
if [ -n "$QA_REPORT" ] && [ "$QA_REPORT" != "-" ]; then
    if [ ! -f "$QA_REPORT" ]; then
        echo "ERROR: QA report not found: $QA_REPORT" >&2
        exit 1
    fi
    if ! command -v jq &>/dev/null; then
        echo "ERROR: jq is required when using --qa-report filtering." >&2
        exit 1
    fi
    while IFS= read -r idx; do
        FAILED_FRAMES["$idx"]=1
    done < <(jq -r '.frame_metrics[] | select(.passed == false) | .frame' "$QA_REPORT")
    echo "QA report loaded: ${#FAILED_FRAMES[@]} failed frames will be diffed"
    if [ "${#FAILED_FRAMES[@]}" -eq 0 ]; then
        echo "All frames passed QA thresholds — no diffs to generate."
        exit 0
    fi
fi

found=0

# Process each annotation type that has PNG outputs.
for annot_type in rgb semantic_segmentation; do
    cap_annot_dir="$CAPTURED_DIR/$annot_type"
    golden_annot_dir="$GOLDEN_DIR/$annot_type"

    if [ ! -d "$cap_annot_dir" ]; then
        continue
    fi
    if [ ! -d "$golden_annot_dir" ]; then
        echo "SKIP $annot_type (no matching golden directory)"
        continue
    fi

    echo "Processing: $annot_type"

    diff_frames=()
    matched=0
    skipped=0
    filtered=0

    for cap_png in $(ls "$cap_annot_dir"/*.png 2>/dev/null | sort -V); do
        basename_file=$(basename "$cap_png")
        golden_png="$golden_annot_dir/$basename_file"

        if [ ! -f "$golden_png" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        # Extract frame index from filename (e.g., rgb_0042.png -> 42).
        frame_idx=$(echo "$basename_file" | grep -oP '\d+' | tail -1)
        frame_idx=$((10#$frame_idx))

        # If we have a QA report, only process failed frames.
        if [ ${#FAILED_FRAMES[@]} -gt 0 ] && [ -z "${FAILED_FRAMES[$frame_idx]:-}" ]; then
            filtered=$((filtered + 1))
            continue
        fi

        diff_tmp="$cap_annot_dir/.diff_tmp_${basename_file}"
        diff_gif_make_frame "$cap_png" "$golden_png" "$diff_tmp" "$AMPLIFY"
        diff_frames+=("$diff_tmp")
        matched=$((matched + 1))
    done

    if [ "${#diff_frames[@]}" -gt 0 ]; then
        diff_gif_write_animation "$DELAY" "$cap_annot_dir/diff_animation.gif" "${diff_frames[@]}"
        echo "  Created: $cap_annot_dir/diff_animation.gif ($matched frames, ${AMPLIFY}x amplified, ${FPS}fps)"
        found=$((found + 1))

        # Write out which frames were diffed when filtering is active.
        if [ ${#FAILED_FRAMES[@]} -gt 0 ]; then
            printf '%s\n' "${!FAILED_FRAMES[@]}" | sort -n > "$cap_annot_dir/diff_failed_frames.txt"
            echo "  Failed frame indices: $cap_annot_dir/diff_failed_frames.txt"
        fi
    else
        echo "  No matching frames found"
    fi
    [ "$skipped" -gt 0 ] && echo "  ($skipped captured frames had no golden match)"
    [ "$filtered" -gt 0 ] && echo "  ($filtered frames skipped — passed QA thresholds)"
done

# Process depth maps (npy → grayscale PNG → diff).
depth_cap="$CAPTURED_DIR/distance_to_image_plane"
depth_golden="$GOLDEN_DIR/distance_to_image_plane"
if [ -d "$depth_cap" ] && [ -d "$depth_golden" ]; then
    echo "Processing: distance_to_image_plane (depth)"
    # Depth is stored as .npy; we need Python to convert to visual PNGs for diffing.
    python3 - "$depth_cap" "$depth_golden" "$AMPLIFY" "$DELAY" "$(printf '%s,' "${!FAILED_FRAMES[@]}")" <<'PYEOF'
import sys, os, glob
import numpy as np
from pathlib import Path

depth_cap_dir = Path(sys.argv[1])
depth_golden_dir = Path(sys.argv[2])
amplify = int(sys.argv[3])
delay = int(sys.argv[4])
failed_csv = sys.argv[5].rstrip(',')
failed_set = set(int(x) for x in failed_csv.split(',') if x)

cap_files = sorted(depth_cap_dir.glob("*.npy"))
if not cap_files:
    print("  No depth .npy files found")
    sys.exit(0)

diff_pngs = []
for cap_f in cap_files:
    golden_f = depth_golden_dir / cap_f.name
    if not golden_f.exists():
        continue
    # Extract frame index.
    digits = ''.join(c for c in cap_f.stem if c.isdigit())
    if not digits:
        continue
    frame_idx = int(digits)
    if failed_set and frame_idx not in failed_set:
        continue

    cap_depth = np.load(str(cap_f))
    golden_depth = np.load(str(golden_f))
    # Compute absolute difference, ignoring NaN.
    with np.errstate(invalid='ignore'):
        diff = np.abs(cap_depth - golden_depth)
    diff = np.nan_to_num(diff, nan=0.0)
    # Normalize to 0-255 range with amplification.
    diff_clamped = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    out_path = depth_cap_dir / f".diff_tmp_{cap_f.stem}.png"
    try:
        from PIL import Image
        Image.fromarray(diff_clamped).save(str(out_path))
    except ImportError:
        import subprocess
        # Fallback: write raw PGM.
        h, w = diff_clamped.shape[:2]
        pgm_path = str(out_path).replace('.png', '.pgm')
        with open(pgm_path, 'wb') as f:
            f.write(f"P5\n{w} {h}\n255\n".encode())
            f.write(diff_clamped.tobytes())
        subprocess.run(["convert", pgm_path, str(out_path)], check=True)
        os.remove(pgm_path)
    diff_pngs.append(str(out_path))

if diff_pngs:
    import subprocess
    gif_path = str(depth_cap_dir / "diff_animation.gif")
    cmd = ["convert", "-delay", str(delay), "-loop", "0"] + diff_pngs + [gif_path]
    subprocess.run(cmd, check=True)
    for p in diff_pngs:
        os.remove(p)
    print(f"  Created: {gif_path} ({len(diff_pngs)} frames)")
else:
    print("  No matching depth frames")
PYEOF
    found=$((found + 1))
fi

if [ "$found" -eq 0 ]; then
    echo "WARNING: No difference GIFs generated. Check that directory structures match." >&2
    exit 1
fi

echo ""
echo "Done. Generated difference GIFs for $found annotation type(s)."
