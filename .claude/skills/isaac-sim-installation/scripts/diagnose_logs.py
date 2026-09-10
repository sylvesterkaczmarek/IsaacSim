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

"""Scan Isaac Sim install or launch logs for likely root causes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PATTERNS: list[tuple[str, str, str]] = [
    (
        r"No matching distribution found.*isaacsim|Could not find a version.*isaacsim",
        "Python package resolution failed",
        "Check Python 3.12, platform support, package version, and --extra-index-url https://pypi.nvidia.com.",
    ),
    (
        r"Do you accept the EULA|EULA.*accept|license agreement",
        "EULA acceptance is missing",
        "Set OMNI_KIT_ACCEPT_EULA=YES for Python or ACCEPT_EULA=Y for Docker before launch/import.",
    ),
    (
        r"nvidia-container-cli|could not select device driver|no NVIDIA GPU|GPU.*not found",
        "Docker GPU runtime is not available",
        "Validate nvidia-smi and NVIDIA Container Toolkit, then restart Docker.",
    ),
    (
        (
            r"\bDISPLAY\b[^\r\n]*(?:not set|unset|missing|invalid|failed|error)"
            r"|(?:failed|unable|cannot|could not) to (?:open|connect to)[^\r\n]*display"
            r"|\bX11\b[^\r\n]*(?:error|failed|denied|rejected|unavailable)"
            r"|\bXAUTHORITY\b[^\r\n]*(?:not set|missing|invalid|failed|error)"
            r"|\bWayland\b[^\r\n]*(?:error|failed|unavailable)"
            r"|\bGLX(?:Bad\w+|[^\r\n]*(?:error|failed))"
        ),
        "Display or graphics session failure",
        "Verify DISPLAY, Xauthority, X11/Wayland state, and NVIDIA driver display stack.",
    ),
    (
        (
            r"\bWebRTC\b[^\r\n]*(?:error|failed|failure|unable|unavailable|cannot|timeout|refused)"
            r"|\bICE\b[^\r\n]*(?:error|failed|failure|disconnected|timeout)"
            r"|\bsignaling\b[^\r\n]*(?:error|failed|failure|unable|unavailable|timeout)"
            r"|(?:49100|47998)[^\r\n]*(?:address already in use|bind|blocked|closed|refused|timeout|unreachable)"
            r"|(?:failed|unable|cannot|could not)[^\r\n]*(?:WebRTC|signaling|stream(?:ing)?)"
        ),
        "WebRTC networking failure",
        "Set ISAACSIM_HOST to a reachable interface and ensure 49100/tcp plus 47998/udp are reachable "
        "through the container's published port mappings under bridge networking.",
    ),
    (
        r"Segmentation fault|core dumped|Fatal Python error|crash",
        "Isaac Sim crashed during startup",
        "Check driver/runtime compatibility and inspect earlier log lines for the first failure before the crash.",
    ),
    (
        r"ModuleNotFoundError|ImportError",
        "Missing Python dependency",
        "Install the missing package in the same environment used to launch Isaac Sim.",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose Isaac Sim install or launch logs.",
        epilog="Exit codes: 0=no known issue, 1=known issue found, 2=log scan error.",
    )
    parser.add_argument("logs", nargs="+", help="Log files to scan.")
    args = parser.parse_args()

    found = False
    operational_error = False
    for log in args.logs:
        path = Path(log).expanduser()
        if not path.exists():
            print(f"{path}: missing")
            operational_error = True
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"{path}: could not read: {exc}")
            operational_error = True
            continue
        print(f"\n{path}")
        matched_file = False
        for pattern, cause, fix in PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                found = True
                matched_file = True
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                if line_end == -1:
                    line_end = len(text)
                evidence = text[line_start:line_end].strip()
                print(f"- Root cause: {cause}")
                print(f"- Evidence: {evidence[:500]}")
                print(f"- Fix: {fix}")
        if not matched_file:
            print("- No known install pattern matched. Inspect the first ERROR/FATAL/Exception line.")

    if operational_error:
        return 2
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
