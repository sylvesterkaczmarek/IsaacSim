#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import tempfile

MAX_JAVA_INT = 2_147_483_647
HITS_RE = re.compile(r'hits="([0-9]+)"')


def clamp_hits(match: re.Match[str]) -> tuple[str, bool]:
    hits_count = int(match.group(1))
    if hits_count <= MAX_JAVA_INT:
        return match.group(0), False

    return f'hits="{MAX_JAVA_INT}"', True


def main() -> None:
    parser = argparse.ArgumentParser(description="Clamp Cobertura line hit counts to Sonar's supported integer range")
    parser.add_argument("coverage_report", help="Cobertura XML coverage report to update in place")
    args = parser.parse_args()

    clamped_hits = 0

    def replace_hits(match: re.Match[str]) -> str:
        nonlocal clamped_hits
        replacement, was_clamped = clamp_hits(match)
        if was_clamped:
            clamped_hits += 1
        return replacement

    report_dir = os.path.dirname(os.path.abspath(args.coverage_report))
    with tempfile.NamedTemporaryFile("w", dir=report_dir, delete=False, encoding="utf-8") as temp_report:
        temp_path = temp_report.name
        with open(args.coverage_report, encoding="utf-8") as coverage_report:
            for line in coverage_report:
                temp_report.write(HITS_RE.sub(replace_hits, line))

    os.replace(temp_path, args.coverage_report)
    print(f"Clamped {clamped_hits} Cobertura hit counts above {MAX_JAVA_INT}")


if __name__ == "__main__":
    main()
