# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble the packaged third-party notice from the public notice inventory."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = REPOSITORY_ROOT / "THIRD_PARTY_NOTICE.md"
DEFAULT_LICENSE_DIRECTORY = REPOSITORY_ROOT / "licenses"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "_build" / "generated" / "ThirdPartyNotices.txt"

TABLE_HEADER = "| Package | Version | Ecosystem | License | Homepage | Surfaces | Required by | License text |"
TABLE_SEPARATOR = "| --- | --- | --- | --- | --- | --- | --- | --- |"
LICENSE_LINK_RE = re.compile(r"^.*\| \[([A-Za-z0-9._+-]+\.txt)\]\(licenses/([A-Za-z0-9._+-]+\.txt)\) \|$")
LICENSE_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\.txt")
NOTICE_PREAMBLE = (
    "THIRD-PARTY NOTICES",
    "===================",
    "",
    "This file is assembled from the generated third-party package inventory and its full license texts.",
    "Coverage is limited to the package records listed below from selected public release payloads. Aggregate "
    "Kit records preserve packaged third-party notice bundles when individual embedded component versions are "
    "unavailable. Do not edit it by hand.",
    "",
)
METADATA_PREFIXES = ("Package: ", "Version: ", "Ecosystem: ", "License: ", "Homepage: ")


class AssemblyError(RuntimeError):
    """Raised when the tracked notice inventory cannot be assembled safely."""


def _read_index_filenames(index_path: Path) -> tuple[str, ...]:
    try:
        contents = index_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise AssemblyError(f"Cannot read the notice index {index_path}: {error}") from error

    lines = contents.splitlines()
    if lines.count(TABLE_HEADER) != 1:
        raise AssemblyError(f"Notice index must contain exactly one generated package table: {index_path}")
    header_index = lines.index(TABLE_HEADER)
    if header_index + 1 >= len(lines) or lines[header_index + 1] != TABLE_SEPARATOR:
        raise AssemblyError(f"Notice index has an invalid generated package table header: {index_path}")

    filenames: list[str] = []
    seen_filenames: set[str] = set()
    for line_number, line in enumerate(lines[header_index + 2 :], start=header_index + 3):
        match = LICENSE_LINK_RE.fullmatch(line)
        if match is None:
            raise AssemblyError(f"Notice index has an invalid package row at {index_path}:{line_number}")
        display_name, target_name = match.groups()
        if display_name != target_name:
            raise AssemblyError(f"Notice index license link text and target differ at {index_path}:{line_number}")
        if target_name in seen_filenames:
            raise AssemblyError(f"Notice index contains duplicate license links: {target_name}")
        seen_filenames.add(target_name)
        filenames.append(target_name)

    if not filenames:
        raise AssemblyError(f"Notice index contains no package rows: {index_path}")
    return tuple(filenames)


def _list_license_filenames(license_directory: Path) -> set[str]:
    if not license_directory.is_dir():
        raise AssemblyError(f"License directory does not exist: {license_directory}")

    filenames: set[str] = set()
    for path in license_directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise AssemblyError(f"License directory contains a non-regular file: {path}")
        if LICENSE_FILENAME_RE.fullmatch(path.name) is None:
            raise AssemblyError(f"License directory contains an invalid filename: {path.name}")
        filenames.add(path.name)
    return filenames


def _read_license_record(path: Path) -> tuple[str, str, str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise AssemblyError(f"Cannot read license record {path}: {error}") from error
    if b"\r" in raw or not text.endswith("\n"):
        raise AssemblyError(f"License record must use LF line endings and end with a newline: {path}")

    lines = text.splitlines()
    if len(lines) < len(METADATA_PREFIXES) + 2:
        raise AssemblyError(f"License record is incomplete: {path}")
    values: list[str] = []
    for line, prefix in zip(lines[: len(METADATA_PREFIXES)], METADATA_PREFIXES, strict=True):
        if not line.startswith(prefix) or line == prefix:
            raise AssemblyError(f"License record has invalid generated metadata: {path}")
        values.append(line.removeprefix(prefix))
    if lines[len(METADATA_PREFIXES)] != "":
        raise AssemblyError(f"License record metadata is not followed by a blank line: {path}")
    return values[0], values[1], text


def render_third_party_notice(index_path: Path, license_directory: Path) -> str:
    """Render one deterministic packaged notice from the tracked index and license files."""
    filenames = _read_index_filenames(index_path)
    indexed = set(filenames)
    available = _list_license_filenames(license_directory)
    if missing := sorted(indexed - available):
        raise AssemblyError(f"Notice index references missing license files: {', '.join(missing)}")
    if extra := sorted(available - indexed):
        raise AssemblyError(f"License directory contains unindexed files: {', '.join(extra)}")

    lines = list(NOTICE_PREAMBLE)
    for index, filename in enumerate(filenames, start=1):
        package, version, license_text = _read_license_record(license_directory / filename)
        heading = f"{index}. {package} {version}"
        lines.extend((heading, "-" * len(heading), license_text.rstrip(), "", ""))
    return "\n".join(lines).rstrip() + "\n"


def write_third_party_notice(index_path: Path, license_directory: Path, output_path: Path) -> None:
    """Atomically write the packaged notice assembled from the tracked public inventory."""
    output = render_third_party_notice(index_path, license_directory).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(output)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(arguments: list[str] | None = None) -> int:
    """Assemble the packaged notice from command-line paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--licenses", type=Path, default=DEFAULT_LICENSE_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    options = parser.parse_args(arguments)
    try:
        write_third_party_notice(options.index, options.licenses, options.output)
    except AssemblyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Assembled third-party notices at {options.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
