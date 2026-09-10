# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect license notices from Python distributions installed by Pixi."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from email.message import Message
from email.parser import Parser
from pathlib import Path

_LICENSE_FILE_PREFIXES = ("copying", "license", "notice", "third_party")
_NORMALIZED_NAME_PATTERN = re.compile(r"[-_.]+")


class PixiLicenseError(RuntimeError):
    """Error raised when installed distribution licenses cannot be collected."""


@dataclass
class _DistributionLicenses:
    """Collected license data for one installed distribution."""

    name: str
    version: str
    metadata: set[str] = field(default_factory=set)
    files: dict[str, str] = field(default_factory=dict)


def _read_distribution_metadata(metadata_path: Path) -> Message:
    """Read one installed distribution's core metadata."""
    try:
        return Parser().parsestr(metadata_path.read_text(encoding="utf-8", errors="replace"))
    except OSError as error:
        raise PixiLicenseError(f"Cannot read installed distribution metadata: {metadata_path}") from error


def _find_license_files(distribution_directory: Path) -> tuple[Path, ...]:
    """Find bundled license and notice files in one ``.dist-info`` directory."""
    root = distribution_directory.resolve()
    license_files: list[Path] = []
    for candidate in sorted(distribution_directory.rglob("*")):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        try:
            relative = candidate.relative_to(distribution_directory)
            candidate.resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        if relative.parts[0].casefold() == "licenses" or candidate.name.casefold().startswith(_LICENSE_FILE_PREFIXES):
            license_files.append(candidate)
    return tuple(license_files)


def _merge_distribution(
    distributions: dict[tuple[str, str], _DistributionLicenses], distribution_directory: Path
) -> None:
    """Merge one installed distribution into the aggregate license inventory."""
    metadata_path = distribution_directory / "METADATA"
    metadata = _read_distribution_metadata(metadata_path)
    name = metadata.get("Name", "").strip()
    version = metadata.get("Version", "").strip()
    if not name or not version:
        raise PixiLicenseError(f"Installed distribution metadata is missing Name or Version: {metadata_path}")

    key = (_NORMALIZED_NAME_PATTERN.sub("-", name).casefold(), version)
    record = distributions.setdefault(key, _DistributionLicenses(name=name, version=version))
    for value in metadata.get_all("License-Expression", []):
        if normalized := " ".join(value.split()):
            record.metadata.add(f"License-Expression: {normalized}")
    for license_path in _find_license_files(distribution_directory):
        relative = license_path.relative_to(distribution_directory).as_posix()
        try:
            contents = license_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as error:
            raise PixiLicenseError(f"Cannot read installed distribution license: {license_path}") from error
        if contents:
            existing = record.files.get(relative)
            if existing is not None and existing != contents:
                raise PixiLicenseError(
                    f"Installed distribution has conflicting license contents for {name} {version}: {relative}"
                )
            record.files[relative] = contents

    if not record.files and not record.metadata:
        for value in metadata.get_all("License", []):
            if normalized := " ".join(value.split()):
                record.metadata.add(f"License: {normalized}")
    if not record.files and not record.metadata:
        raise PixiLicenseError(f"Installed distribution does not provide license information: {name} {version}")


def _render_licenses(distributions: dict[tuple[str, str], _DistributionLicenses]) -> str:
    """Render a deterministic aggregate license document."""
    sections = ["Licenses for Python distributions installed from the locked Pixi environments."]
    for key in sorted(distributions):
        record = distributions[key]
        title = f"{record.name}-{record.version}"
        lines = [title, "-" * len(title)]
        lines.extend(sorted(record.metadata))
        for relative, contents in sorted(record.files.items()):
            if len(record.files) > 1:
                lines.extend(("", f"[{relative}]"))
            lines.extend(("", contents))
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def collect_python_licenses(site_packages_directories: Sequence[Path], output_path: Path) -> int:
    """Collect installed Python distribution licenses into one deterministic file.

    Args:
        site_packages_directories: Installed ``site-packages`` directories to inspect.
        output_path: Aggregate license file to replace atomically.

    Returns:
        Number of unique distribution name and version pairs included.

    Raises:
        PixiLicenseError: If installed metadata or license information is missing or unreadable.
    """
    distributions: dict[tuple[str, str], _DistributionLicenses] = {}
    for site_packages in sorted(set(site_packages_directories)):
        if not site_packages.is_dir():
            raise PixiLicenseError(f"Pixi site-packages directory is missing: {site_packages}")
        for distribution_directory in sorted(site_packages.glob("*.dist-info")):
            if distribution_directory.is_dir():
                _merge_distribution(distributions, distribution_directory)
    if not distributions:
        raise PixiLicenseError("Pixi environments do not contain installed Python distribution metadata")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary.write(_render_licenses(distributions))
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, output_path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise PixiLicenseError(f"Cannot write aggregate Pixi license file: {output_path}") from error
    return len(distributions)
