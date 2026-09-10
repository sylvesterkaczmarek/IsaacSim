# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install the pinned public OVPhysX SDK archive into a Pixi build prefix."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def main() -> None:
    """Copy the complete relocatable OVPhysX SDK below the package prefix."""
    source_directory = Path(os.environ["SRC_DIR"])
    sdk_directory = source_directory / "ovphysx"
    if not (sdk_directory / "include" / "ovphysx").is_dir():
        sdk_directory = source_directory
    if not (sdk_directory / "lib" / "cmake" / "ovphysx").is_dir():
        raise RuntimeError(f"The OVPhysX SDK archive has an unexpected layout: {source_directory}")

    destination = Path(os.environ["PREFIX"]) / "ovphysx"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(sdk_directory, destination, symlinks=True)


if __name__ == "__main__":
    main()
