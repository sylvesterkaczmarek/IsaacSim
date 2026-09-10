# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install the pinned stb header and license into a Pixi build prefix."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def main() -> None:
    """Copy the consumed stb files into the package prefix."""
    source_directory = Path(os.environ["SRC_DIR"])
    prefix = Path(os.environ["PREFIX"])
    include_directory = prefix / "Library" / "include" if os.name == "nt" else prefix / "include"
    license_directory = prefix / "licenses" / "stb"
    include_directory.mkdir(parents=True, exist_ok=True)
    license_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_directory / "stb_image.h", include_directory / "stb_image.h")
    shutil.copy2(source_directory / "LICENSE", license_directory / "LICENSE")


if __name__ == "__main__":
    main()
