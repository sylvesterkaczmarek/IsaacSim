# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Python Hello World example."""

from isaacsim.common.logging import Logger


def main() -> int:
    """Print the Hello World result through the installed Isaac Sim Python API.

    Returns:
        Process exit code.
    """
    logger = Logger("isaacsim.examples.hello_world.python")
    logger.report("Hello World from Python.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
