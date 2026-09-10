# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the C++ with Python Hello World example."""

import _hello_world_cpp


def main() -> int:
    """Call the example-local C++ binding.

    Returns:
        Process exit code.
    """
    _hello_world_cpp.say_hello()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
