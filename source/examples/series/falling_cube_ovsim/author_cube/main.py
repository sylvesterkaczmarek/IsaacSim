# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author a cube on a new stage."""

from __future__ import annotations

from isaacsim.ovsim.api import make_client


def main() -> None:
    """Create a stage and author a cube."""
    client = make_client("in-process")  # local or in-process

    client.control.authoring.create_stage()

    cube = "/World/Cube"
    client.control.authoring.define_prim(cube, "Cube")
    client.data.write(cube, "size", 1.0)
    client.data.write(cube, "position", [0.0, 0.0, 2.0])
    client.data.write(cube, "color", "red")

    position = client.data.read(cube, "position").numpy()[0]
    print(f"Authored {cube} at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}).")


if __name__ == "__main__":
    main()
