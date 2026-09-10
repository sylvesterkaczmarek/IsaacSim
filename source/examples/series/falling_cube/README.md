<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Falling Cube

From a source checkout's examples collection root, run the authoring, simulation, and visualization steps in order.
On Linux:

```bash
./example.sh run series/falling_cube/author_cube/main.py
./example.sh run series/falling_cube/simulate_cube/main.py
./example.sh run falling_cube.visualize_simulation
```

On Windows, use the same arguments with `example.bat`. The source launcher bootstraps the runner with Packman Python;
the runner then prepares and uses the configured library developer build.

In an installed examples workspace, activate the required Python environment and replace `./example.sh` with
`python examples.py` in these commands.
