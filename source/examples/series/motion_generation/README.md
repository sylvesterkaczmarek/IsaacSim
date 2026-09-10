<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Motion Generation

This series demonstrates backend-neutral motion-generation controllers driving robots in a live physics simulation.

The pick-and-place example time-parameterizes a Franka joint-space path, applies its commands through OvPhysX, and
renders the measured robot and object poses with the OVGL debug viewport. Run it from the examples collection root:

```bash
python examples.py run motion_generation.pick_place
```
