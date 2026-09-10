<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# C++ with Python Hello World

This example builds an example-local Python extension with the Python C API. The extension calls the installed
`isaacsim.common.logging` C++ API, and `main.py` imports and invokes the extension.

The script exits with status zero and prints:

```text
[isaacsim.examples.hello_world.cpp_python] Hello World from C++ with Python.
```
