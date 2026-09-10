# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test data behavior."""

import numpy as np
from isaacsim.ovsim.api import make_client

client = make_client("local")
authoring = client.control.authoring
data = client.data


def test_read_write_float_attribute() -> None:
    """Test read write float attribute."""
    assert authoring.create_stage()
    assert authoring.define_prim("/World/A", "Xform")
    assert authoring.create_prim_attribute("/World/A", "myFloat", "float")

    values_in = np.array([3.14], dtype=np.float32)
    data.write("/World/A", "myFloat", values_in)
    result = data.read("/World/A", "myFloat")
    assert result is not None

    assert authoring.close_stage()


def test_write_string_attribute() -> None:
    """Test write string attribute."""
    assert authoring.create_stage()
    assert authoring.define_prim("/World/A", "Xform")
    assert authoring.create_prim_attribute("/World/A", "myString", "string")

    data.write("/World/A", "myString", ["hello"])
    result = data.read("/World/A", "myString")
    assert result is not None

    assert authoring.close_stage()


def test_read_write_with_timestamp() -> None:
    """Test read write with timestamp."""
    assert authoring.create_stage()
    assert authoring.define_prim("/World/A", "Xform")
    assert authoring.create_prim_attribute("/World/A", "myFloat", "float")

    values_in = np.array([1.0], dtype=np.float32)
    data.write("/World/A", "myFloat", values_in, 0.0)
    result = data.read("/World/A", "myFloat", 0.0)
    assert result is not None

    assert authoring.close_stage()


def test_read_write_multiple_paths() -> None:
    """Test read write multiple paths."""
    assert authoring.create_stage()
    assert authoring.define_prim("/World/A", "Xform")
    assert authoring.define_prim("/World/B", "Xform")
    assert authoring.create_prim_attribute("/World/A", "myFloat", "float")
    assert authoring.create_prim_attribute("/World/B", "myFloat", "float")

    values_in = np.array([1.0, 2.0], dtype=np.float32)
    data.write(["/World/A", "/World/B"], "myFloat", values_in)
    result = data.read(["/World/A", "/World/B"], "myFloat")
    assert result is not None

    assert authoring.close_stage()
