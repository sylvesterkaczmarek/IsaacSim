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

"""Test control behavior."""

from typing import Any

import isaacsim.foundation.ovsim as ovsim
import isaacsim_test
import pytest

authoring = ovsim.control.authoring
simulation = ovsim.control.simulation


"""
Stage operations.
"""


def test_create_stage() -> None:
    """Test create stage."""
    assert authoring.create_stage()
    assert authoring.get_parameter("stage", "openusd-stage-id") != -1
    assert authoring.get_parameter("stage", "openusd-stage-ptr") != 0
    assert authoring.get_parameter("stage", "ovstage-stage-id") == -1
    assert authoring.get_parameter("stage", "ovstage-stage-ptr") == 0
    assert authoring.close_stage()


def test_open_stage() -> None:
    """Test open stage."""
    usd_path = isaacsim_test.resolve_resource_path("variant.usda")
    assert authoring.open_stage(usd_path)
    assert authoring.get_parameter("stage", "openusd-stage-id") != -1
    assert authoring.get_parameter("stage", "openusd-stage-ptr") != 0
    assert authoring.get_parameter("stage", "ovstage-stage-id") == -1
    assert authoring.get_parameter("stage", "ovstage-stage-ptr") == 0
    assert authoring.close_stage()


def test_save_stage(tmp_path: Any) -> None:
    """Test save stage.

    Args:
        tmp_path: Temporary directory supplied by pytest.
    """
    output_path = str(tmp_path / "saved.usda")
    assert authoring.create_stage()
    assert authoring.define_prim("/World", "Xform")
    assert authoring.save_stage(output_path)
    assert authoring.close_stage()


def test_close_stage() -> None:
    """Test close stage."""
    assert authoring.create_stage()
    assert authoring.get_parameter("stage", "openusd-stage-id") != -1
    assert authoring.get_parameter("stage", "openusd-stage-ptr") != 0
    assert authoring.close_stage()
    assert authoring.get_parameter("stage", "openusd-stage-id") == -1
    assert authoring.get_parameter("stage", "openusd-stage-ptr") == 0
    with pytest.raises(RuntimeError):
        assert authoring.close_stage()


def test_add_reference_to_stage() -> None:
    """Test add reference to stage."""
    usd_path = isaacsim_test.resolve_resource_path("variant.usda")
    assert authoring.create_stage()
    assert authoring.define_prim("/World/Prim")
    assert authoring.add_reference_to_stage(usd_path, "/World/Prim")
    assert authoring.close_stage()


def test_import_export_stage_from_to_string() -> None:
    """Test import export stage from to string."""
    assert authoring.create_stage()
    assert authoring.define_prim("/World", "Xform")

    usd_string = authoring.export_stage_to_string()
    assert authoring.close_stage()
    assert 'def Xform "World"' in usd_string

    assert authoring.import_stage_from_string(usd_string)
    assert authoring.close_stage()


"""
Prim operations.
"""


def test_define_move_remove_prim() -> None:
    """Test define move remove prim."""
    assert authoring.create_stage()

    assert authoring.define_prim("/World/A", "Xform")
    assert authoring.move_prim("/World/A", "/World/B")
    assert authoring.remove_prim("/World/B")

    assert authoring.close_stage()


def test_create_remove_prim_attribute() -> None:
    """Test create remove prim attribute."""
    assert authoring.create_stage()

    assert authoring.define_prim("/World/A", "Xform")
    assert authoring.create_prim_attribute("/World/A", "attribute1", "float")
    assert authoring.remove_prim_attribute("/World/A", "attribute1")

    with pytest.raises(Exception):
        authoring.create_prim_attribute("/World/A", "attribute2", "nonExistentType")
    with pytest.raises(Exception):
        authoring.remove_prim_attribute("/World/A", "attribute2")

    assert authoring.close_stage()


"""
Authoring parameter operations.
"""


def test_authoring_parameters() -> None:
    """Test authoring parameters."""
    assert authoring.create_stage()

    assert authoring.get_parameter("stage", "openusd-stage-id") != -1
    assert authoring.get_parameter("stage", "openusd-stage-ptr") != 0
    assert authoring.get_parameter("stage", "ovstage-stage-id") == -1
    assert authoring.get_parameter("stage", "ovstage-stage-ptr") == 0

    authoring.set_parameter("stage", "openusd-stage-id", 1)
    authoring.set_parameter("stage", "openusd-stage-ptr", 1.0)
    authoring.set_parameter("stage", "ovstage-stage-id", 1)
    authoring.set_parameter("stage", "ovstage-stage-ptr", 1.0)

    assert authoring.close_stage()


"""
Simulation operations.
"""


def test_simulation_automatic_mode() -> None:
    """Test simulation automatic mode."""
    simulation.play()
    simulation.pause()
    simulation.stop()


def test_simulation_manual_mode() -> None:
    """Test simulation manual mode."""
    assert authoring.create_stage()

    simulation.initialize()
    simulation.step()
    simulation.invalidate()
    simulation.invalidate()  # invalidate without a prior initialize is a no-op

    assert authoring.close_stage()


def test_simulation_parameters() -> None:
    """Test simulation parameters."""
    with pytest.raises(Exception):
        simulation.set_parameter("provider", "parameter", 1)
    with pytest.raises(Exception):
        simulation.get_parameter("provider", "parameter")
