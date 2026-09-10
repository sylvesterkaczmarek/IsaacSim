# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Test stage behavior."""

from typing import Any

import hypothesis
import hypothesis.strategies as st
import isaacsim_test
import numpy as np
import pytest
from isaacsim.common.exceptions import PrimPathError, PrimPathStringError
from isaacsim.foundation.objects import Stage
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdUtils

from .fixtures import stage  # noqa: F401 - imported so pytest can discover the fixture

"""
Utility functions.
"""


def _get_prim_at_path(stage: Any, path: Any) -> Any:
    return UsdUtils.StageCache.Get().Find(Usd.StageCache.Id.FromLongInt(stage.get_stage_id())).GetPrimAtPath(path)


"""
Test cases.
"""


def test_get_stage_ptr(capsys: Any) -> None:
    """Test get stage ptr.

    Args:
        capsys: Pytest output-capture fixture.
    """
    stage = Stage("openusd").create_stage()
    ptr = stage.get_stage_ptr()
    assert isinstance(ptr, int)
    assert ptr != 0
    stage.close_stage()
    # invalid stage returns 0
    assert stage.get_stage_ptr() == 0


def test_create(capsys: Any) -> None:
    """Test create.

    Args:
        capsys: Pytest output-capture fixture.
    """
    stage = Stage("openusd")
    reference = stage.create_stage()
    assert reference is stage
    assert stage.get_stage_id() != -1
    assert stage.is_valid()
    assert stage.close_stage()
    assert not stage.is_valid()
    assert not stage.close_stage()

    # successive creates produce different IDs
    stage1 = Stage("openusd").create_stage()
    stage2 = Stage("openusd").create_stage()
    assert stage1 is not stage2
    assert stage1.get_stage_id() != stage2.get_stage_id()
    assert stage1.close_stage()
    assert stage2.close_stage()


def test_open(capsys: Any) -> None:
    """Test open.

    Args:
        capsys: Pytest output-capture fixture.
    """
    usd_path = isaacsim_test.resolve_resource_path("scene.usda")
    stage = Stage("openusd")
    reference = stage.open_stage(usd_path)
    assert reference is stage
    assert stage.get_stage_id() != -1
    assert stage.is_valid()
    assert stage.close_stage()
    assert not stage.is_valid()
    assert not stage.close_stage()


def test_save(capsys: Any, tmp_path: Any) -> None:
    """Test save.

    Args:
        capsys: Pytest output-capture fixture.
        tmp_path: Temporary directory supplied by pytest.
    """
    output_path = tmp_path / "saved.usda"

    # save creates the file and returns True
    stage = Stage("openusd").create_stage()
    stage.define_prim("/World", "Xform")
    assert stage.save_stage(str(output_path))
    assert output_path.exists()
    assert stage.close_stage()

    # saved file can be reopened
    reopened = Stage("openusd").open_stage(str(output_path))
    assert reopened.is_valid()
    prim = _get_prim_at_path(reopened, "/World")
    assert prim.IsValid()
    assert reopened.close_stage()

    # saving on a closed stage raises
    closed = Stage("openusd").create_stage()
    closed.close_stage()
    with pytest.raises(RuntimeError):
        closed.save_stage(str(output_path))


def test_export_import_stage_string(capsys: Any) -> None:
    # export produces a non-empty USDA string
    """Test export import stage string.

    Args:
        capsys: Pytest output-capture fixture.
    """
    src = Stage("openusd").create_stage()
    src.define_prim("/World/Cube", "Cube")
    usda_string = src.export_stage_to_string()
    assert 'def Cube "Cube"' in usda_string
    src.close_stage()
    with pytest.raises(RuntimeError):
        src.export_stage_to_string()

    # import round-trips: new stage has the same prim
    dst = Stage("openusd").import_stage_from_string(usda_string)
    assert dst.is_valid()
    assert dst.get_stage_id() != -1
    assert _get_prim_at_path(dst, "/World/Cube").IsValid()
    dst.close_stage()

    # invalid USDA string
    assert Stage("openusd").import_stage_from_string("This is not valid USDA").get_stage_id() == -1


def test_add_reference(capsys: Any) -> None:
    """Test add reference.

    Args:
        capsys: Pytest output-capture fixture.
    """
    usd_path = isaacsim_test.resolve_resource_path("variant.usda")

    # add reference
    stage = Stage("openusd").create_stage()
    stage.add_reference(usd_path, "/World/Reference")
    prim = _get_prim_at_path(stage, "/World/Reference")
    assert prim.IsValid()
    stage.close_stage()

    # variant selections applied
    stage = Stage("openusd").create_stage()
    stage.add_reference(usd_path, "/World/Reference", variants={"color": "blue", "radius": "large"})
    prim = _get_prim_at_path(stage, "/World/Reference")
    assert prim.IsValid()
    variant_sets = prim.GetVariantSets()
    assert variant_sets.GetVariantSet("color").GetVariantSelection() == "blue"
    assert variant_sets.GetVariantSet("radius").GetVariantSelection() == "large"
    stage.close_stage()

    # exceptions
    stage = Stage("openusd").create_stage()
    with pytest.raises(PrimPathStringError):
        stage.add_reference(usd_path, "/World/")
    with pytest.raises(RuntimeError):
        stage.add_reference("/unknown/file.usda", "/World/Reference")
    stage.close_stage()


def test_define_prim(capsys: Any, stage: Any) -> None:
    """Test define prim.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
    """
    specs = [
        # UsdGeomTokensType
        ("Camera", UsdGeom.Camera),
        ("Capsule", UsdGeom.Capsule),
        ("Cone", UsdGeom.Cone),
        ("Cube", UsdGeom.Cube),
        ("Cylinder", UsdGeom.Cylinder),
        ("Mesh", UsdGeom.Mesh),
        ("Plane", UsdGeom.Plane),
        ("Points", UsdGeom.Points),
        ("Scope", UsdGeom.Scope),
        ("Sphere", UsdGeom.Sphere),
        ("Xform", UsdGeom.Xform),
        # UsdLuxTokensType
        ("CylinderLight", UsdLux.CylinderLight),
        ("DiskLight", UsdLux.DiskLight),
        ("DistantLight", UsdLux.DistantLight),
        ("DomeLight", UsdLux.DomeLight),
        ("RectLight", UsdLux.RectLight),
        ("SphereLight", UsdLux.SphereLight),
        # UsdPhysicsTokensType
        ("PhysicsScene", UsdPhysics.Scene),
    ]
    for token, prim_type in specs:
        stage.define_prim(f"/{token}", token)
        prim = _get_prim_at_path(stage, f"/{token}")
        assert prim.IsA(prim_type), f"Prim at path '{prim.GetPath()}' is not a {prim_type}"
    # redefining a prim with the same type does not throw
    stage.define_prim("/Sphere", "Sphere")
    # exceptions
    # - non-absolute path
    with pytest.raises(PrimPathStringError):
        stage.define_prim("World")
    # - non-valid path
    with pytest.raises(PrimPathStringError):
        stage.define_prim("/World/")
    # - prim already exists with a different type
    with pytest.raises(RuntimeError):
        stage.define_prim("/Sphere", "Cube")


def test_move_prim(capsys: Any, stage: Any) -> None:
    """Test move prim.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
    """
    stage.define_prim("/World/A")
    stage.define_prim("/World/B")
    # move A into B
    success, new_path = stage.move_prim("/World/A", "/World/B")
    assert success
    assert new_path == "/World/B/A" and _get_prim_at_path(stage, new_path).IsValid()
    assert not _get_prim_at_path(stage, "/World/A").IsValid()
    # move A next to B
    success, new_path = stage.move_prim("/World/B/A", "/World")
    assert success
    assert new_path == "/World/A" and _get_prim_at_path(stage, new_path).IsValid()
    assert not _get_prim_at_path(stage, "/World/B/A").IsValid()
    # move A to root
    success, new_path = stage.move_prim("/World/A", "/")
    assert success
    assert new_path == "/A" and _get_prim_at_path(stage, new_path).IsValid()
    assert not _get_prim_at_path(stage, "/World/A").IsValid()
    # move A to an unexisting path
    success, new_path = stage.move_prim("/A", "/World/C")
    assert success
    assert new_path == "/World/C" and _get_prim_at_path(stage, new_path).IsValid()
    assert not _get_prim_at_path(stage, "/A").IsValid()
    # exceptions
    # - target is not a valid prim
    with pytest.raises(PrimPathError):
        stage.move_prim("/NonExistent", "/")
    # - destination is not a valid path string
    with pytest.raises(PrimPathStringError):
        stage.move_prim("/World/B", "?")
    # - destination has unexisting parents
    with pytest.raises(ValueError):
        stage.move_prim("/World/B", "/World/X/Y")


def test_remove_prim(capsys: Any, stage: Any) -> None:
    # delete a locally defined prim
    """Test remove prim.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
    """
    stage.define_prim("/World/A")
    assert stage.remove_prim("/World/A")
    assert not _get_prim_at_path(stage, "/World/A").IsValid()
    # delete an ancestral prim (defined inside the reference)
    usd_path = isaacsim_test.resolve_resource_path("variant.usda")
    stage.add_reference(usd_path, "/World/Reference")
    assert not stage.remove_prim("/World/Reference/Cube")
    assert stage.remove_prim("/World/Reference")
    assert not _get_prim_at_path(stage, "/World/Reference").IsValid()
    # exceptions
    # - delete non-existent prim
    with pytest.raises(PrimPathError):
        stage.remove_prim("/NonExistent")


def test_up_axis(capsys: Any, stage: Any) -> None:
    # supported up axes (case-insensitive)
    """Test up axis.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
    """
    for up_axis in ["Y", "Z", "y", "z"]:
        stage.set_up_axis(up_axis)
        assert stage.get_up_axis() == up_axis.upper()
    # invalid up axis
    with pytest.raises(ValueError):
        stage.set_up_axis("X")


@hypothesis.given(
    meters_per_unit=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    kilograms_per_unit=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_units(capsys: Any, stage: Any, meters_per_unit: Any, kilograms_per_unit: Any) -> None:
    """Test units.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
        meters_per_unit: Meters per unit values.
        kilograms_per_unit: Kilograms per unit values.
    """
    stage.set_units(meters_per_unit=meters_per_unit, kilograms_per_unit=kilograms_per_unit)
    assert np.allclose(stage.get_units(), (meters_per_unit, kilograms_per_unit))


@hypothesis.given(
    start_time_code=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    end_time_code=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    time_codes_per_second=st.floats(min_value=1.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
)
def test_time_code(
    capsys: Any, stage: Any, start_time_code: Any, end_time_code: Any, time_codes_per_second: Any
) -> None:
    """Test time code.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
        start_time_code: Start time code values.
        end_time_code: End time code values.
        time_codes_per_second: Time codes per second values.
    """
    stage.set_time_code(
        start_time_code=start_time_code, end_time_code=end_time_code, time_codes_per_second=time_codes_per_second
    )
    assert np.allclose(stage.get_time_code(), (start_time_code, end_time_code, time_codes_per_second))
