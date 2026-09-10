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

import isaacsim.foundation.utils.stage as stage_utils
from isaacsim.foundation.objects import Stage

from .fixtures import stage  # noqa: F401 - imported so pytest can discover the fixture


def test_get_default_stage(stage: Any) -> None:
    """Test get default stage.

    Args:
        stage: Stage used by the test.
    """
    default = stage_utils.get_default_stage()
    assert default.get_stage_id() == stage.get_stage_id()


def test_set_default_stage(stage: Any) -> None:
    """Test set default stage.

    Args:
        stage: Stage used by the test.
    """
    other = Stage("openusd").create_stage(template=None, make_default=False)
    assert stage_utils.get_default_stage().get_stage_id() == stage.get_stage_id()
    try:
        stage_utils.set_default_stage(other)
        assert stage_utils.get_default_stage().get_stage_id() == other.get_stage_id()
    finally:
        stage_utils.set_default_stage(stage)
        other.close_stage()


def test_get_active_stage_falls_back_to_default(stage: Any) -> None:
    """Test get active stage falls back to default.

    Args:
        stage: Stage used by the test.
    """
    active = stage_utils.get_active_stage()
    assert active.get_stage_id() == stage.get_stage_id()


def test_use_stage(stage: Any) -> None:
    """Test use stage.

    Args:
        stage: Stage used by the test.
    """
    other = Stage("openusd").create_stage()
    stage_utils.set_default_stage(stage)  # create_stage() sets itself as default; restore ours
    try:
        with stage_utils.use_stage(other):
            assert stage_utils.get_active_stage().get_stage_id() == other.get_stage_id()
        assert stage_utils.get_active_stage().get_stage_id() == stage.get_stage_id()
    finally:
        other.close_stage()
