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

"""Provide shared test fixtures."""

from collections.abc import Iterator
from typing import Any

import pytest
from isaacsim.foundation.objects import Stage


@pytest.fixture(scope="function")
def stage(request: Any) -> Iterator[Stage]:
    """Create and destroy a stage for a test.

    Args:
        request: Pytest fixture request.

    Yields:
        The test stage.
    """
    stage = None
    if hasattr(request, "param") and "usd_path" in request.param:
        stage = Stage("openusd").open_stage(request.param["usd_path"])
    else:
        stage = Stage("openusd").create_stage()
    yield stage
    stage.close_stage()
