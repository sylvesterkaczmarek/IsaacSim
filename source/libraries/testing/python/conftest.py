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

"""Configure the standalone libraries test environment."""

import os
import pathlib
import sys

import hypothesis
import warp as wp

# ensure that Warp is initialized before running any tests
# to avoid displaying a message in the middle of the test output
wp.init()

# shared hypothesis settings for the whole test suite. Individual tests can still
# override any of these by applying their own `@hypothesis.settings(...)`
hypothesis.settings.register_profile(
    "isaacsim",
    max_examples=10,
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
)
hypothesis.settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "isaacsim"))

# ensure this directory is on sys.path so helpers.py can be found when pytest is invoked
_this_dir = str(pathlib.Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
