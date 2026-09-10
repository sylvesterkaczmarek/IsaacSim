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

"""Test viewport behavior."""

from isaacsim.ovgl_viewport.debug import Camera, ViewportConfig


def test_viewport_config_defaults() -> None:
    """Test viewport config defaults."""
    config = ViewportConfig()

    assert isinstance(config.camera, Camera)
    assert config.width == 1280
    assert config.height == 720
    assert config.visible
