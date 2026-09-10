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

"""Request a graceful shutdown of the running Isaac Sim application.

Injected globals (via ``isaacsim_send.py --arg``):
    exit_code: Process exit code. Defaults to zero.
"""

if "exit_code" not in dir():
    exit_code = 0

import omni.kit.app
import omni.timeline

requested_exit_code = int(exit_code)
if requested_exit_code < 0 or requested_exit_code > 255:
    raise ValueError("exit_code must be between 0 and 255")

timeline = omni.timeline.get_timeline_interface()
if timeline.is_playing():
    timeline.stop()

app = omni.kit.app.get_app()
if app is None:
    raise RuntimeError("Isaac Sim application is unavailable")

print(f"Shutdown requested with exit code {requested_exit_code}")
app.post_quit(requested_exit_code)
