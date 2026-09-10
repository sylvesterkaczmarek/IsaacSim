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

from __future__ import annotations

from collections.abc import Callable

# ()
CreateStageFn = Callable[[], bool]
# (usd_path)
OpenStageFn = Callable[[str], bool]
# (usd_path)
SaveStageFn = Callable[[str], bool]
# ()
CloseStageFn = Callable[[], bool]
# (usd_path, path, type_name)
AddReferenceToStageFn = Callable[[str, str, str], bool]

# (path, type_name)
DefinePrimFn = Callable[[str, str], bool]
# (target_path, destination_path)
MovePrimFn = Callable[[str, str], bool]
# (path)
RemovePrimFn = Callable[[str], bool]

# (path, attribute_name, type_name)
CreatePrimAttributeFn = Callable[[str, str, str], bool]
# (path, attribute_name)
RemovePrimAttributeFn = Callable[[str, str], bool]
