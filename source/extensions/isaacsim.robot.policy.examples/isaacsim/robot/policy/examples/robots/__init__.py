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

"""Compatibility imports for policy classes removed in extension version 7.0.0."""

from .anymal import AnymalFlatTerrainPolicy as AnymalFlatTerrainPolicy
from .cartpole import CartpolePolicy as CartpolePolicy
from .franka import FrankaOpenDrawerPolicy as FrankaOpenDrawerPolicy
from .go2 import Go2FlatTerrainPolicy as Go2FlatTerrainPolicy
from .h1 import H1FlatTerrainPolicy as H1FlatTerrainPolicy
from .spot import SpotFlatTerrainPolicy as SpotFlatTerrainPolicy

__all__ = [
    "AnymalFlatTerrainPolicy",
    "CartpolePolicy",
    "FrankaOpenDrawerPolicy",
    "Go2FlatTerrainPolicy",
    "H1FlatTerrainPolicy",
    "SpotFlatTerrainPolicy",
]
