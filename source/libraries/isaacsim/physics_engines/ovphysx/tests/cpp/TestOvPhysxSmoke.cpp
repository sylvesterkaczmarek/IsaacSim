// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Minimal smoke test: initialize ovphysx, create a CPU-mode instance, destroy it.
// Passes on machines with no CUDA driver.

#include "ovphysx/ovphysx.h"

#include <doctest/doctest.h>

TEST_CASE("ovphysx smoke: initialize, create CPU instance, destroy")
{
    REQUIRE(ovphysx_initialize().status == OVPHYSX_API_SUCCESS);

    ovphysx_create_args arguments = OVPHYSX_CREATE_ARGS_DEFAULT;
    ovphysx_handle_t handle = OVPHYSX_INVALID_HANDLE;
    REQUIRE(ovphysx_create_instance(&arguments, &handle).status == OVPHYSX_API_SUCCESS);

    REQUIRE(ovphysx_destroy_instance(handle).status == OVPHYSX_API_SUCCESS);
    REQUIRE(ovphysx_shutdown().status == OVPHYSX_API_SUCCESS);
}
