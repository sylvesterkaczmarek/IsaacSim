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

#include <doctest/doctest.h>
#include <isaacsim/foundation/utils/Backend.hpp>

#include <stdexcept>

using namespace isaacsim::foundation::utils;

TEST_SUITE("Backend")
{

    TEST_CASE("BackendGuard | isBackendSet | getCurrentBackend")
    {
        // no backend active outside a guard
        CHECK_UNARY_FALSE(isBackendSet());

        {
            BackendGuard guard("backend_1");
            CHECK_UNARY(isBackendSet());
            CHECK_EQ(getCurrentBackend({ "backend_1", "backend_2" }), "backend_1");
        }

        // no backend active outside a guard
        CHECK_UNARY_FALSE(isBackendSet());

        // restored after exception
        try
        {
            BackendGuard guard("backend_1");
            // fallback to 1st supported backend
            CHECK_EQ(getCurrentBackend({ "backend_2" }), "backend_2");
            throw std::runtime_error("forced exception");
        }
        catch (const std::runtime_error&)
        {
        }
        CHECK_UNARY_FALSE(isBackendSet());
    }

    TEST_CASE("BackendGuard: nested guards")
    {
        CHECK_UNARY_FALSE(isBackendSet());
        {
            BackendGuard outer("backend_1");
            CHECK_UNARY(isBackendSet());
            CHECK_EQ(getCurrentBackend({ "backend_1", "backend_2" }), "backend_1");
            {
                BackendGuard inner("backend_2");
                CHECK_UNARY(isBackendSet());
                CHECK_EQ(getCurrentBackend({ "backend_1", "backend_2" }), "backend_2");
            }
            CHECK_UNARY(isBackendSet());
            CHECK_EQ(getCurrentBackend({ "backend_1", "backend_2" }), "backend_1");
        }
        CHECK_UNARY_FALSE(isBackendSet());
    }

    TEST_CASE("BackendGuard(raiseOnUnsupported)")
    {
        // via the context flag
        CHECK_UNARY_FALSE(shouldRaiseOnUnsupported());
        {
            BackendGuard guard("unknown", /*raiseOnUnsupported=*/true);
            CHECK_UNARY(shouldRaiseOnUnsupported());
            CHECK_THROWS_AS(getCurrentBackend({ "backend" }), std::runtime_error);
        }
        // via the call-site parameter
        CHECK_UNARY_FALSE(shouldRaiseOnUnsupported());
        {
            BackendGuard guard("unknown");
            CHECK_UNARY_FALSE(shouldRaiseOnUnsupported());
            CHECK_THROWS_AS(getCurrentBackend({ "backend" }, true), std::runtime_error);
        }
        // explicit false at the call site overrides the context flag
        CHECK_UNARY_FALSE(shouldRaiseOnUnsupported());
        {
            BackendGuard guard("unknown", /*raiseOnUnsupported=*/true);
            CHECK_UNARY(shouldRaiseOnUnsupported());
            CHECK_EQ(getCurrentBackend({ "backend" }, false), "backend");
        }
        CHECK_UNARY_FALSE(shouldRaiseOnUnsupported());
    }

    TEST_CASE("BackendGuard(raiseOnFallback)")
    {
        CHECK_UNARY_FALSE(shouldRaiseOnFallback());
        {
            BackendGuard guard("backend", /*raiseOnUnsupported=*/false, /*raiseOnFallback=*/true);
            CHECK_UNARY(shouldRaiseOnFallback());
        }
        CHECK_UNARY_FALSE(shouldRaiseOnFallback());
    }
}
