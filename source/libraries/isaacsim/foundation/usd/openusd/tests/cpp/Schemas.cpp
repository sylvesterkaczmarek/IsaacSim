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
#include <isaacsim/foundation/usd/openusd/Usd.hpp>
#include <pxr/base/tf/token.h>
#include <pxr/base/tf/type.h>
#include <pxr/usd/usd/schemaRegistry.h>

using namespace isaacsim::foundation::usd::openusd;

TEST_SUITE("Schemas")
{

    TEST_CASE("Newton schemas")
    {
        SUBCASE("Check if schemas are registered")
        {
            // Core scene APIs
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("NewtonSceneAPI")).IsUnknown());
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("NewtonXpbdSceneAPI")).IsUnknown());

            // Body / collision APIs
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("NewtonMassAPI")).IsUnknown());
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("NewtonCollisionAPI")).IsUnknown());
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("NewtonMaterialAPI")).IsUnknown());

            // Joint / articulation APIs
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("NewtonArticulationRootAPI"))
                    .IsUnknown());
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("NewtonJointAPI")).IsUnknown());

            // Actuator (concrete typed schema) and control APIs
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("NewtonActuator")).IsUnknown());
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("NewtonPDControlAPI")).IsUnknown());
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("NewtonPIDControlAPI")).IsUnknown());
        }

        SUBCASE("Apply API")
        {
            const int64_t stageId = createStage();
            REQUIRE(isStageValid(stageId));

            definePrim(stageId, "/World");

            std::vector<std::string> schemas;
            schemas = getAppliedSchemas(stageId, "/World");
            CHECK_EQ(schemas, std::vector<std::string>{});
            CHECK_UNARY(applyApi(stageId, "/World", "NewtonSceneAPI"));
            schemas = getAppliedSchemas(stageId, "/World");
            CHECK_EQ(schemas, std::vector<std::string>{ "NewtonSceneAPI" });

            closeStage(stageId);
        }
    }

    TEST_CASE("PhysX schemas")
    {
        SUBCASE("Check if PhysxSchema APIs are registered")
        {
            // Scene-level physics config
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("PhysxSceneAPI")).IsUnknown());

            // Rigid body simulation
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("PhysxRigidBodyAPI")).IsUnknown());

            // Articulations (robot joints)
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("PhysxArticulationAPI")).IsUnknown());

            // Collision and joint APIs
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("PhysxCollisionAPI")).IsUnknown());
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("PhysxJointAPI")).IsUnknown());
        }

        SUBCASE("Check if OmniUsdPhysicsDeformableSchema APIs are registered")
        {
            // Generic body and deformable body APIs
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("OmniPhysicsBodyAPI")).IsUnknown());
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("OmniPhysicsDeformableBodyAPI"))
                    .IsUnknown());

            // Simulation APIs
            CHECK_UNARY_FALSE(PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(
                                  PXR_NS::TfToken("OmniPhysicsVolumeDeformableSimAPI"))
                                  .IsUnknown());
            CHECK_UNARY_FALSE(PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(
                                  PXR_NS::TfToken("OmniPhysicsSurfaceDeformableSimAPI"))
                                  .IsUnknown());

            // Attachment (concrete typed schema)
            CHECK_UNARY_FALSE(
                PXR_NS::UsdSchemaRegistry::GetTypeFromSchemaTypeName(PXR_NS::TfToken("OmniPhysicsAttachment")).IsUnknown());
        }

        SUBCASE("Apply PhysxRigidBodyAPI")
        {
            const int64_t stageId = createStage();
            REQUIRE(isStageValid(stageId));

            definePrim(stageId, "/World");

            std::vector<std::string> schemas;
            schemas = getAppliedSchemas(stageId, "/World");
            CHECK_EQ(schemas, std::vector<std::string>{});
            CHECK_UNARY(applyApi(stageId, "/World", "PhysxRigidBodyAPI"));
            schemas = getAppliedSchemas(stageId, "/World");
            CHECK_EQ(schemas, std::vector<std::string>{ "PhysxRigidBodyAPI" });

            closeStage(stageId);
        }

        SUBCASE("Apply OmniPhysicsBodyAPI")
        {
            const int64_t stageId = createStage();
            REQUIRE(isStageValid(stageId));

            definePrim(stageId, "/World");

            std::vector<std::string> schemas;
            schemas = getAppliedSchemas(stageId, "/World");
            CHECK_EQ(schemas, std::vector<std::string>{});
            CHECK_UNARY(applyApi(stageId, "/World", "OmniPhysicsBodyAPI"));
            schemas = getAppliedSchemas(stageId, "/World");
            CHECK_EQ(schemas, std::vector<std::string>{ "OmniPhysicsBodyAPI" });

            closeStage(stageId);
        }
    }
}
