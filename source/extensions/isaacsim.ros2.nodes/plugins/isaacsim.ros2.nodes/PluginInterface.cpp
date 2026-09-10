// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define CARB_EXPORTS


#include <carb/Framework.h>
#include <carb/PluginUtils.h>
#include <carb/logging/Log.h>
#include <carb/settings/ISettings.h>

#include <isaacsim/ros2/nodes/IRos2Nodes.hpp>
#include <isaacsim/ros2/nodes/TfAggregationManager.h>
#include <omni/fabric/IToken.h>
#include <omni/graph/core/NodeTypeRegistrar.h>
#include <omni/graph/core/ogn/Registration.h>
#include <omni/kit/IStageUpdate.h>
#include <omni/kit/KitUpdateOrder.h>

#include <memory>

namespace
{

/**
 * @brief Plugin descriptor for the ROS2 Nodes module
 * @details
 * Defines the plugin metadata including name, description, author, and hot reload capability.
 * This descriptor is used by the Carbonite framework to identify and manage the plugin.
 */
const struct carb::PluginImplDesc g_kPluginDesc = { "isaacsim.ros2.nodes", "Isaac Sim ROS2 Nodes", "NVIDIA",
                                                    carb::PluginHotReload::eEnabled, "dev" };

std::unique_ptr<isaacsim::ros2::nodes::TfAggregationManager> g_tfAggregationManager = nullptr;
omni::kit::StageUpdatePtr g_stageUpdate = nullptr;
omni::kit::StageUpdateNode* g_tfAggregationStageUpdateNode = nullptr;

void CARB_ABI onTfAggregationStageUpdate(float currentTime,
                                         float elapsedSecs,
                                         const omni::kit::StageUpdateSettings* settings,
                                         void* userData)
{
    (void)currentTime;
    (void)elapsedSecs;
    (void)userData;

    if (settings && !settings->playComputegraph)
    {
        return;
    }

    if (g_tfAggregationManager)
    {
        g_tfAggregationManager->flushAll();
    }
}

} // anonymous namespace

/**
 * @brief Plugin implementation declaration for the ROS2 Nodes interface
 * @details
 * Registers the IRos2Nodes interface with the Carbonite plugin framework,
 * making it available to other components in the system.
 */
CARB_PLUGIN_IMPL(g_kPluginDesc, isaacsim::ros2::nodes::IRos2Nodes)

/**
 * @brief Plugin dependency declaration
 * @details
 * Declares that this plugin depends on the graph registry, token system, and settings interfaces.
 * These dependencies must be available before this plugin can be loaded.
 */
CARB_PLUGIN_IMPL_DEPS(omni::graph::core::IGraphRegistry,
                      omni::fabric::IToken,
                      carb::settings::ISettings,
                      omni::kit::IStageUpdate)

/**
 * @brief OGN node declaration
 * @details
 * Declares that this plugin will register OGN (Omniverse Graph Nodes) with the system.
 */
DECLARE_OGN_NODES()

/**
 * @brief Fills the interface with required functionality
 * @details
 * Initializes the IRos2Nodes interface with the implementation of the required functions.
 * Currently, this implementation is empty as it initializes the interface to its default state.
 *
 * @param[out] iface The interface to be filled with functionality
 */
void fillInterface(isaacsim::ros2::nodes::IRos2Nodes& iface)
{
    iface = {};
}

/**
 * @brief Plugin startup function
 * @details
 * Called by the Carbonite framework when the plugin is loaded.
 * Initializes the OGN nodes provided by this plugin.
 */
CARB_EXPORT void carbOnPluginStartup()
{
    INITIALIZE_OGN_NODES()

    auto* stageUpdateInterface = carb::getCachedInterface<omni::kit::IStageUpdate>();
    if (!stageUpdateInterface)
    {
        CARB_LOG_WARN("ROS2 TF aggregation could not acquire IStageUpdate; aggregate publishing is disabled");
        return;
    }

    g_stageUpdate = stageUpdateInterface->getStageUpdate();
    if (!g_stageUpdate)
    {
        CARB_LOG_WARN("ROS2 TF aggregation could not acquire StageUpdate; aggregate publishing is disabled");
        return;
    }

    omni::kit::StageUpdateNodeDesc desc = { nullptr };
    desc.displayName = "IsaacRos2TfAggregation";
    desc.order = omni::kit::update::eIUsdStageUpdateOmnigraph + 1;
    desc.onUpdate = onTfAggregationStageUpdate;

    g_tfAggregationStageUpdateNode = g_stageUpdate->createStageUpdateNode(desc);
    if (!g_tfAggregationStageUpdateNode)
    {
        CARB_LOG_WARN("ROS2 TF aggregation failed to create its StageUpdate node; aggregate publishing is disabled");
        g_stageUpdate = nullptr;
        return;
    }

    g_tfAggregationManager = std::make_unique<isaacsim::ros2::nodes::TfAggregationManager>();
    isaacsim::ros2::nodes::setTfAggregationManager(g_tfAggregationManager.get());
}

/**
 * @brief Plugin shutdown function
 * @details
 * Called by the Carbonite framework when the plugin is being unloaded.
 * Releases any resources associated with the OGN nodes provided by this plugin.
 */
CARB_EXPORT void carbOnPluginShutdown()
{
    if (g_stageUpdate && g_tfAggregationStageUpdateNode)
    {
        g_stageUpdate->destroyStageUpdateNode(g_tfAggregationStageUpdateNode);
    }
    g_tfAggregationStageUpdateNode = nullptr;
    g_stageUpdate = nullptr;

    isaacsim::ros2::nodes::setTfAggregationManager(nullptr);
    g_tfAggregationManager.reset();

    RELEASE_OGN_NODES()
}
