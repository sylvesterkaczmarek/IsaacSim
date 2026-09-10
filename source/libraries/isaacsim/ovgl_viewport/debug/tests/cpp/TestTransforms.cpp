// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaacsim/ovgl_viewport/debug/Viewport.hpp"

#include <doctest/doctest.h>
#include <ovstage/ovstage.h>
#include <ovstage/ovstage_population.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace
{

namespace viewport = isaacsim::ovgl_viewport::debug;

constexpr const char* g_kSceneSuffix = R"usda(
def Camera "Camera"
{
    float2 clippingRange = (0.1, 1000)
    float focalLength = 18.147562
    float horizontalAperture = 20.955
    matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 10, 1))
    uniform token[] xformOpOrder = ["xformOp:transform"]
}

def RenderVar "Color"
{
    uniform token dataType = "color4f"
    string sourceName = "LdrColor"
}

def RenderProduct "Product"
{
    rel camera = </Camera>
    rel orderedVars = [</Color>]
    int2 resolution = (128, 128)
}
)usda";

class Stage final
{
public:
    explicit Stage(const std::string& text)
    {
        ovstage_instance_desc_t description{};
        if (ovstage_create_instance(&description, &m_stage) != OVSTAGE_OK || !m_stage)
        {
            throw std::runtime_error("Unable to create OVStage test instance");
        }
        const ovx_string_t source{ text.c_str(), text.size() };
        const ovstage_population_enqueue_result_t population =
            ovstage_population_open_usd_from_string(m_stage, source, 1, 0.0, OVSTAGE_POPULATION_DOMAIN_ALL);
        if (population.status != OVSTAGE_OK ||
            ovstage_population_wait_op(m_stage, population.op_index, OVSTAGE_TIMEOUT_INFINITE, nullptr) != OVSTAGE_OK)
        {
            ovstage_destroy_instance(m_stage);
            m_stage = nullptr;
            throw std::runtime_error("Unable to populate OVStage test instance");
        }

        ovstage_write_floor_desc_t floorDescription{};
        floorDescription.ordinal = 1;
        floorDescription.scope = OVSTAGE_SCOPE_ALL;
        const ovstage_enqueue_result_t floor = ovstage_advance_write_floor(m_stage, &floorDescription);
        if (floor.status != OVSTAGE_OK ||
            ovstage_wait_op(m_stage, floor.op_index, OVSTAGE_TIMEOUT_INFINITE, nullptr) != OVSTAGE_OK)
        {
            ovstage_destroy_instance(m_stage);
            m_stage = nullptr;
            throw std::runtime_error("Unable to seal populated OVStage test instance");
        }
        (void)ovstage_release_op(m_stage, floor.op_index);
    }

    ~Stage()
    {
        if (m_stage)
        {
            ovstage_destroy_instance(m_stage);
        }
    }

    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;

    ovstage_instance_t* get() const
    {
        return m_stage;
    }

private:
    ovstage_instance_t* m_stage{ nullptr };
};

viewport::Frame renderScene(const std::string& scene)
{
    Stage stage("#usda 1.0\n" + scene + g_kSceneSuffix);
    viewport::ViewportConfig configuration;
    configuration.visible = false;
    configuration.renderProductPath = "/Product";
    viewport::Viewport renderer(
        stage.get(), [](const viewport::CameraPose&) {}, configuration);
    return renderer.render();
}

double getForegroundCentroidX(const viewport::Frame& frame)
{
    double weightedX = 0.0;
    size_t foregroundPixels = 0;
    for (uint32_t y = 0; y < frame.height; ++y)
    {
        for (uint32_t x = 0; x < frame.width; ++x)
        {
            const size_t offset = (static_cast<size_t>(y) * frame.width + x) * 4;
            const uint32_t brightness = std::to_integer<uint8_t>(frame.rgba[offset]) +
                                        std::to_integer<uint8_t>(frame.rgba[offset + 1]) +
                                        std::to_integer<uint8_t>(frame.rgba[offset + 2]);
            if (brightness > 60)
            {
                weightedX += x;
                ++foregroundPixels;
            }
        }
    }
    REQUIRE_UNARY(foregroundPixels > 0);
    return weightedX / static_cast<double>(foregroundPixels);
}

} // namespace

TEST_SUITE("isaacsim.ovgl_viewport.debug transforms")
{
    TEST_CASE("Composes an ancestor transform")
    {
        const viewport::Frame frame = renderScene(R"usda(
def Xform "World"
{
    def Xform "Offset"
    {
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Cube "Cube"
        {
            color3f[] primvars:displayColor = [(1, 1, 1)]
            double size = 1
        }
    }
}
)usda");

        CHECK_UNARY(getForegroundCentroidX(frame) > 80.0);
    }

    TEST_CASE("Composes a scene-graph instance transform")
    {
        const viewport::Frame frame = renderScene(R"usda(
def Xform "Prototype" (
    instanceable = true
)
{
    def Cube "Cube"
    {
        color3f[] primvars:displayColor = [(1, 1, 1)]
        double size = 1
    }
}

def Xform "Instance" (
    instanceable = true
    prepend references = </Prototype>
)
{
    double3 xformOp:translate = (3, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
)usda");

        CHECK_UNARY(getForegroundCentroidX(frame) > 70.0);
    }
}
