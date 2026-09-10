// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaacsim/foundation/objects/Camera.hpp"
#include "isaacsim/foundation/objects/Stage.hpp"
#include "isaacsim/ovgl_viewport/debug/Viewport.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

namespace array = isaacsim::common::array;
namespace foundation = isaacsim::foundation::objects;
namespace viewport = isaacsim::ovgl_viewport::debug;

constexpr uint32_t g_kInitialWidth = 1280;
constexpr uint32_t g_kInitialHeight = 720;
constexpr const char* g_kCameraPath = "/IsaacSimViewportCamera";
constexpr const char* g_kRenderProductPath = "/IsaacSimViewportRenderProduct";

struct CommandLine
{
    std::string stageSource;
    uint64_t maximumFrames{ 0 };
    bool headless{ false };
};

void printUsage(const char* executable)
{
    std::cout << "Usage: " << executable << " [USD_PATH_OR_URL] [--frames COUNT] [--headless]\n\n"
              << "Displays an optional USD scene, or a procedural cube when no source is provided,\n"
              << "using OVStage and isaacsim.ovgl_viewport.debug.\n\n"
              << "Controls:\n"
              << "  Left drag       Mouse look\n"
              << "  Mouse wheel     Dolly\n"
              << "  Hold W/S        Move forward/backward\n"
              << "  Hold A/D        Strafe left/right\n"
              << "  Hold Q/E        Move down/up\n"
              << "  Shift+movement  Move faster\n"
              << "  R               Reset view\n"
              << "  H               Toggle debug HUD\n"
              << "  Escape          Quit\n\n"
              << "Options:\n"
              << "  --headless      Render without creating a user-visible window\n";
}

CommandLine parseCommandLine(int argumentCount, char** arguments)
{
    CommandLine commandLine;
    bool stageSourceSet = false;
    for (int index = 1; index < argumentCount; ++index)
    {
        const std::string argument = arguments[index];
        if (argument == "--help" || argument == "-h")
        {
            printUsage(arguments[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (argument == "--frames")
        {
            if (++index >= argumentCount)
            {
                throw std::invalid_argument("--frames requires a non-negative integer");
            }
            const std::string frameCount = arguments[index];
            size_t parsedCharacters = 0;
            if (frameCount.empty() || frameCount.front() == '-')
            {
                throw std::invalid_argument("--frames requires a non-negative integer");
            }
            try
            {
                commandLine.maximumFrames = std::stoull(frameCount, &parsedCharacters);
            }
            catch (const std::exception&)
            {
                throw std::invalid_argument("--frames requires a non-negative integer");
            }
            if (parsedCharacters != frameCount.size())
            {
                throw std::invalid_argument("--frames requires a non-negative integer");
            }
            continue;
        }
        if (argument == "--headless")
        {
            commandLine.headless = true;
            continue;
        }
        if (!argument.empty() && argument[0] == '-')
        {
            throw std::invalid_argument("Unknown option: " + argument);
        }
        if (stageSourceSet)
        {
            throw std::invalid_argument("Only one USD source may be provided");
        }
        commandLine.stageSource = argument;
        stageSourceSet = true;
    }
    return commandLine;
}

bool isRemoteStageSource(std::string_view source)
{
    constexpr std::string_view schemes[] = { "file://", "http://", "https://", "omni://", "omniverse://" };
    for (const std::string_view scheme : schemes)
    {
        if (source.compare(0, scheme.size(), scheme) == 0)
            return true;
    }
    return false;
}

std::string getStageLayerIdentifier(const std::string& stageSource)
{
    const std::string identifier = isRemoteStageSource(stageSource) ?
                                       stageSource :
                                       std::filesystem::absolute(stageSource).lexically_normal().string();
    if (identifier.find_first_of("@\r\n") != std::string::npos)
    {
        throw std::invalid_argument("USD sources containing '@' or a newline are not supported");
    }
    return identifier;
}

std::string getSourceFilename(const std::string& stageSource)
{
    const size_t suffix = stageSource.find_first_of("?#");
    return std::filesystem::path(stageSource.substr(0, suffix)).filename().string();
}

std::string buildStageText(const std::string& stageSource)
{
    std::ostringstream stage;
    stage << "#usda 1.0\n";
    stage << "(\n";
    if (stageSource.empty())
    {
        stage << "    defaultPrim = \"World\"\n    metersPerUnit = 1\n    upAxis = \"Z\"\n";
    }
    else
    {
        stage << "    subLayers = [@" << getStageLayerIdentifier(stageSource) << "@]\n";
    }
    stage << ")\n\n";
    if (stageSource.empty())
    {
        stage << R"usda(def Xform "World"
{
    def Cube "Cube"
    {
        double size = 1
        matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }

    def DistantLight "KeyLight"
    {
        color3f color = (1, 0.92, 0.78)
        float intensity = 3000
        matrix4d xformOp:transform = ( (0.9063, -0.2424, 0.3462, 0), (0, 0.8192, 0.5736, 0), (-0.4226, -0.5198, 0.7424, 0), (0, 0, 0, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}

)usda";
    }
    stage << R"usda(def Camera "IsaacSimViewportCamera"
{
    float2 clippingRange = (1, 10000000)
    float focalLength = 18.147562
    float horizontalAperture = 20.955
    matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 6, 1) )
    uniform token[] xformOpOrder = ["xformOp:transform"]
}

def RenderVar "IsaacSimViewportColor"
{
    uniform token dataType = "color4f"
    uniform string sourceName = "LdrColor"
}

def RenderProduct "IsaacSimViewportRenderProduct"
{
    rel camera = </IsaacSimViewportCamera>
    rel orderedVars = [</IsaacSimViewportColor>]
    uniform int2 resolution = (1280, 720)
}
)usda";
    return stage.str();
}

viewport::Camera getInitialCamera(const std::string& stageSource)
{
    viewport::Camera camera;
    camera.target = { -10.0, -4.0, 4.6 };
    camera.distance = 88.0;

    const std::string filename = getSourceFilename(stageSource);
    if (stageSource.empty())
    {
        camera.target = { 0.0, 0.0, 0.0 };
        camera.distance = 6.0;
    }
    else if (filename == "franka.usd")
    {
        camera.target = { 0.0, 0.0, 0.55 };
        camera.pitchRadians = 0.25;
        camera.distance = 3.0;
    }
    else if (filename == "robot-ovrtx.usda")
    {
        camera.target = { 0.0, -0.0116547517, 0.8218411361 };
        camera.yawRadians = -1.3245;
        camera.pitchRadians = 0.0489;
        camera.distance = 3.56;
    }
    return camera;
}

int runViewer(const CommandLine& commandLine)
{
    if (!std::getenv("OVGL_SS"))
    {
#ifdef _WIN32
        _putenv_s("OVGL_SS", "1");
#else
        setenv("OVGL_SS", "1", 0);
#endif
    }
    if (!commandLine.stageSource.empty() && !isRemoteStageSource(commandLine.stageSource) &&
        !std::filesystem::is_regular_file(commandLine.stageSource))
    {
        throw std::runtime_error("USD stage does not exist: " + commandLine.stageSource);
    }

    if (commandLine.stageSource.empty())
    {
        std::cout << "Populating OVStage with the procedural cube ..." << std::endl;
    }
    else
    {
        std::cout << "Populating OVStage from " << commandLine.stageSource << " ..." << std::endl;
    }
    foundation::Stage stage("ovstage");
    stage.importStageFromString(buildStageText(commandLine.stageSource));
    if (!stage.isValid())
    {
        throw std::runtime_error("OVStage failed to populate the USD file");
    }

    try
    {
        std::cout << "OVStage is ready." << std::endl;
        foundation::Camera camera(g_kCameraPath, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                                  /*resetXformOpProperties=*/false);

        viewport::ViewportConfig viewportConfig;
        viewportConfig.title = "Isaac Sim OVStage + OVGL Viewport";
        viewportConfig.width = g_kInitialWidth;
        viewportConfig.height = g_kInitialHeight;
        viewportConfig.maximumFrames = commandLine.maximumFrames;
        viewportConfig.visible = !commandLine.headless;
        viewportConfig.camera = getInitialCamera(commandLine.stageSource);
        viewportConfig.renderProductPath = g_kRenderProductPath;
        void* nativeStage = stage.getStagePtr();
        if (!nativeStage)
        {
            throw std::runtime_error("Foundation did not retain a native stage for the loaded scene");
        }
        const viewport::CameraPoseWriter cameraPoseWriter = [&camera](const viewport::CameraPose& pose)
        {
            array::Array positions(
                std::vector<std::vector<double>>{ { pose.position[0], pose.position[1], pose.position[2] } });
            array::Array orientations(std::vector<std::vector<double>>{
                { pose.orientation[0], pose.orientation[1], pose.orientation[2], pose.orientation[3] } });
            camera.setWorldPoses(positions, orientations);
        };
        viewport::Viewport viewer(static_cast<ovstage_instance_t*>(nativeStage), cameraPoseWriter, viewportConfig);
        uint64_t renderedFrames = 0;
        while (viewer.pollEvents())
        {
            renderedFrames = viewer.render().frameNumber;
        }
        std::cout << "Viewport rendered " << renderedFrames << " frame(s)." << std::endl;
    }
    catch (...)
    {
        stage.closeStage();
        throw;
    }
    if (!stage.closeStage())
    {
        throw std::runtime_error("Unable to close OVStage");
    }
    return EXIT_SUCCESS;
}

} // namespace

int main(int argumentCount, char** arguments)
{
    try
    {
        return runViewer(parseCommandLine(argumentCount, arguments));
    }
    catch (const std::exception& exception)
    {
        std::cerr << "OVGL viewer error: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
}
