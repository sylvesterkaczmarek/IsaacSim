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

#include "isaacsim/ovgl_viewport/debug/Viewport.hpp"

#include "Renderer.hpp"

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace isaacsim
{
namespace ovgl_viewport
{
namespace debug
{
namespace
{

constexpr double g_kPi = 3.14159265358979323846;
constexpr double g_kMaximumPitch = 0.49 * g_kPi;
constexpr double g_kMovementUpdatesPerSecond = 25.0;
constexpr double g_kMaximumMovementDeltaSeconds = 0.1;
constexpr double g_kMaximumMouseDeltaPerPoll = 25.0;
constexpr double g_kFpsSampleSeconds = 0.5;

class FrameRateCounter final
{
public:
    void recordFrame(std::chrono::steady_clock::time_point now)
    {
        if (m_hasPreviousFrame)
        {
            m_sampleSeconds += std::chrono::duration<double>(now - m_previousFrame).count();
            ++m_sampleIntervals;
            if (m_sampleSeconds >= g_kFpsSampleSeconds)
            {
                m_framesPerSecond = static_cast<double>(m_sampleIntervals) / m_sampleSeconds;
                m_sampleSeconds = 0.0;
                m_sampleIntervals = 0;
            }
        }
        m_previousFrame = now;
        m_hasPreviousFrame = true;
    }

    double getFramesPerSecond() const
    {
        return m_framesPerSecond;
    }

private:
    std::chrono::steady_clock::time_point m_previousFrame;
    double m_sampleSeconds{ 0.0 };
    double m_framesPerSecond{ 0.0 };
    uint64_t m_sampleIntervals{ 0 };
    bool m_hasPreviousFrame{ false };
};

std::array<double, 3> normalize(const std::array<double, 3>& value)
{
    const double length = std::hypot(value[0], value[1], value[2]);
    if (!std::isfinite(length) || length <= std::numeric_limits<double>::epsilon())
    {
        throw std::runtime_error("Cannot normalize a zero-length camera vector");
    }
    return { value[0] / length, value[1] / length, value[2] / length };
}

std::array<double, 3> cross(const std::array<double, 3>& left, const std::array<double, 3>& right)
{
    return { left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2],
             left[0] * right[1] - left[1] * right[0] };
}

std::array<double, 3> getCameraEye(const Camera& camera)
{
    const double horizontal = std::cos(camera.pitchRadians);
    const std::array<double, 3> eye{
        camera.target[0] + camera.distance * horizontal * std::cos(camera.yawRadians),
        camera.target[1] + camera.distance * horizontal * std::sin(camera.yawRadians),
        camera.target[2] + camera.distance * std::sin(camera.pitchRadians),
    };
    if (!std::all_of(eye.begin(), eye.end(), [](double value) { return std::isfinite(value); }))
    {
        throw std::runtime_error("Viewport camera eye is outside the finite coordinate range");
    }
    return eye;
}

CameraPose getCameraPose(const Camera& camera)
{
    const std::array<double, 3> eye = getCameraEye(camera);
    const std::array<double, 3> forward =
        normalize({ camera.target[0] - eye[0], camera.target[1] - eye[1], camera.target[2] - eye[2] });
    const std::array<double, 3> right = normalize(cross(forward, { 0.0, 0.0, 1.0 }));
    const std::array<double, 3> up = normalize(cross(right, forward));

    const std::array<double, 3> backward{ -forward[0], -forward[1], -forward[2] };
    const double matrix[3][3] = { { right[0], right[1], right[2] },
                                  { up[0], up[1], up[2] },
                                  { backward[0], backward[1], backward[2] } };
    std::array<double, 4> orientation{};
    const double trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
    if (trace > 0.0)
    {
        const double scale = 2.0 * std::sqrt(trace + 1.0);
        orientation = { 0.25 * scale, (matrix[1][2] - matrix[2][1]) / scale, (matrix[2][0] - matrix[0][2]) / scale,
                        (matrix[0][1] - matrix[1][0]) / scale };
    }
    else
    {
        size_t diagonal = 0;
        if (matrix[1][1] > matrix[diagonal][diagonal])
        {
            diagonal = 1;
        }
        if (matrix[2][2] > matrix[diagonal][diagonal])
        {
            diagonal = 2;
        }
        const size_t next = (diagonal + 1) % 3;
        const size_t last = (diagonal + 2) % 3;
        const double scale = 2.0 * std::sqrt(1.0 + matrix[diagonal][diagonal] - matrix[next][next] - matrix[last][last]);
        orientation[diagonal + 1] = 0.25 * scale;
        orientation[0] = (matrix[next][last] - matrix[last][next]) / scale;
        orientation[next + 1] = (matrix[diagonal][next] + matrix[next][diagonal]) / scale;
        orientation[last + 1] = (matrix[diagonal][last] + matrix[last][diagonal]) / scale;
    }
    return { eye, orientation };
}

void translateCamera(Camera& camera,
                     double forwardAmount,
                     double rightAmount,
                     double verticalAmount,
                     bool accelerated,
                     double elapsedSeconds)
{
    const std::array<double, 3> forward{ -std::cos(camera.yawRadians), -std::sin(camera.yawRadians), 0.0 };
    const std::array<double, 3> right{ -std::sin(camera.yawRadians), std::cos(camera.yawRadians), 0.0 };
    double step = std::clamp(camera.distance * 0.04, 0.1, 2.0);
    if (accelerated)
    {
        step *= 4.0;
    }
    step *= g_kMovementUpdatesPerSecond * elapsedSeconds;
    for (size_t axis = 0; axis < camera.target.size(); ++axis)
    {
        camera.target[axis] += step * (forwardAmount * forward[axis] + rightAmount * right[axis]);
    }
    camera.target[2] += step * verticalAmount;
}

void rotateCamera(Camera& camera, double deltaX, double deltaY, uint32_t viewportWidth, uint32_t viewportHeight)
{
    const std::array<double, 3> eye = getCameraEye(camera);
    const double yawDelta = deltaX * g_kPi / static_cast<double>(std::max(viewportWidth, 1U));
    const double pitchDelta = deltaY * (0.5 * g_kPi) / static_cast<double>(std::max(viewportHeight, 1U));
    camera.yawRadians = std::remainder(camera.yawRadians - yawDelta, 2.0 * g_kPi);
    camera.pitchRadians = std::clamp(camera.pitchRadians + pitchDelta, -g_kMaximumPitch, g_kMaximumPitch);

    const double horizontal = std::cos(camera.pitchRadians);
    const std::array<double, 3> eyeOffset{
        camera.distance * horizontal * std::cos(camera.yawRadians),
        camera.distance * horizontal * std::sin(camera.yawRadians),
        camera.distance * std::sin(camera.pitchRadians),
    };
    for (size_t axis = 0; axis < camera.target.size(); ++axis)
    {
        camera.target[axis] = eye[axis] - eyeOffset[axis];
    }
}

std::runtime_error makeSdlError(std::string_view operation)
{
    const char* detail = SDL_GetError();
    return std::runtime_error(detail && detail[0] != '\0' ? std::string(operation) + ": " + detail :
                                                            std::string(operation));
}

[[noreturn]] void throwSdlError(std::string_view operation)
{
    throw makeSdlError(operation);
}

class SdlWindow final
{
public:
    SdlWindow(uint32_t width, uint32_t height, const std::string& title)
    {
        if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
        {
            throwSdlError("Unable to initialize SDL video");
        }
        m_videoInitialized = true;
        m_window =
            SDL_CreateWindow(title.c_str(), static_cast<int>(width), static_cast<int>(height), SDL_WINDOW_RESIZABLE);
        if (!m_window)
        {
            const std::runtime_error error = makeSdlError("Unable to create SDL viewport window");
            reset();
            throw error;
        }
        m_windowId = SDL_GetWindowID(m_window);
        if (m_windowId == 0)
        {
            const std::runtime_error error = makeSdlError("Unable to query the SDL viewport window identifier");
            reset();
            throw error;
        }
        // Keep presentation on a different graphics API from OVGL's OpenGL ES
        // context. SDL's GPU renderer selects Vulkan on Linux and Direct3D on
        // Windows; software remains a portable fallback when neither is present.
        m_renderer = SDL_CreateRenderer(m_window, "gpu,software");
        if (!m_renderer)
        {
            const std::runtime_error error = makeSdlError("Unable to create SDL viewport presenter");
            reset();
            throw error;
        }
    }

    ~SdlWindow()
    {
        reset();
    }

    SdlWindow(const SdlWindow&) = delete;
    SdlWindow& operator=(const SdlWindow&) = delete;

    SDL_WindowID getId() const
    {
        return m_windowId;
    }

    std::array<uint32_t, 2> getPixelSize() const
    {
        int width = 0;
        int height = 0;
        if (!SDL_GetWindowSizeInPixels(m_window, &width, &height))
        {
            throwSdlError("Unable to query the SDL viewport pixel size");
        }
        return { static_cast<uint32_t>(std::max(width, 1)), static_cast<uint32_t>(std::max(height, 1)) };
    }

    std::array<uint32_t, 2> getSize() const
    {
        int width = 0;
        int height = 0;
        if (!SDL_GetWindowSize(m_window, &width, &height))
        {
            throwSdlError("Unable to query the SDL viewport size");
        }
        return { static_cast<uint32_t>(std::max(width, 1)), static_cast<uint32_t>(std::max(height, 1)) };
    }

    void setMouseCapture(bool enabled)
    {
        // Capture keeps an active drag working outside the window. Motion inside
        // the window remains usable when a platform cannot provide capture.
        (void)SDL_CaptureMouse(enabled);
    }

    void present(const Frame& frame, uint32_t windowWidth, uint32_t windowHeight, std::string_view hudText)
    {
        if (frame.width == 0 || frame.height == 0 || windowWidth == 0 || windowHeight == 0 ||
            frame.width > static_cast<uint32_t>(std::numeric_limits<int>::max() / 4) ||
            frame.width > std::numeric_limits<size_t>::max() / 4 ||
            frame.height > std::numeric_limits<size_t>::max() / (static_cast<size_t>(frame.width) * 4) ||
            frame.rgba.size() < static_cast<size_t>(frame.width) * frame.height * 4)
        {
            throw std::runtime_error("OVGL color frame has an invalid byte extent");
        }
        ensureTexture(frame.width, frame.height);
        if (!SDL_UpdateTexture(m_texture, nullptr, frame.rgba.data(), static_cast<int>(frame.width * 4)))
        {
            throwSdlError("Unable to upload the OVGL frame to SDL");
        }

        const double scale =
            std::min(static_cast<double>(windowWidth) / frame.width, static_cast<double>(windowHeight) / frame.height);
        const float presentedWidth =
            std::max(1.0F, std::min(static_cast<float>(windowWidth), static_cast<float>(frame.width * scale)));
        const float presentedHeight =
            std::max(1.0F, std::min(static_cast<float>(windowHeight), static_cast<float>(frame.height * scale)));
        const SDL_FRect destination{ (static_cast<float>(windowWidth) - presentedWidth) * 0.5F,
                                     (static_cast<float>(windowHeight) - presentedHeight) * 0.5F, presentedWidth,
                                     presentedHeight };

        if (!SDL_SetRenderDrawColor(m_renderer, 0, 0, 0, SDL_ALPHA_OPAQUE) || !SDL_RenderClear(m_renderer) ||
            !SDL_RenderTexture(m_renderer, m_texture, nullptr, &destination))
        {
            throwSdlError("Unable to present the OVGL frame with SDL");
        }
        if (!hudText.empty())
        {
            drawHud(hudText);
        }
        if (!SDL_RenderPresent(m_renderer))
        {
            throwSdlError("Unable to swap the SDL viewport window");
        }
    }

private:
    void drawHud(std::string_view text)
    {
        constexpr float margin = 10.0F;
        constexpr float padding = 6.0F;
        const std::string nullTerminatedText(text);
        const SDL_FRect background{ margin, margin,
                                    static_cast<float>(text.size() * SDL_DEBUG_TEXT_FONT_CHARACTER_SIZE) + 2 * padding,
                                    SDL_DEBUG_TEXT_FONT_CHARACTER_SIZE + 2 * padding };
        if (!SDL_SetRenderDrawColor(m_renderer, 0, 0, 0, SDL_ALPHA_OPAQUE) ||
            !SDL_RenderFillRect(m_renderer, &background) ||
            !SDL_SetRenderDrawColor(m_renderer, 255, 255, 255, SDL_ALPHA_OPAQUE) ||
            !SDL_RenderDebugText(m_renderer, margin + padding, margin + padding, nullTerminatedText.c_str()))
        {
            throwSdlError("Unable to draw the SDL viewport HUD");
        }
    }

    void ensureTexture(uint32_t width, uint32_t height)
    {
        if (m_texture && m_textureWidth == width && m_textureHeight == height)
        {
            return;
        }
        SDL_DestroyTexture(m_texture);
        m_texture = SDL_CreateTexture(m_renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING,
                                      static_cast<int>(width), static_cast<int>(height));
        if (!m_texture)
        {
            throwSdlError("Unable to create the SDL viewport texture");
        }
        if (!SDL_SetTextureScaleMode(m_texture, SDL_SCALEMODE_LINEAR))
        {
            SDL_DestroyTexture(m_texture);
            m_texture = nullptr;
            throwSdlError("Unable to configure SDL viewport scaling");
        }
        m_textureWidth = width;
        m_textureHeight = height;
    }

    void reset()
    {
        SDL_DestroyTexture(m_texture);
        m_texture = nullptr;
        SDL_DestroyRenderer(m_renderer);
        m_renderer = nullptr;
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
        m_windowId = 0;
        if (m_videoInitialized)
        {
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
            m_videoInitialized = false;
        }
    }

    SDL_Window* m_window{ nullptr };
    SDL_WindowID m_windowId{ 0 };
    SDL_Renderer* m_renderer{ nullptr };
    SDL_Texture* m_texture{ nullptr };
    uint32_t m_textureWidth{ 0 };
    uint32_t m_textureHeight{ 0 };
    bool m_videoInitialized{ false };
};

} // namespace

class Viewport::Impl
{
public:
    Impl(ovstage_instance_t* stage, CameraPoseWriter cameraPoseWriter, ViewportConfig config)
        : m_cameraPoseWriter(std::move(cameraPoseWriter)),
          m_config(std::move(config)),
          m_camera(m_config.camera),
          m_resetCamera(m_config.camera),
          m_width(m_config.width),
          m_height(m_config.height)
    {
        if (!m_cameraPoseWriter)
        {
            throw std::invalid_argument("Viewport camera-pose writer must not be empty");
        }
        if (m_config.width == 0 || m_config.height == 0 ||
            m_config.width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
            m_config.height > static_cast<uint32_t>(std::numeric_limits<int>::max()))
        {
            throw std::invalid_argument("Viewport dimensions must be positive SDL-compatible pixel extents");
        }
        if (m_config.renderProductPath.empty() || m_config.renderProductPath.front() != '/')
        {
            throw std::invalid_argument("Viewport RenderProduct path must be absolute");
        }
        const bool finiteTarget = std::all_of(
            m_camera.target.begin(), m_camera.target.end(), [](double value) { return std::isfinite(value); });
        if (!finiteTarget || !std::isfinite(m_camera.yawRadians) || !std::isfinite(m_camera.pitchRadians) ||
            !std::isfinite(m_camera.distance) || m_camera.distance <= 0.0 ||
            std::abs(m_camera.pitchRadians) > g_kMaximumPitch)
        {
            throw std::invalid_argument("Viewport camera pose must be finite, non-singular, and have positive distance");
        }
        m_renderer = std::make_unique<details::Renderer>(stage);
        if (m_config.visible)
        {
            m_window = std::make_unique<SdlWindow>(m_width, m_height, m_config.title);
            const auto pixelSize = m_window->getPixelSize();
            m_width = pixelSize[0];
            m_height = pixelSize[1];
        }
    }

    bool pollEvents()
    {
        if (!m_running || (m_config.maximumFrames != 0 && m_frame.frameNumber >= m_config.maximumFrames))
        {
            m_running = false;
            return false;
        }
        const auto pollTime = std::chrono::steady_clock::now();
        const double elapsedSeconds = std::chrono::duration<double>(pollTime - m_previousPollTime).count();
        m_previousPollTime = pollTime;
        const bool movementWasActive = hasMovementInput();
        if (m_window)
        {
            SDL_Event event{};
            while (SDL_PollEvent(&event))
            {
                handleEvent(event);
            }
        }
        applyMouseLook();
        if (m_running && hasMovementInput())
        {
            applyMovement(movementWasActive ? std::clamp(elapsedSeconds, 0.0, g_kMaximumMovementDeltaSeconds) : 0.0);
        }
        if (m_running && m_cameraDirty)
        {
            m_cameraPoseWriter(getCameraPose(m_camera));
            m_cameraDirty = false;
        }
        return m_running;
    }

    const Frame& render()
    {
        if (!m_running)
        {
            throw std::runtime_error("Cannot render a closed OVGL viewport");
        }
        m_renderer->render(m_config.renderProductPath, m_frame);
        if (m_window)
        {
            m_frameRateCounter.recordFrame(std::chrono::steady_clock::now());
            presentFrame();
        }
        return m_frame;
    }

private:
    void handleEvent(const SDL_Event& event)
    {
        switch (event.type)
        {
        case SDL_EVENT_QUIT:
            m_running = false;
            break;
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            if (event.window.windowID == m_window->getId())
            {
                m_running = false;
            }
            break;
        case SDL_EVENT_KEY_DOWN:
            if (event.key.windowID == m_window->getId() && !event.key.repeat)
            {
                handleKey(event.key.scancode, true);
            }
            break;
        case SDL_EVENT_KEY_UP:
            if (event.key.windowID == m_window->getId())
            {
                handleKey(event.key.scancode, false);
            }
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (event.button.windowID == m_window->getId() && event.button.button == SDL_BUTTON_LEFT)
            {
                m_pendingMouseDelta = {};
                m_previousMousePosition = { event.button.x, event.button.y };
                m_dragging = true;
                m_window->setMouseCapture(true);
            }
            break;
        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (event.button.windowID == m_window->getId() && event.button.button == SDL_BUTTON_LEFT)
            {
                m_dragging = false;
                m_window->setMouseCapture(false);
            }
            break;
        case SDL_EVENT_MOUSE_WHEEL:
            if (event.wheel.windowID == m_window->getId())
            {
                const double wheelY = event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -event.wheel.y : event.wheel.y;
                const double scale = std::pow(0.88, wheelY);
                m_camera.distance = std::clamp(m_camera.distance * scale, 0.25, 10000.0);
                m_cameraDirty = true;
            }
            break;
        case SDL_EVENT_MOUSE_MOTION:
            if (event.motion.windowID == m_window->getId() && m_dragging)
            {
                const double deltaX = event.motion.x - m_previousMousePosition[0];
                const double deltaY = event.motion.y - m_previousMousePosition[1];
                m_previousMousePosition = { event.motion.x, event.motion.y };
                if (std::isfinite(deltaX) && std::isfinite(deltaY))
                {
                    m_pendingMouseDelta[0] += deltaX;
                    m_pendingMouseDelta[1] += deltaY;
                }
            }
            break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            if (event.window.windowID == m_window->getId())
            {
                m_width = static_cast<uint32_t>(std::max(event.window.data1, 1));
                m_height = static_cast<uint32_t>(std::max(event.window.data2, 1));
                if (m_frame.frameNumber != 0)
                {
                    presentFrame();
                }
            }
            break;
        case SDL_EVENT_WINDOW_EXPOSED:
            if (event.window.windowID == m_window->getId() && m_frame.frameNumber != 0)
            {
                presentFrame();
            }
            break;
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            if (event.window.windowID == m_window->getId())
            {
                clearInput();
            }
            break;
        default:
            break;
        }
    }

    void handleKey(SDL_Scancode key, bool pressed)
    {
        if (key == SDL_SCANCODE_ESCAPE && pressed)
        {
            m_running = false;
            return;
        }
        if (key == SDL_SCANCODE_R)
        {
            if (pressed && !m_resetPressed)
            {
                m_camera = m_resetCamera;
                m_cameraDirty = true;
            }
            m_resetPressed = pressed;
            return;
        }
        if (key == SDL_SCANCODE_H)
        {
            if (pressed && !m_hudPressed)
            {
                m_hudVisible = !m_hudVisible;
            }
            m_hudPressed = pressed;
            return;
        }
        if (key == SDL_SCANCODE_LSHIFT)
        {
            m_leftShiftPressed = pressed;
            return;
        }
        if (key == SDL_SCANCODE_RSHIFT)
        {
            m_rightShiftPressed = pressed;
            return;
        }
        if (key == SDL_SCANCODE_W)
        {
            m_movementKeys.forward = pressed;
        }
        else if (key == SDL_SCANCODE_S)
        {
            m_movementKeys.backward = pressed;
        }
        else if (key == SDL_SCANCODE_A)
        {
            m_movementKeys.left = pressed;
        }
        else if (key == SDL_SCANCODE_D)
        {
            m_movementKeys.right = pressed;
        }
        else if (key == SDL_SCANCODE_Q)
        {
            m_movementKeys.down = pressed;
        }
        else if (key == SDL_SCANCODE_E)
        {
            m_movementKeys.up = pressed;
        }
    }

    bool hasMovementInput() const
    {
        return m_movementKeys.forward || m_movementKeys.backward || m_movementKeys.left || m_movementKeys.right ||
               m_movementKeys.down || m_movementKeys.up;
    }

    void applyMouseLook()
    {
        double deltaX = m_pendingMouseDelta[0];
        double deltaY = m_pendingMouseDelta[1];
        m_pendingMouseDelta = {};
        if (!std::isfinite(deltaX) || !std::isfinite(deltaY))
        {
            return;
        }

        // SDL can queue several drag events between slow frames. Bound their
        // combined motion once per poll so a delayed frame cannot fling the camera.
        const double magnitude = std::hypot(deltaX, deltaY);
        if (magnitude > g_kMaximumMouseDeltaPerPoll)
        {
            const double scale = g_kMaximumMouseDeltaPerPoll / magnitude;
            deltaX *= scale;
            deltaY *= scale;
        }
        if (magnitude > std::numeric_limits<double>::epsilon())
        {
            const std::array<uint32_t, 2> viewportSize = m_window->getSize();
            rotateCamera(m_camera, deltaX, deltaY, viewportSize[0], viewportSize[1]);
            m_cameraDirty = true;
        }
    }

    void presentFrame()
    {
        std::string_view hudText;
        char fpsText[32]{};
        if (m_hudVisible)
        {
            const double framesPerSecond = m_frameRateCounter.getFramesPerSecond();
            if (framesPerSecond > 0.0)
            {
                std::snprintf(fpsText, sizeof(fpsText), "FPS: %.1f", framesPerSecond);
            }
            else
            {
                std::snprintf(fpsText, sizeof(fpsText), "FPS: --");
            }
            hudText = fpsText;
        }
        m_window->present(m_frame, m_width, m_height, hudText);
    }

    void applyMovement(double elapsedSeconds)
    {
        double forwardAmount = static_cast<double>(m_movementKeys.forward) - m_movementKeys.backward;
        double rightAmount = static_cast<double>(m_movementKeys.right) - m_movementKeys.left;
        double verticalAmount = static_cast<double>(m_movementKeys.up) - m_movementKeys.down;
        const double magnitude = std::hypot(forwardAmount, rightAmount, verticalAmount);
        if (magnitude <= std::numeric_limits<double>::epsilon() || elapsedSeconds <= 0.0)
        {
            return;
        }
        forwardAmount /= magnitude;
        rightAmount /= magnitude;
        verticalAmount /= magnitude;
        translateCamera(m_camera, forwardAmount, rightAmount, verticalAmount, m_leftShiftPressed || m_rightShiftPressed,
                        elapsedSeconds);
        m_cameraDirty = true;
    }

    void clearInput()
    {
        if (m_window && m_dragging)
        {
            m_window->setMouseCapture(false);
        }
        m_movementKeys = {};
        m_leftShiftPressed = false;
        m_rightShiftPressed = false;
        m_resetPressed = false;
        m_hudPressed = false;
        m_dragging = false;
        m_pendingMouseDelta = {};
    }

    struct MovementKeys
    {
        bool forward{ false };
        bool backward{ false };
        bool left{ false };
        bool right{ false };
        bool down{ false };
        bool up{ false };
    };

    std::unique_ptr<details::Renderer> m_renderer;
    CameraPoseWriter m_cameraPoseWriter;
    ViewportConfig m_config;
    Camera m_camera;
    Camera m_resetCamera;
    std::unique_ptr<SdlWindow> m_window;
    Frame m_frame;
    uint32_t m_width;
    uint32_t m_height;
    std::chrono::steady_clock::time_point m_previousPollTime{ std::chrono::steady_clock::now() };
    FrameRateCounter m_frameRateCounter;
    MovementKeys m_movementKeys;
    std::array<double, 2> m_pendingMouseDelta{};
    std::array<double, 2> m_previousMousePosition{};
    bool m_running{ true };
    bool m_cameraDirty{ true };
    bool m_dragging{ false };
    bool m_leftShiftPressed{ false };
    bool m_rightShiftPressed{ false };
    bool m_resetPressed{ false };
    bool m_hudPressed{ false };
    bool m_hudVisible{ false };
};

Viewport::Viewport(ovstage_instance_t* stage, CameraPoseWriter cameraPoseWriter, ViewportConfig config)
    : m_impl(std::make_unique<Impl>(stage, std::move(cameraPoseWriter), std::move(config)))
{
}

Viewport::~Viewport() = default;
Viewport::Viewport(Viewport&&) noexcept = default;
Viewport& Viewport::operator=(Viewport&&) noexcept = default;

bool Viewport::pollEvents()
{
    return m_impl->pollEvents();
}

const Frame& Viewport::render()
{
    return m_impl->render();
}

} // namespace debug
} // namespace ovgl_viewport
} // namespace isaacsim
