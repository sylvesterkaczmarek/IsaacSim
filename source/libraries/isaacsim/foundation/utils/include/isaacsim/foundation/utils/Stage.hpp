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

#pragma once

#include <isaacsim/foundation/objects/Stage.hpp>
#include <isaacsim/foundation/utils/Export.h>

#include <memory>

namespace isaacsim
{
namespace foundation
{
namespace utils
{

/**
 * @brief Set the process-wide default stage.
 * @details The default stage is shared across all threads and is used as a fallback
 *          by @c getActiveStage() when no thread-local active stage has been set.
 * @param[in] stage The stage to set as the default.
 */
ISAACSIM_FOUNDATION_UTILS_API void setDefaultStage(objects::Stage stage);

/**
 * @brief Get the process-wide default stage.
 * @return The current default stage.
 * @throws std::runtime_error if no default stage has been set.
 */
ISAACSIM_FOUNDATION_UTILS_API objects::Stage getDefaultStage();

/**
 * @brief Get the active stage for the calling thread.
 * @details Returns the thread-local active stage if one has been set (e.g. via @c StageGuard)
 *          and the stage is still valid. Falls back to the process-wide default stage otherwise.
 * @return The active stage.
 * @throws std::runtime_error if neither a valid thread-local active stage nor a default stage exists.
 */
ISAACSIM_FOUNDATION_UTILS_API objects::Stage getActiveStage();

/**
 * @class StageGuard
 * @brief RAII guard that temporarily overrides the thread-local active stage.
 * @details On construction, installs @p stage as the thread-local active stage returned by
 *          @c getActiveStage(). On destruction, restores whatever stage was active before.
 *          Non-copyable and non-movable to prevent guard lifetimes from being transferred.
 */
class ISAACSIM_FOUNDATION_UTILS_API StageGuard
{
public:
    /**
     * @brief Install @p stage as the thread-local active stage.
     * @param[in] stage The stage to activate for the duration of this guard's lifetime.
     */
    explicit StageGuard(objects::Stage stage);

    /**
     * @brief Restore the previously active stage.
     */
    ~StageGuard() = default;

    StageGuard(const StageGuard&) = delete;
    StageGuard& operator=(const StageGuard&) = delete;
    StageGuard(StageGuard&&) = delete;
    StageGuard& operator=(StageGuard&&) = delete;

private:
    std::unique_ptr<objects::details::StageGuard> m_stageGuard;
};

} // namespace utils
} // namespace foundation
} // namespace isaacsim
