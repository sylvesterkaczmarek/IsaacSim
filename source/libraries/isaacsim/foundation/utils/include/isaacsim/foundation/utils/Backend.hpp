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

#include <isaacsim/foundation/utils/Export.h>

#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

namespace isaacsim
{
namespace foundation
{
namespace utils
{

namespace details
{

struct BackendContext
{
    std::optional<std::string> backend;
    bool raiseOnUnsupported = false;
    bool raiseOnFallback = false;
};

extern thread_local BackendContext g_context;

} // namespace details

/**
 * @class BackendGuard
 * @brief RAII guard that sets the active backend for the current thread within a scope.
 * @details
 * On construction, saves the current thread-local backend context and activates the
 * requested backend. On destruction, the previous context is automatically restored.
 * Nested guards are supported: each guard restores exactly the state it found on entry.
 *
 * @note This class is not copyable or movable. Declare it as a named local variable
 *       to ensure the scope matches the intended lifetime.
 *
 * @warning Declaring a temporary (unnamed) BackendGuard restores the backend immediately.
 *          Always declare a named variable: `BackendGuard guard("tensor");`
 *
 * @see getCurrentBackend(), isBackendSet()
 */
class ISAACSIM_FOUNDATION_UTILS_API BackendGuard
{
public:
    /**
     * @brief Activates the given backend for the current thread scope.
     * @param[in] backend The backend name to activate (e.g. @c "usd", @c "tensor").
     * @param[in] raiseOnUnsupported Whether to raise an exception when the active backend
     *            is not in the supported list passed to getCurrentBackend().
     * @param[in] raiseOnFallback Whether to raise an exception when a fallback backend
     *            is selected because the active backend is unsupported.
     */
    explicit BackendGuard(const std::string& backend, bool raiseOnUnsupported = false, bool raiseOnFallback = false);

    /**
     * @brief Restores the backend context that was active before this guard was created.
     */
    ~BackendGuard();

    BackendGuard(const BackendGuard&) = delete;
    BackendGuard& operator=(const BackendGuard&) = delete;
    BackendGuard(BackendGuard&&) = delete;
    BackendGuard& operator=(BackendGuard&&) = delete;

private:
    /**
     * @brief Snapshot of the context captured at guard construction.
     */
    details::BackendContext m_previous;
};

/**
 * @brief Returns the currently active backend name for this thread.
 * @details
 * If a BackendGuard is active on this thread, its backend name is returned. Otherwise,
 * the first entry of @p supportedBackends is used as the default.
 *
 * If the active backend is not in @p supportedBackends, the behavior depends on
 * @p raiseOnUnsupported (and the corresponding flag stored in the context):
 * - If either is @c true, a @c std::runtime_error is thrown.
 * - Otherwise, the first entry of @p supportedBackends is returned as a fallback.
 *
 * @param[in] supportedBackends Ordered list of backend names accepted by the caller.
 *            The first entry is used as the default and fallback.
 * @param[in] raiseOnUnsupported Override for the raise-on-unsupported flag. When not
 *            @c nullopt, takes precedence over the value stored in the context.
 *
 * @return The active backend name, or the first entry of @p supportedBackends as fallback.
 *
 * @throws std::invalid_argument If @p supportedBackends is empty.
 * @throws std::runtime_error If the active backend is unsupported and raising is enabled.
 *
 * @pre @p supportedBackends must not be empty.
 */
ISAACSIM_FOUNDATION_UTILS_API std::string getCurrentBackend(const std::initializer_list<std::string>& supportedBackends,
                                                            std::optional<bool> raiseOnUnsupported = std::nullopt);

/**
 * @brief Returns the currently active backend name for this thread.
 * @details Vector overload of getCurrentBackend for callers with a runtime-determined backend list.
 *          Behavior is identical to the initializer_list overload.
 *
 * @param[in] supportedBackends Ordered list of backend names accepted by the caller.
 * @param[in] raiseOnUnsupported Override for the raise-on-unsupported flag.
 * @return The active backend name, or the first entry of @p supportedBackends as fallback.
 *
 * @throws std::invalid_argument If @p supportedBackends is empty.
 * @throws std::runtime_error If the active backend is unsupported and raising is enabled.
 */
ISAACSIM_FOUNDATION_UTILS_API std::string getCurrentBackend(const std::vector<std::string>& supportedBackends,
                                                            std::optional<bool> raiseOnUnsupported = std::nullopt);

/**
 * @brief Checks whether a backend is currently set on this thread.
 * @return @c true if a BackendGuard is active on this thread, @c false otherwise.
 */
ISAACSIM_FOUNDATION_UTILS_API bool isBackendSet();

/**
 * @brief Checks whether the current context requests an exception on an unsupported backend.
 * @return @c true if raise-on-unsupported is enabled in the active context.
 */
ISAACSIM_FOUNDATION_UTILS_API bool shouldRaiseOnUnsupported();

/**
 * @brief Checks whether the current context requests an exception on a fallback backend.
 * @return @c true if raise-on-fallback is enabled in the active context.
 */
ISAACSIM_FOUNDATION_UTILS_API bool shouldRaiseOnFallback();

} // namespace utils
} // namespace foundation
} // namespace isaacsim
