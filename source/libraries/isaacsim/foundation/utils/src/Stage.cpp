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

#include <isaacsim/foundation/utils/Stage.hpp>

namespace isaacsim
{
namespace foundation
{
namespace utils
{

void setDefaultStage(objects::Stage stage)
{
    objects::details::setDefaultStage(std::move(stage));
}

objects::Stage getDefaultStage()
{
    return objects::details::getDefaultStage();
}

objects::Stage getActiveStage()
{
    return objects::details::getActiveStage();
}

StageGuard::StageGuard(objects::Stage stage)
    : m_stageGuard(std::make_unique<objects::details::StageGuard>(std::move(stage)))
{
}

} // namespace utils
} // namespace foundation
} // namespace isaacsim
