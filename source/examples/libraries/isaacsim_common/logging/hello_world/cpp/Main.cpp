// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <isaacsim/common/logging/Logging.hpp>

int main()
{
    isaacsim::common::logging::Logger logger("isaacsim.examples.hello_world.cpp");
    logger.report("Hello World from C++.");
    return 0;
}
