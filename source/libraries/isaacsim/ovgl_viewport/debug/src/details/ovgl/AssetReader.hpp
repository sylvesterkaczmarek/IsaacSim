// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <OmniClient.h>
#include <string>
#include <string_view>

namespace isaacsim
{
namespace ovgl_viewport
{
namespace debug
{
namespace details
{
namespace ovgl
{

/** Return whether source names an asset that OmniClient can localize. */
bool isRemoteAsset(std::string_view source);

/** Keeps an OmniClient cache entry pinned while its local file is being read. */
class LocalAssetFile final
{
public:
    LocalAssetFile() = default;
    ~LocalAssetFile();

    LocalAssetFile(const LocalAssetFile&) = delete;
    LocalAssetFile& operator=(const LocalAssetFile&) = delete;

    /** Resolve source to a readable local path. Plain filesystem paths pass through unchanged. */
    bool open(const std::string& source, std::string& error);

    /** Return the path resolved by the most recent successful open(). */
    const std::string& getPath() const;

private:
    OmniClientRequestId m_request{ kInvalidRequestId };
    std::string m_path;
};

}
}
}
}
}
