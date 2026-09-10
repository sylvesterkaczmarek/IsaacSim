// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "AssetReader.hpp"

#include <utility>

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
namespace
{

struct LocalFileResult
{
    OmniClientResult result{ eOmniClientResult_Error };
    std::string path;
};

void receiveLocalFile(void* userData, OmniClientResult result, const char* localFilePath) OMNICLIENT_CALLBACK_NOEXCEPT
{
    LocalFileResult& output = *static_cast<LocalFileResult*>(userData);
    output.result = result;
    if (localFilePath)
        output.path = localFilePath;
}

}

bool isRemoteAsset(std::string_view source)
{
    constexpr std::string_view schemes[] = { "file://", "http://", "https://", "omni://", "omniverse://" };
    for (const std::string_view scheme : schemes)
    {
        if (source.compare(0, scheme.size(), scheme) == 0)
            return true;
    }
    return false;
}

LocalAssetFile::~LocalAssetFile()
{
    if (m_request != kInvalidRequestId)
        omniClientStop(m_request);
}

bool LocalAssetFile::open(const std::string& source, std::string& error)
{
    if (m_request != kInvalidRequestId)
    {
        omniClientStop(m_request);
        m_request = kInvalidRequestId;
    }
    m_path.clear();
    error.clear();
    if (!isRemoteAsset(source))
    {
        m_path = source;
        return true;
    }

    LocalFileResult result;
    m_request = omniClientGetLocalFile(source.c_str(), true, &result, receiveLocalFile);
    if (m_request == kInvalidRequestId)
    {
        error = "OmniClient rejected the asset request: " + source;
        return false;
    }
    omniClientWait(m_request);
    if (result.result != eOmniClientResult_Ok || result.path.empty())
    {
        error = "OmniClient could not cache '" + source + "': " + omniClientGetResultString(result.result);
        omniClientStop(m_request);
        m_request = kInvalidRequestId;
        return false;
    }
    m_path = std::move(result.path);
    return true;
}

const std::string& LocalAssetFile::getPath() const
{
    return m_path;
}

}
}
}
}
}
