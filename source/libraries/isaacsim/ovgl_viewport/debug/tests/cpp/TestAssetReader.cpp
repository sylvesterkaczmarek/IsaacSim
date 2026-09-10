// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "AssetReader.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace
{

bool expect(bool condition, const char* message)
{
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

} // namespace

int main()
{
    namespace asset = isaacsim::ovgl_viewport::debug::details::ovgl;

    bool ok = true;
    ok &= expect(!asset::isRemoteAsset("textures/albedo.png"), "relative filesystem path is local");
    ok &= expect(asset::isRemoteAsset("https://example.com/albedo.png"), "HTTPS asset is remote");
    ok &= expect(asset::isRemoteAsset("omniverse://server/asset.png"), "Omniverse asset is remote");

    asset::LocalAssetFile localPath;
    std::string error;
    ok &= expect(localPath.open("textures/albedo.png", error), "plain filesystem path passes through");
    ok &= expect(localPath.getPath() == "textures/albedo.png", "plain filesystem path remains unchanged");

    const int uniqueMarker = 0;
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("ovgl-asset-reader-test-" + std::to_string(reinterpret_cast<std::uintptr_t>(&uniqueMarker)) + ".txt");
    struct RemoveFile
    {
        std::filesystem::path path;
        ~RemoveFile()
        {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    } cleanup{ path };

    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream << "cached asset";
    }

    asset::LocalAssetFile fileUrl;
    ok &= expect(fileUrl.open("file://" + path.string(), error), "OmniClient resolves a file URL");
    if (!fileUrl.getPath().empty())
    {
        std::ifstream stream(fileUrl.getPath(), std::ios::binary);
        std::string contents((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        ok &= expect(contents == "cached asset", "resolved file remains readable while the request is alive");
    }

    return ok ? 0 : 1;
}
