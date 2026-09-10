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

extern "C" __global__ void invert(
    int width,
    int height,
    float strength,
    cudaTextureObject_t inputImage,
    cudaSurfaceObject_t outputInverted)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        uchar4 pixel = tex2D<uchar4>(inputImage, x, y);

        // lerp(original, 255-original, strength) per RGB channel
        unsigned char r = (unsigned char)(pixel.x + strength * (255 - 2 * pixel.x));
        unsigned char g = (unsigned char)(pixel.y + strength * (255 - 2 * pixel.y));
        unsigned char b = (unsigned char)(pixel.z + strength * (255 - 2 * pixel.z));

        uchar4 out = { r, g, b, pixel.w };
        surf2Dwrite<uchar4>(out, outputInverted, x * sizeof(uchar4), y);
    }
}
