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

extern "C" __global__ void grayscale(
    int width,
    int height,
    cudaTextureObject_t inputLdrColor,
    cudaSurfaceObject_t outputLdrGrayscale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        uchar4 pixel = tex2D<uchar4>(inputLdrColor, x, y);

        // ITU-R BT.601 luminance weights
        float luminance = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
        unsigned char gray = (unsigned char)min(255.0f, max(0.0f, luminance));

        uchar4 out = { gray, gray, gray, pixel.w };
        surf2Dwrite<uchar4>(out, outputLdrGrayscale, x * sizeof(uchar4), y);
    }
}
