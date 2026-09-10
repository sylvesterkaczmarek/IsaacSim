-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

function invert(inputs, outputs)
    assert(#inputs["Image"].shape == 2, "Input must be a 2D image")
    assert(inputs["Image"].dtype == cuda.uchar4, "Input must be uchar4")

    local height = inputs["Image"].shape[1]
    local width  = inputs["Image"].shape[2]

    outputs["Inverted"] = cuda.image(width, height, cuda.uchar4)

    return cuda.kernel({
        -- void invert(int, int, float, cudaTextureObject_t, cudaSurfaceObject_t)
        args = {
            cuda.int(width),                           -- -> int width
            cuda.int(height),                          -- -> int height
            cuda.float(inputs["strength"]),             -- -> float strength
            cuda.TextureObject(inputs["Image"]),        -- -> cudaTextureObject_t inputImage
            cuda.SurfaceObject(outputs["Inverted"]),    -- -> cudaSurfaceObject_t outputInverted
        },
        block = { 32, 32 },
        grid = { math.ceil(width/32), math.ceil(height/32) },
    })
end
