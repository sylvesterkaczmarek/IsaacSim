-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

function grayscale(inputs, outputs)
    assert(#inputs["LdrColor"].shape == 2, "Input must be a 2D image")
    assert(inputs["LdrColor"].dtype == cuda.uchar4, "Input must be uchar4")

    local height = inputs["LdrColor"].shape[1]
    local width  = inputs["LdrColor"].shape[2]

    outputs["LdrGrayscale"] = cuda.image(width, height, cuda.uchar4)

    return cuda.kernel({
        args = {
            cuda.int(width),                             -- -> int width
            cuda.int(height),                            -- -> int height
            cuda.TextureObject(inputs["LdrColor"]),      -- -> cudaTextureObject_t inputLdrColor
            cuda.SurfaceObject(outputs["LdrGrayscale"]), -- -> cudaSurfaceObject_t outputLdrGrayscale
        },
        block = { 32, 32 },
        grid = { math.ceil(width/32), math.ceil(height/32) },
    })
end
