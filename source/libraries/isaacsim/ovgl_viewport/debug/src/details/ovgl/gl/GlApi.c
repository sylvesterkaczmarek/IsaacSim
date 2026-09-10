// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "GlApi.h"

#if defined(_WIN32)

#    include <SDL3/SDL.h>

#    include <stdio.h>

#    define OVGL_DEFINE_GL_FUNCTION(name, type) type ovgl_gl##name = NULL;
OVGL_GL_FUNCTIONS(OVGL_DEFINE_GL_FUNCTION)
#    undef OVGL_DEFINE_GL_FUNCTION

int ovgl_gl_load(void)
{
    static SDL_InitState initState;
    int loaded = 1;

    if (!SDL_ShouldInit(&initState))
    {
        return 1;
    }

#    define OVGL_LOAD_GL_FUNCTION(name, type)                                                                          \
        ovgl_gl##name = (type)SDL_GL_GetProcAddress("gl" #name);                                                       \
        if (!ovgl_gl##name)                                                                                            \
        {                                                                                                              \
            fprintf(stderr, "[ovgl] SDL could not load gl%s: %s\n", #name, SDL_GetError());                            \
            loaded = 0;                                                                                                \
        }
    OVGL_GL_FUNCTIONS(OVGL_LOAD_GL_FUNCTION)
#    undef OVGL_LOAD_GL_FUNCTION
    SDL_SetInitialized(&initState, loaded != 0);
    return loaded;
}

#endif
