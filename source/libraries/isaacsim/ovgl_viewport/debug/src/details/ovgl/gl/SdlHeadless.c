// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "EglHeadless.h"

#include <SDL3/SDL.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct EglHeadless
{
    SDL_Window* window;
    SDL_GLContext context;
    int width;
    int height;
};

EglHeadless* egl_headless_create(int width, int height)
{
    EglHeadless* ctx = NULL;
    if (width <= 0 || height <= 0)
        return NULL;
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
    {
        fprintf(stderr, "[ovgl] SDL video initialization failed: %s\n", SDL_GetError());
        return NULL;
    }
    if (!SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4) || !SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1) ||
        !SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE) ||
        !SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 0) || !SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24))
    {
        fprintf(stderr, "[ovgl] SDL OpenGL attribute setup failed: %s\n", SDL_GetError());
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
        return NULL;
    }

    ctx = (EglHeadless*)calloc(1, sizeof(EglHeadless));
    if (!ctx)
    {
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
        return NULL;
    }
    ctx->window = SDL_CreateWindow("OVGL offscreen context", width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    if (!ctx->window)
    {
        fprintf(stderr, "[ovgl] SDL OpenGL window creation failed: %s\n", SDL_GetError());
        egl_headless_destroy(ctx);
        return NULL;
    }
    ctx->context = SDL_GL_CreateContext(ctx->window);
    if (!ctx->context)
    {
        fprintf(stderr, "[ovgl] SDL OpenGL context creation failed: %s\n", SDL_GetError());
        egl_headless_destroy(ctx);
        return NULL;
    }
    if (!SDL_GL_MakeCurrent(ctx->window, ctx->context))
    {
        fprintf(stderr, "[ovgl] SDL could not make the OpenGL context current: %s\n", SDL_GetError());
        egl_headless_destroy(ctx);
        return NULL;
    }
    ctx->width = width;
    ctx->height = height;
    return ctx;
}

void egl_headless_destroy(EglHeadless* ctx)
{
    if (!ctx)
        return;
    if (ctx->context && SDL_GL_GetCurrentContext() == ctx->context)
        (void)SDL_GL_MakeCurrent(ctx->window, NULL);
    SDL_GL_DestroyContext(ctx->context);
    SDL_DestroyWindow(ctx->window);
    free(ctx);
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
}

int egl_headless_resize(EglHeadless* ctx, int width, int height)
{
    if (!ctx || width <= 0 || height <= 0)
        return 0;
    ctx->width = width;
    ctx->height = height;
    return egl_headless_make_current(ctx);
}

int egl_headless_is_software(const EglHeadless* ctx)
{
    (void)ctx;
    return 0;
}

int egl_headless_make_current(EglHeadless* ctx)
{
    return ctx && SDL_GL_MakeCurrent(ctx->window, ctx->context) ? 1 : 0;
}

void egl_headless_release_current(EglHeadless* ctx)
{
    if (ctx && SDL_GL_GetCurrentContext() == ctx->context)
        (void)SDL_GL_MakeCurrent(ctx->window, NULL);
}

int egl_headless_has_current(void)
{
    return SDL_GL_GetCurrentContext() != NULL ? 1 : 0;
}

uintptr_t egl_headless_current_context_id(void)
{
    return (uintptr_t)SDL_GL_GetCurrentContext();
}

int egl_headless_suspend_foreign(void)
{
    return 0;
}

void egl_headless_restore_foreign(void)
{
}

const char* egl_headless_foreign_state(void)
{
    return "foreign GL context suspension is not required by the SDL Windows backend";
}
