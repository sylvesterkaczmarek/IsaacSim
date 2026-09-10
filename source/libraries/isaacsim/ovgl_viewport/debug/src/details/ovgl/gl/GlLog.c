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

/* Leveled stderr sink for the OpenGL layer. */
#include "GlLog.h"

#include <stdarg.h>
#include <stdio.h>

/* Plain (non-atomic) globals: gl runs its GL work on one thread per
 * context, and the threshold/hook are configured before chatter starts
 * (renderer create). Matches the process-global carb threshold shape. */
static int g_gl_log_threshold = GL_LOG_VERBOSE;
static gl_log_hook_fn g_gl_log_hook = 0;
static void* g_gl_log_hook_user = 0;

void gl_log_set_threshold(int threshold)
{
    g_gl_log_threshold = threshold;
}

int gl_log_get_threshold(void)
{
    return g_gl_log_threshold;
}

void gl_log_set_hook(gl_log_hook_fn fn, void* user)
{
    /* Order matters for a racing reader: clear the fn first on removal. */
    if (!fn)
    {
        g_gl_log_hook = 0;
        g_gl_log_hook_user = 0;
        return;
    }
    g_gl_log_hook_user = user;
    g_gl_log_hook = fn;
}

void gl_log(int level, const char* channel, const char* fmt, ...)
{
    char line[4096];
    int prefix_len;
    va_list args;
    const int want_stderr = level >= g_gl_log_threshold;
    const gl_log_hook_fn hook = g_gl_log_hook;

    if (!want_stderr && !hook)
        return;
    if (!channel)
        channel = "gl";
    prefix_len = snprintf(line, sizeof(line), "%s: ", channel);
    if (prefix_len < 0 || (size_t)prefix_len >= sizeof(line))
        prefix_len = 0;
    va_start(args, fmt);
    vsnprintf(line + prefix_len, sizeof(line) - (size_t)prefix_len, fmt ? fmt : "", args);
    va_end(args);
    if (want_stderr)
    {
        fprintf(stderr, "%s\n", line);
        fflush(stderr);
    }
    if (hook)
        hook(g_gl_log_hook_user, level, channel, line);
}
