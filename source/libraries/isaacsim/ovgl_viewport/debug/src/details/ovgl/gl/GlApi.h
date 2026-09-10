// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(_WIN32)

#    define SDL_OPENGL_1_FUNCTION_TYPEDEFS
#    include <SDL3/SDL_opengl.h>

#    define OVGL_GL_FUNCTIONS(X)                                                                                       \
        X(ActiveTexture, PFNGLACTIVETEXTUREPROC)                                                                       \
        X(AttachShader, PFNGLATTACHSHADERPROC)                                                                         \
        X(BindBuffer, PFNGLBINDBUFFERPROC)                                                                             \
        X(BindBufferBase, PFNGLBINDBUFFERBASEPROC)                                                                     \
        X(BindBufferRange, PFNGLBINDBUFFERRANGEPROC)                                                                   \
        X(BindFramebuffer, PFNGLBINDFRAMEBUFFERPROC)                                                                   \
        X(BindRenderbuffer, PFNGLBINDRENDERBUFFERPROC)                                                                 \
        X(BindSampler, PFNGLBINDSAMPLERPROC)                                                                           \
        X(BindTexture, PFNGLBINDTEXTUREPROC)                                                                           \
        X(BindVertexArray, PFNGLBINDVERTEXARRAYPROC)                                                                   \
        X(BlendColor, PFNGLBLENDCOLORPROC)                                                                             \
        X(BlendEquationSeparate, PFNGLBLENDEQUATIONSEPARATEPROC)                                                       \
        X(BlendFunc, PFNGLBLENDFUNCPROC)                                                                               \
        X(BlendFuncSeparate, PFNGLBLENDFUNCSEPARATEPROC)                                                               \
        X(BlitFramebuffer, PFNGLBLITFRAMEBUFFERPROC)                                                                   \
        X(BufferData, PFNGLBUFFERDATAPROC)                                                                             \
        X(BufferSubData, PFNGLBUFFERSUBDATAPROC)                                                                       \
        X(CheckFramebufferStatus, PFNGLCHECKFRAMEBUFFERSTATUSPROC)                                                     \
        X(Clear, PFNGLCLEARPROC)                                                                                       \
        X(ClearColor, PFNGLCLEARCOLORPROC)                                                                             \
        X(ClearDepth, PFNGLCLEARDEPTHPROC)                                                                             \
        X(ClearDepthf, PFNGLCLEARDEPTHFPROC)                                                                           \
        X(ColorMask, PFNGLCOLORMASKPROC)                                                                               \
        X(CompileShader, PFNGLCOMPILESHADERPROC)                                                                       \
        X(CompressedTexImage2D, PFNGLCOMPRESSEDTEXIMAGE2DPROC)                                                         \
        X(CreateProgram, PFNGLCREATEPROGRAMPROC)                                                                       \
        X(CreateShader, PFNGLCREATESHADERPROC)                                                                         \
        X(CullFace, PFNGLCULLFACEPROC)                                                                                 \
        X(DeleteBuffers, PFNGLDELETEBUFFERSPROC)                                                                       \
        X(DeleteFramebuffers, PFNGLDELETEFRAMEBUFFERSPROC)                                                             \
        X(DeleteProgram, PFNGLDELETEPROGRAMPROC)                                                                       \
        X(DeleteRenderbuffers, PFNGLDELETERENDERBUFFERSPROC)                                                           \
        X(DeleteShader, PFNGLDELETESHADERPROC)                                                                         \
        X(DeleteTextures, PFNGLDELETETEXTURESPROC)                                                                     \
        X(DeleteVertexArrays, PFNGLDELETEVERTEXARRAYSPROC)                                                             \
        X(DepthFunc, PFNGLDEPTHFUNCPROC)                                                                               \
        X(DepthMask, PFNGLDEPTHMASKPROC)                                                                               \
        X(DepthRange, PFNGLDEPTHRANGEPROC)                                                                             \
        X(DepthRangef, PFNGLDEPTHRANGEFPROC)                                                                           \
        X(Disable, PFNGLDISABLEPROC)                                                                                   \
        X(DrawArrays, PFNGLDRAWARRAYSPROC)                                                                             \
        X(DrawBuffer, PFNGLDRAWBUFFERPROC)                                                                             \
        X(DrawElements, PFNGLDRAWELEMENTSPROC)                                                                         \
        X(DrawElementsBaseVertex, PFNGLDRAWELEMENTSBASEVERTEXPROC)                                                     \
        X(DrawElementsInstanced, PFNGLDRAWELEMENTSINSTANCEDPROC)                                                       \
        X(Enable, PFNGLENABLEPROC)                                                                                     \
        X(EnableVertexAttribArray, PFNGLENABLEVERTEXATTRIBARRAYPROC)                                                   \
        X(Finish, PFNGLFINISHPROC)                                                                                     \
        X(FramebufferRenderbuffer, PFNGLFRAMEBUFFERRENDERBUFFERPROC)                                                   \
        X(FramebufferTexture2D, PFNGLFRAMEBUFFERTEXTURE2DPROC)                                                         \
        X(FrontFace, PFNGLFRONTFACEPROC)                                                                               \
        X(GenBuffers, PFNGLGENBUFFERSPROC)                                                                             \
        X(GenFramebuffers, PFNGLGENFRAMEBUFFERSPROC)                                                                   \
        X(GenRenderbuffers, PFNGLGENRENDERBUFFERSPROC)                                                                 \
        X(GenTextures, PFNGLGENTEXTURESPROC)                                                                           \
        X(GenVertexArrays, PFNGLGENVERTEXARRAYSPROC)                                                                   \
        X(GenerateMipmap, PFNGLGENERATEMIPMAPPROC)                                                                     \
        X(GetBooleanv, PFNGLGETBOOLEANVPROC)                                                                           \
        X(GetError, PFNGLGETERRORPROC)                                                                                 \
        X(GetFloatv, PFNGLGETFLOATVPROC)                                                                               \
        X(GetInteger64i_v, PFNGLGETINTEGER64I_VPROC)                                                                   \
        X(GetIntegeri_v, PFNGLGETINTEGERI_VPROC)                                                                       \
        X(GetIntegerv, PFNGLGETINTEGERVPROC)                                                                           \
        X(GetProgramInfoLog, PFNGLGETPROGRAMINFOLOGPROC)                                                               \
        X(GetProgramiv, PFNGLGETPROGRAMIVPROC)                                                                         \
        X(GetShaderInfoLog, PFNGLGETSHADERINFOLOGPROC)                                                                 \
        X(GetShaderiv, PFNGLGETSHADERIVPROC)                                                                           \
        X(GetString, PFNGLGETSTRINGPROC)                                                                               \
        X(GetUniformBlockIndex, PFNGLGETUNIFORMBLOCKINDEXPROC)                                                         \
        X(GetUniformLocation, PFNGLGETUNIFORMLOCATIONPROC)                                                             \
        X(IsEnabled, PFNGLISENABLEDPROC)                                                                               \
        X(IsFramebuffer, PFNGLISFRAMEBUFFERPROC)                                                                       \
        X(LinkProgram, PFNGLLINKPROGRAMPROC)                                                                           \
        X(MapBufferRange, PFNGLMAPBUFFERRANGEPROC)                                                                     \
        X(PatchParameteri, PFNGLPATCHPARAMETERIPROC)                                                                   \
        X(PixelStorei, PFNGLPIXELSTOREIPROC)                                                                           \
        X(PolygonMode, PFNGLPOLYGONMODEPROC)                                                                           \
        X(PolygonOffset, PFNGLPOLYGONOFFSETPROC)                                                                       \
        X(PrimitiveRestartIndex, PFNGLPRIMITIVERESTARTINDEXPROC)                                                       \
        X(ReadBuffer, PFNGLREADBUFFERPROC)                                                                             \
        X(ReadPixels, PFNGLREADPIXELSPROC)                                                                             \
        X(RenderbufferStorage, PFNGLRENDERBUFFERSTORAGEPROC)                                                           \
        X(Scissor, PFNGLSCISSORPROC)                                                                                   \
        X(ShaderSource, PFNGLSHADERSOURCEPROC)                                                                         \
        X(TexImage2D, PFNGLTEXIMAGE2DPROC)                                                                             \
        X(TexParameterf, PFNGLTEXPARAMETERFPROC)                                                                       \
        X(TexParameteri, PFNGLTEXPARAMETERIPROC)                                                                       \
        X(Uniform1f, PFNGLUNIFORM1FPROC)                                                                               \
        X(Uniform1i, PFNGLUNIFORM1IPROC)                                                                               \
        X(Uniform1iv, PFNGLUNIFORM1IVPROC)                                                                             \
        X(Uniform2f, PFNGLUNIFORM2FPROC)                                                                               \
        X(Uniform3fv, PFNGLUNIFORM3FVPROC)                                                                             \
        X(Uniform4f, PFNGLUNIFORM4FPROC)                                                                               \
        X(Uniform4fv, PFNGLUNIFORM4FVPROC)                                                                             \
        X(UniformBlockBinding, PFNGLUNIFORMBLOCKBINDINGPROC)                                                           \
        X(UniformMatrix3fv, PFNGLUNIFORMMATRIX3FVPROC)                                                                 \
        X(UniformMatrix4fv, PFNGLUNIFORMMATRIX4FVPROC)                                                                 \
        X(UnmapBuffer, PFNGLUNMAPBUFFERPROC)                                                                           \
        X(UseProgram, PFNGLUSEPROGRAMPROC)                                                                             \
        X(VertexAttribDivisor, PFNGLVERTEXATTRIBDIVISORPROC)                                                           \
        X(VertexAttribIPointer, PFNGLVERTEXATTRIBIPOINTERPROC)                                                         \
        X(VertexAttribPointer, PFNGLVERTEXATTRIBPOINTERPROC)                                                           \
        X(Viewport, PFNGLVIEWPORTPROC)

#    ifdef __cplusplus
extern "C"
{
#    endif

#    define OVGL_DECLARE_GL_FUNCTION(name, type) extern type ovgl_gl##name;
    OVGL_GL_FUNCTIONS(OVGL_DECLARE_GL_FUNCTION)
#    undef OVGL_DECLARE_GL_FUNCTION

    int ovgl_gl_load(void);

#    ifdef __cplusplus
}
#    endif

#    define glActiveTexture ovgl_glActiveTexture
#    define glAttachShader ovgl_glAttachShader
#    define glBindBuffer ovgl_glBindBuffer
#    define glBindBufferBase ovgl_glBindBufferBase
#    define glBindBufferRange ovgl_glBindBufferRange
#    define glBindFramebuffer ovgl_glBindFramebuffer
#    define glBindRenderbuffer ovgl_glBindRenderbuffer
#    define glBindSampler ovgl_glBindSampler
#    define glBindTexture ovgl_glBindTexture
#    define glBindVertexArray ovgl_glBindVertexArray
#    define glBlendColor ovgl_glBlendColor
#    define glBlendEquationSeparate ovgl_glBlendEquationSeparate
#    define glBlendFunc ovgl_glBlendFunc
#    define glBlendFuncSeparate ovgl_glBlendFuncSeparate
#    define glBlitFramebuffer ovgl_glBlitFramebuffer
#    define glBufferData ovgl_glBufferData
#    define glBufferSubData ovgl_glBufferSubData
#    define glCheckFramebufferStatus ovgl_glCheckFramebufferStatus
#    define glClear ovgl_glClear
#    define glClearColor ovgl_glClearColor
#    define glClearDepth ovgl_glClearDepth
#    define glClearDepthf ovgl_glClearDepthf
#    define glColorMask ovgl_glColorMask
#    define glCompileShader ovgl_glCompileShader
#    define glCompressedTexImage2D ovgl_glCompressedTexImage2D
#    define glCreateProgram ovgl_glCreateProgram
#    define glCreateShader ovgl_glCreateShader
#    define glCullFace ovgl_glCullFace
#    define glDeleteBuffers ovgl_glDeleteBuffers
#    define glDeleteFramebuffers ovgl_glDeleteFramebuffers
#    define glDeleteProgram ovgl_glDeleteProgram
#    define glDeleteRenderbuffers ovgl_glDeleteRenderbuffers
#    define glDeleteShader ovgl_glDeleteShader
#    define glDeleteTextures ovgl_glDeleteTextures
#    define glDeleteVertexArrays ovgl_glDeleteVertexArrays
#    define glDepthFunc ovgl_glDepthFunc
#    define glDepthMask ovgl_glDepthMask
#    define glDepthRange ovgl_glDepthRange
#    define glDepthRangef ovgl_glDepthRangef
#    define glDisable ovgl_glDisable
#    define glDrawArrays ovgl_glDrawArrays
#    define glDrawBuffer ovgl_glDrawBuffer
#    define glDrawElements ovgl_glDrawElements
#    define glDrawElementsBaseVertex ovgl_glDrawElementsBaseVertex
#    define glDrawElementsInstanced ovgl_glDrawElementsInstanced
#    define glEnable ovgl_glEnable
#    define glEnableVertexAttribArray ovgl_glEnableVertexAttribArray
#    define glFinish ovgl_glFinish
#    define glFramebufferRenderbuffer ovgl_glFramebufferRenderbuffer
#    define glFramebufferTexture2D ovgl_glFramebufferTexture2D
#    define glFrontFace ovgl_glFrontFace
#    define glGenBuffers ovgl_glGenBuffers
#    define glGenFramebuffers ovgl_glGenFramebuffers
#    define glGenRenderbuffers ovgl_glGenRenderbuffers
#    define glGenTextures ovgl_glGenTextures
#    define glGenVertexArrays ovgl_glGenVertexArrays
#    define glGenerateMipmap ovgl_glGenerateMipmap
#    define glGetBooleanv ovgl_glGetBooleanv
#    define glGetError ovgl_glGetError
#    define glGetFloatv ovgl_glGetFloatv
#    define glGetInteger64i_v ovgl_glGetInteger64i_v
#    define glGetIntegeri_v ovgl_glGetIntegeri_v
#    define glGetIntegerv ovgl_glGetIntegerv
#    define glGetProgramInfoLog ovgl_glGetProgramInfoLog
#    define glGetProgramiv ovgl_glGetProgramiv
#    define glGetShaderInfoLog ovgl_glGetShaderInfoLog
#    define glGetShaderiv ovgl_glGetShaderiv
#    define glGetString ovgl_glGetString
#    define glGetUniformBlockIndex ovgl_glGetUniformBlockIndex
#    define glGetUniformLocation ovgl_glGetUniformLocation
#    define glIsEnabled ovgl_glIsEnabled
#    define glIsFramebuffer ovgl_glIsFramebuffer
#    define glLinkProgram ovgl_glLinkProgram
#    define glMapBufferRange ovgl_glMapBufferRange
#    define glPatchParameteri ovgl_glPatchParameteri
#    define glPixelStorei ovgl_glPixelStorei
#    define glPolygonMode ovgl_glPolygonMode
#    define glPolygonOffset ovgl_glPolygonOffset
#    define glPrimitiveRestartIndex ovgl_glPrimitiveRestartIndex
#    define glReadBuffer ovgl_glReadBuffer
#    define glReadPixels ovgl_glReadPixels
#    define glRenderbufferStorage ovgl_glRenderbufferStorage
#    define glScissor ovgl_glScissor
#    define glShaderSource ovgl_glShaderSource
#    define glTexImage2D ovgl_glTexImage2D
#    define glTexParameterf ovgl_glTexParameterf
#    define glTexParameteri ovgl_glTexParameteri
#    define glUniform1f ovgl_glUniform1f
#    define glUniform1i ovgl_glUniform1i
#    define glUniform1iv ovgl_glUniform1iv
#    define glUniform2f ovgl_glUniform2f
#    define glUniform3fv ovgl_glUniform3fv
#    define glUniform4f ovgl_glUniform4f
#    define glUniform4fv ovgl_glUniform4fv
#    define glUniformBlockBinding ovgl_glUniformBlockBinding
#    define glUniformMatrix3fv ovgl_glUniformMatrix3fv
#    define glUniformMatrix4fv ovgl_glUniformMatrix4fv
#    define glUnmapBuffer ovgl_glUnmapBuffer
#    define glUseProgram ovgl_glUseProgram
#    define glVertexAttribDivisor ovgl_glVertexAttribDivisor
#    define glVertexAttribIPointer ovgl_glVertexAttribIPointer
#    define glVertexAttribPointer ovgl_glVertexAttribPointer
#    define glViewport ovgl_glViewport

#elif defined(NUSD_DESKTOP_GL)
#    ifdef __APPLE__
#        define GL_SILENCE_DEPRECATION
#        include <OpenGL/gl3.h>
#        include <OpenGL/gl3ext.h>
#    else
#        include <GL/gl.h>
#        include <GL/glext.h>
#    endif
#elif defined(ISAACSIM_OVGL_USE_GLCOREARB_HEADERS)
#    define GL_GLEXT_PROTOTYPES 1
#    include <GL/glcorearb.h>
#else
// gl32.h defines GL_APIENTRY, which the extension declarations require.
// clang-format off
#    include <GLES3/gl32.h>
#    include <GLES2/gl2ext.h>
// clang-format on
#endif
