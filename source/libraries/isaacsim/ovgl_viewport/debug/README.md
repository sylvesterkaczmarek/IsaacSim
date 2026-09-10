# OVGL Viewport

`isaacsim.ovgl_viewport.debug` is a concrete OVGL viewport for tests and debugging. It borrows a populated OVStage,
renders one authored RenderProduct, presents its `LdrColor` output through SDL, and provides first-person debug camera
navigation. Linux uses the existing headless EGL/OpenGL ES context. Windows uses a hidden SDL desktop OpenGL context
for offscreen rendering and loads the OpenGL entry points through SDL, so no ANGLE runtime is required.

The application owns scene authoring, the camera prim, RenderVar and RenderProduct settings, stage lifetime, and its
simulation loop. The viewport has no Foundation or physics dependency. Call `poll_events()`, update the application,
then call `render()` on the application main thread. Set `ViewportConfig.visible` to `False` to capture the same RGBA8
output without a user-visible presentation window. On Linux, invisible rendering uses EGL and does not require X11 or
Wayland. On Windows, it still uses a hidden SDL desktop OpenGL window and therefore requires a desktop-capable graphics
session with OpenGL 4.1 support. The returned `Frame` is borrowed and reused by the next `render()` call; copy
`frame.rgba` before rendering another frame when comparing captures.

Stage population and USD composition remain the application's responsibility. For resolved material images and
dome-light HDRs, the implementation privately uses its pinned OmniClient dependency to obtain cache-local files. This
does not add asset loading or storage concepts to the viewport's public API.

In a visible viewport, press `H` to toggle the window-side debug HUD. Its initial metric is viewport frames per second.
The HUD is deliberately composed after the rendered image, so it does not modify the borrowed `Frame` capture.

This module intentionally does not define a rendering-provider registry, backend-neutral renderer, rendering manager,
generic output model, or asynchronous render API. Those abstractions are deferred until production viewport and
rendering requirements are known.

## Source layout

- `include/isaacsim/ovgl_viewport/debug` contains the supported C++ API.
- `src` contains SDL presentation and viewport orchestration; `src/details` contains all private OVStage and OVGL
  implementation code.
- `src/details/ovgl/gl` contains the internal OpenGL rasterizer. The root Pixi manifest and lock supply the pinned
  `stb_image` headers and SDL SDK; SDL is linked privately as a shared implementation detail. The private Packman
  manifest supplies the pinned OmniClient SDK. Their licenses are installed with the runtime package.
- `bindings/python` and `python` expose the same viewport API to Python.
- `tests/cpp` and `tests/python` contain language-specific tests.
