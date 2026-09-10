# Public API for module omni.kit.loop:

## Other



# Public API for module omni.kit.loop.bindings._loop:

## Classes

- class RunLoopRunner
  - def get_manual_mode(self, name: str = '') -> bool
  - def get_manual_step_size(self, name: str = '') -> float
  - def set_manual_mode(self, enabled: bool = True, name: str = '')
  - def set_manual_step_size(self, dt: float = 0.01667, name: str = '')

## Functions

- def acquire_loop_interface(plugin_name: str = None, library_path: str = None) -> RunLoopRunner
- def release_loop_interface(arg0: RunLoopRunner)
