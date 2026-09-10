# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional Kit runtime services used by the headless SysID backend.

This is the only portable-backend module that imports Kit application services.
All imports are deferred so importing :mod:`isaacsim.robot_setup.sysid` remains
safe in an ordinary Python process.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .errors import SysIdRuntimeUnavailableError

YieldHook = Callable[[], Awaitable[None]]

_yield_hook: YieldHook | None = None


class CarbLogHandler(logging.Handler):
    """Forward standard backend log records to the Kit console."""

    def emit(self, record: logging.LogRecord) -> None:
        """Forward one Python log record to the matching Carb log level.

        Args:
            record: Python log record to forward.
        """
        try:
            import carb

            message = self.format(record)
            if record.levelno >= logging.ERROR:
                carb.log_error(message)
            elif record.levelno >= logging.WARNING:
                carb.log_warn(message)
            else:
                carb.log_info(message)
        except Exception:
            self.handleError(record)


def install_carb_logging() -> CarbLogHandler | None:
    """Install one package-scoped Carb logging handler when Kit is available.

    Returns:
        Installed handler, an existing package handler, or ``None`` outside Kit.
    """
    if not kit_runtime_available():
        return None
    logger = logging.getLogger("isaacsim.robot_setup.sysid")
    for handler in logger.handlers:
        if isinstance(handler, CarbLogHandler):
            return handler
    handler = CarbLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


def remove_carb_logging(handler: CarbLogHandler | None) -> None:
    """Remove a handler installed by :func:`install_carb_logging`.

    Args:
        handler: Package handler to detach, or ``None``.
    """
    if handler is not None:
        logging.getLogger("isaacsim.robot_setup.sysid").removeHandler(handler)


def set_yield_hook(hook: YieldHook | None) -> None:
    """Override cooperative scheduling for tests or an embedding application.

    Args:
        hook: Awaitable callback, or ``None`` to restore automatic yielding.
    """
    global _yield_hook
    _yield_hook = hook


async def yield_control() -> None:
    """Yield to Kit when available, otherwise to the asyncio event loop."""
    if _yield_hook is not None:
        await _yield_hook()
        return
    if not kit_runtime_available():
        await asyncio.sleep(0)
        return
    from isaacsim.core.experimental.utils import app as app_utils

    await app_utils.update_app_async()


def kit_runtime_available() -> bool:
    """Return whether the process has a bootstrapped Kit application.

    Returns:
        ``True`` only when Kit modules and a live application are available.
    """
    try:
        import omni.kit.app
    except ModuleNotFoundError as exc:
        missing = str(exc.name or "")
        if missing in {"omni", "omni.kit", "omni.kit.app"}:
            return False
        raise
    try:
        return omni.kit.app.get_app() is not None
    except RuntimeError:
        # Kit's Python modules may be installed without a live IApp interface
        # (for example Isaac Sim's python.bat used as an ordinary interpreter).
        return False


def require_kit_runtime(feature: str) -> None:
    """Raise a specific error when a Kit-hosted feature is requested.

    Args:
        feature: Human-readable feature name included in the error.
    """
    if not kit_runtime_available():
        raise SysIdRuntimeUnavailableError(
            f"{feature} requires a bootstrapped Isaac Sim/Kit runtime. "
            "Start SimulationApp or enable isaacsim.robot_setup.sysid in a Kit application."
        )


def get_current_stage() -> Any | None:
    """Return the active Kit stage without importing Kit at module load time.

    Returns:
        Active USD stage from the Core Experimental facade.
    """
    require_kit_runtime("Stage access")
    from isaacsim.core.experimental.utils import stage as stage_utils

    return stage_utils.get_current_stage()


async def open_stage_async(path: str) -> Any:
    """Open a USD stage through the Core Experimental stage facade.

    Args:
        path: USD stage path or URL.

    Returns:
        Newly opened USD stage.
    """
    require_kit_runtime("Opening an Isaac Sim stage")
    from isaacsim.core.experimental.utils import stage as stage_utils

    success, stage = await stage_utils.open_stage_async(path)
    if not success or stage is None:
        raise RuntimeError(f"Failed to open USD stage: {path}")
    return stage


def setup_simulation(*, dt: float | None = None, device: str | None = None) -> None:
    """Configure and initialize physics through ``SimulationManager``.

    Args:
        dt: Optional physics timestep in seconds.
        device: Optional physics device identifier.
    """
    require_kit_runtime("Physics setup")
    from isaacsim.core.simulation_manager import SimulationManager

    kwargs: dict[str, Any] = {}
    if dt is not None:
        kwargs["dt"] = float(dt)
    if device is not None:
        kwargs["device"] = device
    SimulationManager.setup_simulation(**kwargs)


def initialize_physics() -> None:
    """Initialize physics through ``SimulationManager``."""
    require_kit_runtime("Physics initialization")
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.initialize_physics()


def get_physics_dt() -> float:
    """Return the active physics step through ``SimulationManager``.

    Returns:
        Physics timestep in seconds.
    """
    require_kit_runtime("Physics timing")
    from isaacsim.core.simulation_manager import SimulationManager

    physics_scenes = SimulationManager.get_physics_scenes()
    if physics_scenes:
        return float(physics_scenes[0].get_dt())
    # Preserve the historical behavior before SimulationManager has populated
    # its scene registry. The compatibility fallback can create the default
    # scene; normal initialized rollouts use PhysicsScene.get_dt() above.
    return float(SimulationManager.get_physics_dt())


def get_physics_sim_view_warp() -> Any:
    """Return the Warp physics view required by Fabric clone binding.

    ``SimulationManager`` does not expose this clone-specific view publicly, so
    the private access is isolated here until a supported API is available.

    Returns:
        Warp physics simulation view.
    """
    require_kit_runtime("Fabric physics view")
    from isaacsim.core.simulation_manager import SimulationManager

    return SimulationManager._physics_sim_view__warp


def step_simulation(*, steps: int = 1, update_fabric: bool = True) -> None:
    """Advance physics through ``SimulationManager``.

    Args:
        steps: Number of physics steps to execute.
        update_fabric: Whether to synchronize Fabric after stepping.
    """
    step_count = int(steps)
    if step_count < 0:
        raise ValueError("steps must be non-negative.")
    if step_count == 0:
        return
    require_kit_runtime("Physics stepping")
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.step(steps=step_count, update_fabric=bool(update_fabric))


def flush_physics_changes() -> None:
    """Flush pending physics changes through the Kit simulation interface."""
    require_kit_runtime("Physics change flushing")
    import omni.physics.core

    omni.physics.core.get_physics_simulation_interface().flush_changes()


def enable_fabric(enabled: bool) -> None:
    """Enable or disable Fabric through ``SimulationManager``.

    Args:
        enabled: Whether Fabric updates are enabled.
    """
    require_kit_runtime("Fabric configuration")
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.enable_fabric(bool(enabled))


def register_physics_callback(callback: Callable[..., Any], event: Any) -> int:
    """Register a simulation callback through ``SimulationManager``.

    Args:
        callback: Callback invoked for the simulation event.
        event: Simulation event used to trigger the callback.

    Returns:
        Callback registration identifier.
    """
    require_kit_runtime("Physics callbacks")
    from isaacsim.core.simulation_manager import SimulationManager

    return SimulationManager.register_callback(callback, event)


def register_physics_post_step_callback(callback: Callable[..., Any]) -> int:
    """Register a post-physics-step callback without leaking event enums.

    Args:
        callback: Callback invoked after each physics step.

    Returns:
        Callback registration identifier.
    """
    require_kit_runtime("Physics callbacks")
    from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

    return SimulationManager.register_callback(callback, SimulationEvent.PHYSICS_POST_STEP)


def deregister_physics_callback(callback_id: int) -> bool:
    """Remove a callback previously registered through ``SimulationManager``.

    Args:
        callback_id: Identifier returned when the callback was registered.

    Returns:
        Whether the callback was removed.
    """
    require_kit_runtime("Physics callbacks")
    from isaacsim.core.simulation_manager import SimulationManager

    return bool(SimulationManager.deregister_callback(callback_id))
