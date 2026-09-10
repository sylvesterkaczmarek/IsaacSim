"""Replay recordings with a custom robot registered at runtime.

Use when replay_directory.py cannot find your robot class (KeyError).
Register the robot before calling load_scenario(), then run the render loop.
Run with $ISAAC_SIM_DIR/python.sh --enable isaacsim.replicator.mobility_gen.examples.
"""

import glob
import os

from isaacsim import SimulationApp


def replay_with_custom_robot(
    input_dir: str,
    custom_robot_class=None,
) -> None:
    """Replay all recordings in input_dir using a custom robot class.

    Args:
        input_dir: Directory containing MobilityGen recording subdirectories.
        custom_robot_class: A registered WheeledMobilityGenRobot or
            PolicyMobilityGenRobot subclass.  If None, only built-in robots
            from isaacsim.replicator.mobility_gen.examples are available.
    """
    simulation_app = SimulationApp(launch_config={"headless": True, "multi_gpu": False})

    import omni.timeline
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.replicator.experimental.mobility_gen import ROBOTS, load_scenario

    if custom_robot_class is not None:
        ROBOTS.register()(custom_robot_class)

    for recording_path in sorted(glob.glob(os.path.join(input_dir, "*"))):
        scenario = load_scenario(recording_path)  # KeyError if robot class missing
        SimulationManager.initialize_physics()
        # initialize_physics() does not start the Kit timeline, so physics is
        # advanced explicitly below; play() is still needed for rendering.
        omni.timeline.get_timeline_interface().play()
        simulation_app.update()
        scenario.enable_rgb_rendering()
        # ... rest of render loop (mirrors replay_directory.py internals):
        #   SimulationManager.step(steps=1); simulation_app.update()
        #   rep.orchestrator.step(rt_subframes=..., delta_time=0.0, pause_timeline=False)
        #   rep.orchestrator.wait_until_complete()

        # Tear the recording down before the next load_scenario(): stop the
        # timeline first, because leaving it playing across disable_rendering()
        # crashes Kit natively.
        omni.timeline.get_timeline_interface().stop()
        scenario.disable_rendering()

    # Stop the timeline again so Kit's shutdown receives the stop event; without
    # it close() can hang waiting for physics teardown.
    omni.timeline.get_timeline_interface().stop()
    simulation_app.close()
