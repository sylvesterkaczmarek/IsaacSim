"""Phase 1 trajectory recording for MobilityGen SDG.

Records N episodes of headless physics simulation (no rendering) and writes
them to `$MOBILITY_GEN_DATA/recordings/`, ready for `replay_directory.py`.

Run with:
    $ISAAC_SIM_DIR/python.sh record_trajectories.py \
        --enable isaacsim.replicator.mobility_gen.examples
"""

import os
import tempfile

from isaacsim import SimulationApp


def record_trajectories(
    scene_usd: str,
    omap_yaml: str,
    robot_type: str = "CarterRobot",
    scenario: str = "RandomPathFollowingScenario",
    num_episodes: int = 5,
    max_steps: int = 2000,
    data_dir: str = None,
) -> None:
    """Record robot trajectories headlessly using MobilityGen.

    Args:
        scene_usd: Path to the warehouse/environment USD.
        omap_yaml: Path to the ROS-format occupancy map YAML.
        robot_type: A name registered in ROBOTS, e.g. JetbotRobot | CarterRobot |
            H1Robot | SpotRobot, or one of the MultiSensor variants.
        scenario: RandomPathFollowingScenario | RandomAccelerationScenario.
            The teleoperation scenarios need a UI and cannot run headless.
        num_episodes: Number of episodes to record.
        max_steps: Maximum physics steps per episode.
        data_dir: Root data directory; defaults to $MOBILITY_GEN_DATA or ~/MobilityGenData.
    """
    simulation_app = SimulationApp(launch_config={"headless": True, "multi_gpu": False})

    import omni.timeline
    from isaacsim.core.experimental.utils.stage import open_stage, save_stage
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.replicator.experimental.mobility_gen import ROBOTS, SCENARIOS, OccupancyMap, RecordingSession

    if data_dir is None:
        data_dir = os.environ.get(
            "MOBILITY_GEN_DATA",
            os.path.join(os.environ.get("WORKSPACE_DIR", os.path.expanduser("~")), "MobilityGenData"),
        )

    robot_cls = ROBOTS.get(robot_type)
    scenario_cls = SCENARIOS.get(scenario)
    occupancy_map = OccupancyMap.from_ros_yaml(omap_yaml)

    opened, _ = open_stage(scene_usd)
    if not opened:
        simulation_app.close()
        raise RuntimeError(f"Could not open scene USD: {scene_usd}")
    simulation_app.update()

    cached_stage = os.path.join(tempfile.mkdtemp(), "stage.usd")
    save_stage(cached_stage)

    recordings_dir = os.path.join(data_dir, "recordings")
    os.makedirs(recordings_dir, exist_ok=True)

    # RecordingSession owns the ground plane, robot spawn, Config and writer.
    session = RecordingSession()
    session.build(
        robot_cls,
        scenario_cls,
        occupancy_map,
        scene_usd=scene_usd,
        cached_stage_path=cached_stage,
        recordings_dir=recordings_dir,
    )

    # initialize() expects a playing application. initialize_physics() does not
    # start the Kit timeline, so physics is stepped explicitly in the loop below;
    # simulation_app.update() on its own will not tick it.
    omni.timeline.get_timeline_interface().play()
    simulation_app.update()
    session.initialize()

    for episode in range(num_episodes):
        session.reset()
        session.enable_recording()

        for _ in range(max_steps):
            SimulationManager.step(steps=1)
            simulation_app.update()
            if not session.step(robot_cls.physics_dt):
                break

        print(f"Episode {episode + 1}/{num_episodes}: {session.step_count} steps -> {session.recording_path}")
        session.disable_recording()

    session.close()
    omni.timeline.get_timeline_interface().stop()
    simulation_app.close()
