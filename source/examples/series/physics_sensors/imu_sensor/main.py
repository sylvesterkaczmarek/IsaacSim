# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the position, orientation, velocity, and acceleration of a prim as an IMU would."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import imu
import viewer
import warp as wp
from imu import BodyState, ImuReading, ImuSensor, read_quaternion_from_wxyz
from isaacsim.foundation.objects import Cube, Prim, Stage, Xform
from isaacsim.foundation.prims import ColliderBody, RigidBody
from isaacsim.physics.entities import RigidBodyEntity
from isaacsim.physics.manager import PhysicsManager
from isaacsim.physics_engines.ovphysx import activate, set_suppress_readback, shutdown

# The name OvPhysX registers itself under, which the registry matches exactly.
ENGINE_NAME = "ovphysx"

BODY_PATH = "/World/Body"
SENSOR_PATH = "/World/Body/Imu"

# Sensor mount in the body frame: raised along body Z and yawed a quarter turn about it. The yaw
# keeps the sensor frame distinct from the body frame while leaving sensor +Z along world up, so at
# rest the reading is one gravity on Z and during the fall the velocity is along -Z.
MOUNT_TRANSLATION = (0.0, 0.0, 0.25)
# Ordered wxyz, because this is authored onto the stage. It is converted once, on the way into the
# sensor, which works in the xyzw ordering the physics buffers use.
MOUNT_ORIENTATION = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))

# Physics scene configuration, authored on the stage and reused as the sensor's gravity vector.
GRAVITY_DIRECTION = (0.0, 0.0, -1.0)
GRAVITY_MAGNITUDE = 9.81

# The body is dropped from `DROP_HEIGHT` while spinning about world Z. Its underside starts 1.5 m
# up, so it lands after roughly 0.55 s (33 steps) and settles well before the last step.
TIME_STEP = 1.0 / 60.0
STEP_COUNT = 180
BODY_SIZE = 1.0
DROP_HEIGHT = 2.0
INITIAL_ANGULAR_VELOCITY = (0.0, 0.0, 3.0)

# Rolling average window, equivalent to the IsaacImuSensor filter width attributes.
FILTER_WIDTH = 5

# `FREE_FALL_STEP` is airborne and late enough for the sample buffer to be full. The airborne
# window ends before first contact at step 33, so its acceleration is exactly gravity.
FREE_FALL_STEP = 25
AIRBORNE_STEPS = range(1, 31)

# Tolerances for the reported checks, in m/s^2, rad/s, and m.
ACCELERATION_TOLERANCE = 0.5
ANGULAR_VELOCITY_TOLERANCE = 0.25
VELOCITY_TOLERANCE = 0.05
POSITION_TOLERANCE = 0.01


def _author_stage() -> tuple[str, list[float], list[float]]:
    """Author the scene and read the sensor mount back from USD.

    Returns:
        The flattened USDA, the mount translation, and the mount orientation as ``xyzw``.

    """
    stage = Stage("openusd").create_stage()
    try:
        stage.set_up_axis("Z")
        stage.set_units(meters_per_unit=1.0, kilograms_per_unit=1.0)
        stage.define_prim("/World")

        Cube("/World/Ground", sizes=1.0, translations=[0.0, 0.0, -0.5], scales=[10.0, 10.0, 1.0])
        Cube(BODY_PATH, sizes=BODY_SIZE, translations=[0.0, 0.0, DROP_HEIGHT])

        stage.define_prim("/World/PhysicsScene", "PhysicsScene")
        physics_scene = Prim("/World/PhysicsScene")
        physics_scene.set_attribute_values("physics:gravityDirection", list(GRAVITY_DIRECTION))
        physics_scene.set_attribute_values("physics:gravityMagnitude", GRAVITY_MAGNITUDE)
        ColliderBody("/World/Ground")
        ColliderBody(BODY_PATH)
        RigidBody(BODY_PATH)

        # A plain Xform child of the body. It is not simulated, so its pose is a fixed offset.
        stage.define_prim(SENSOR_PATH, "Xform")
        sensor = Xform(
            SENSOR_PATH,
            translations=list(MOUNT_TRANSLATION),
            orientations=list(MOUNT_ORIENTATION),
        )
        mount_translations, mount_orientations = sensor.get_local_poses()
        mount_translation = mount_translations[0].list()
        # Foundation returns wxyz; the sensor works in xyzw.
        mount_orientation = read_quaternion_from_wxyz(mount_orientations[0].list())
        return stage.export_stage_to_string(), mount_translation, mount_orientation
    finally:
        stage.close_stage()


def _read_body_state(body: RigidBodyEntity) -> BodyState:
    """Read the world-frame state of a single rigid body.

    Args:
        body: Rigid body entity wrapping exactly one prim.

    Returns:
        The body state, with the orientation passed through unreordered.

    """
    # The entity hands back the engine's ordering, xyzw, which is the sensor's, so nothing is
    # reordered here. Were that to change, the free-fall checks below fail rather than drift
    # quietly: the body yaws, so reading its quaternion the other way tilts the sensor frame.
    positions, orientations = body.get_world_poses()
    linear_velocities, angular_velocities = body.get_velocities()
    return BodyState(
        position=positions[0].list(),
        orientation=orientations[0].list(),
        linear_velocity=linear_velocities[0].list(),
        angular_velocity=angular_velocities[0].list(),
    )


def _read_engine_acceleration(body: RigidBodyEntity) -> list[float]:
    """Read the engine's own linear acceleration for a single rigid body.

    `RigidBodyEntity` exposes no accessor for this, so it is read by buffer name.

    Args:
        body: Rigid body entity wrapping exactly one prim.

    Returns:
        World-frame linear acceleration at the body origin.

    """
    return body.get_data("accelerations")[0].list()[0:3]


def _format_vector(values: Sequence[float]) -> str:
    """Format a short sequence of numbers for display.

    Args:
        values: Numbers to format.

    Returns:
        A bracketed, comma-separated string.

    """
    return "[" + ", ".join(f"{float(value): .3f}" for value in values) + "]"


def _print_reading(label: str, reading: ImuReading) -> None:
    """Print one reading.

    Args:
        label: Phase name shown above the reading.
        reading: Reading to print.

    """
    print(f"{label} at t = {reading.time:.3f} s:")
    print(f"  world position               {_format_vector(reading.position)} m")
    print(f"  world orientation (xyzw)     {_format_vector(reading.orientation)}")
    print(f"  sensor linear velocity       {_format_vector(reading.linear_velocity)} m/s")
    print(f"  sensor angular velocity      {_format_vector(reading.angular_velocity)} rad/s")
    print(f"  sensor linear acceleration   {_format_vector(reading.linear_acceleration)} m/s^2")


def _compute_difference(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute the distance between two vectors.

    Args:
        left: Left operand.
        right: Right operand.

    Returns:
        The Euclidean distance.

    """
    difference = wp.vec3d([float(value) for value in left]) - wp.vec3d([float(value) for value in right])
    return float(wp.length(difference))


def _check_free_fall(reading: ImuReading) -> None:
    """Check the airborne reading and print the outcome.

    Args:
        reading: Reading captured while the body is airborne.

    Raises:
        RuntimeError: If the reading does not match free fall.

    """
    # In free fall an accelerometer measures no specific force.
    world_acceleration = wp.quat_rotate(reading.orientation, reading.linear_acceleration)
    if _compute_difference(world_acceleration, [0.0, 0.0, 0.0]) > ACCELERATION_TOLERANCE:
        raise RuntimeError(f"Free-fall acceleration {world_acceleration} is not near zero.")
    print("Free-fall proper acceleration is near zero.")

    # Body and mount yaw about the same axis, so sensor Z stays along world up.
    expected_angular_velocity = [0.0, 0.0, INITIAL_ANGULAR_VELOCITY[2]]
    if _compute_difference(reading.angular_velocity, expected_angular_velocity) > ANGULAR_VELOCITY_TOLERANCE:
        raise RuntimeError(f"Free-fall angular velocity {reading.angular_velocity} is not about the sensor Z axis.")
    print("Free-fall angular velocity is read on the sensor Z axis.")

    velocity = reading.linear_velocity
    if velocity[2] >= 0.0 or _compute_difference(velocity, [0.0, 0.0, velocity[2]]) > VELOCITY_TOLERANCE:
        raise RuntimeError(f"Free-fall sensor velocity {velocity} is not along the sensor -Z axis.")
    print("Free-fall velocity is read on the sensor -Z axis.")


def _check_resting(reading: ImuReading, gravity: list[float]) -> None:
    """Check the settled reading and print the outcome.

    Args:
        reading: Reading captured after the body has settled.
        gravity: World-frame gravitational acceleration.

    Raises:
        RuntimeError: If the reading does not match rest.

    """
    # At rest the reading is the reaction to gravity, along world up.
    world_acceleration = wp.quat_rotate(reading.orientation, reading.linear_acceleration)
    expected = [-float(value) for value in gravity]
    if _compute_difference(world_acceleration, expected) > ACCELERATION_TOLERANCE:
        raise RuntimeError(f"Resting acceleration {world_acceleration} does not match {expected}.")
    print("Resting accelerometer reads one gravity along world up.")

    # At rest the cube's center is half a side up, and it has only yawed, so the offset is world Z.
    expected_position = [0.0, 0.0, 0.5 * BODY_SIZE + MOUNT_TRANSLATION[2]]
    if _compute_difference(reading.position, expected_position) > POSITION_TOLERANCE:
        raise RuntimeError(f"Resting sensor position {reading.position} does not match {expected_position}.")
    print("Resting sensor sits one mount offset above the body center.")


def _parse_arguments() -> argparse.Namespace:
    """Parse the example's command line.

    Returns:
        Parsed arguments.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the unit tests for imu.py and exit, without starting a simulation.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Show a live window with the body and the sensor readings side by side. Needs the "
        "optional 'ovui' and 'glfw' packages.",
    )
    return parser.parse_args()


def _run_batch(
    body: RigidBodyEntity,
    sensor: ImuSensor,
    gravity: list[float],
) -> None:
    """Simulate to completion, then report and check the readings.

    Args:
        body: Rigid body entity wrapping exactly one prim.
        sensor: Sensor to drive.
        gravity: World-frame gravitational acceleration.

    Raises:
        RuntimeError: If any reported check does not hold.

    """
    print(f"IMU at {SENSOR_PATH} mounted on {BODY_PATH}.")
    print("Reporting specific force, the proper acceleration.")

    free_fall_reading: ImuReading | None = None
    previous_linear_velocity: list[float] | None = None
    engine_difference = 0.0
    gravity_difference = 0.0
    cross_checked = False

    manager = PhysicsManager.get_instance()
    for step in range(STEP_COUNT + 1):
        manager.step()
        # The buffers hold the state after the step, so the manager's own clock is the sample time.
        time = manager.get_simulated_time()
        state = _read_body_state(body)
        sensor.sample(time, state)

        if previous_linear_velocity is not None and step in AIRBORNE_STEPS:
            finite_difference = (state.linear_velocity - previous_linear_velocity) * wp.float64(1.0 / TIME_STEP)
            # The engine's buffer holds this same difference, so this only confirms the slices
            # line up, not that either value is right.
            engine_difference = max(
                engine_difference, _compute_difference(finite_difference, _read_engine_acceleration(body))
            )
            # Independent of both buffers: airborne, the difference must reproduce gravity.
            gravity_difference = max(gravity_difference, _compute_difference(finite_difference, gravity))
            cross_checked = True
        previous_linear_velocity = state.linear_velocity

        if step == FREE_FALL_STEP:
            free_fall_reading = sensor.read()

    resting_reading = sensor.read()
    if free_fall_reading is None or resting_reading is None:
        raise RuntimeError("The sensor did not buffer enough samples to produce a reading.")

    _print_reading("Free fall", free_fall_reading)
    _print_reading("At rest", resting_reading)

    # Each sample is stamped with the time the step advanced to, so a reading carries the time of
    # its newest sample. Nothing else here would notice the whole series shifted by a step.
    for label, reading, step in (
        ("free-fall", free_fall_reading, FREE_FALL_STEP),
        ("resting", resting_reading, STEP_COUNT),
    ):
        expected_time = (step + 1) * TIME_STEP
        if abs(reading.time - expected_time) > 0.5 * TIME_STEP:
            raise RuntimeError(f"The {label} reading is stamped {reading.time:.3f} s, expected {expected_time:.3f} s.")
    print("Readings are stamped with the time their newest sample was taken.")

    if not cross_checked:
        raise RuntimeError("The airborne acceleration cross-check did not run.")
    print(f"Largest finite-difference to engine acceleration gap: {engine_difference:.3f} m/s^2.")
    if engine_difference > ACCELERATION_TOLERANCE:
        raise RuntimeError("The finite-difference acceleration disagrees with the engine acceleration.")
    print(f"Largest airborne finite-difference to gravity gap: {gravity_difference:.3f} m/s^2.")
    if gravity_difference > ACCELERATION_TOLERANCE:
        raise RuntimeError("The airborne finite-difference acceleration does not match gravity.")
    print("Airborne finite-difference acceleration matches gravity.")

    _check_free_fall(free_fall_reading)
    _check_resting(resting_reading, gravity)


def main() -> None:
    """Simulate a spinning falling body and report IMU readings for a sensor mounted on it."""
    arguments = _parse_arguments()

    if arguments.self_test:
        # The tests need no simulation, so run them before any engine or stage is created.
        import test_imu

        raise SystemExit(1 if test_imu.run_all() else 0)

    if not activate():
        raise RuntimeError("OvPhysX physics engine is unavailable.")

    ovstage_stage = None
    simulation_initialized = False
    try:

        wp.init()
        stage_text, mount_translation, mount_orientation = _author_stage()
        ovstage_stage = Stage("ovstage").import_stage_from_string(stage_text, make_default=False)
        manager = PhysicsManager.get_instance()
        if not manager.switch_physics_engine(ENGINE_NAME):
            raise RuntimeError("OvPhysX physics engine is unavailable.")
        set_suppress_readback(False)

        manager.setup(TIME_STEP)
        # ovstage_stage is held for the lifetime of the simulation: the manager takes a pointer to
        # it and does not own it.
        if not manager.initialize(ovstage_stage.get_stage_ptr(), 0):
            raise RuntimeError("Physics initialization failed.")
        simulation_initialized = True
        body = RigidBodyEntity(ENGINE_NAME, BODY_PATH)
        body.set_velocities(angular_velocities=list(INITIAL_ANGULAR_VELOCITY))

        gravity = wp.vec3d([value * GRAVITY_MAGNITUDE for value in GRAVITY_DIRECTION])
        sensor = ImuSensor(
            gravity,
            mount_translation=mount_translation,
            mount_orientation=mount_orientation,
            filter_width=FILTER_WIDTH,
        )

        if arguments.ui:

            def _advance(step: int) -> BodyState:
                """Step the simulation once, sample the sensor, and return the new body state.

                Args:
                    step: Current simulation step.

                Returns:
                    The resulting value.
                """
                manager.step()
                body_state = _read_body_state(body)
                sensor.sample(manager.get_simulated_time(), body_state)
                return body_state

            def _restart() -> None:
                """Teleport the body back to its drop pose and respin it.

                The tensor buffer stores a position followed by an xyzw quaternion, which is the
                sensor's ordering too, so the identity goes in as-is. Buffered samples are discarded
                because the jump makes the finite difference across it meaningless.
                """
                body.set_world_poses([0.0, 0.0, viewer.DROP_HEIGHT], list(imu.IDENTITY_QUATERNION))
                body.set_velocities([0.0, 0.0, 0.0], list(INITIAL_ANGULAR_VELOCITY))
                sensor.reset()

            last_reading = viewer.run(
                sensor=sensor,
                advance=_advance,
                restart=_restart,
                format_vector=_format_vector,
                sensor_path=SENSOR_PATH,
                body_size=BODY_SIZE,
                step_count=STEP_COUNT,
            )
            if last_reading is not None:
                _print_reading("Last reading before exit", last_reading)
        else:
            _run_batch(body, sensor, gravity)
    finally:
        if simulation_initialized:
            manager.invalidate()
        if ovstage_stage is not None:
            ovstage_stage.close_stage()
        shutdown()


if __name__ == "__main__":
    main()
