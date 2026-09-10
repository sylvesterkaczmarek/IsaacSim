# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the position, velocity, and effort of every degree of freedom of an articulation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import viewer
import warp as wp
from isaacsim.foundation.objects import Stage
from isaacsim.foundation.prims import Articulation
from isaacsim.physics.entities import ArticulationEntity
from isaacsim.physics.manager import PhysicsManager
from isaacsim.physics_engines.ovphysx import activate, set_suppress_readback, shutdown
from joint_state import ArticulationState, Dof, JointStateReading, JointStateSensor

# The scene is USDA text rather than object API calls: a joint points at the bodies it connects
# through the `physics:body0` and `physics:body1` relationships, and `Prim` writes attributes only.
# `Stage.import_stage_from_string()` takes the text directly, so it still lives in this file.
#
# A two-DOF robot: a carriage that slides along a fixed mast, carrying a bar that spins about the
# vertical axis. Neither joint is driven or damped, so both DOFs move under gravity and inertia
# alone and their state has a closed form for the checks below to use.
#
# Authored in centimetres on purpose. Every joint quantity the physics engine reports is in stage
# units, so a translation DOF reads centimetres here while a rotation DOF reads radians whatever the
# stage's linear unit. Converting one and not the other is what joint_state.py exists to show.
SCENE = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 0.01
    kilogramsPerUnit = 1
    upAxis = "Z"
    timeCodesPerSecond = 60
)

def Xform "World"
{
    def PhysicsScene "PhysicsScene"
    {
        vector3f physics:gravityDirection = (0, 0, -1)
        # Stage units per second squared: 981 cm/s^2 is one standard gravity on a centimeter stage.
        float physics:gravityMagnitude = 981
    }

    def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
    {
        # A fixed joint with no body0 anchors the mast to the world, making this a fixed-base
        # articulation. It constrains every axis, so it contributes no DOF.
        def PhysicsFixedJoint "WorldJoint"
        {
            rel physics:body1 = </World/Robot/Mast>
        }

        # The links carry no collision geometry: nothing in this scene touches anything else, and
        # mass and inertia are stated outright rather than derived from colliders.
        #
        # Damping is stated outright too. PhysX applies a small angular damping to a rigid body by
        # default, which quietly bleeds a fiftieth of the spin rate away every second; zeroing it
        # is what makes "undriven and undamped" true and the closed forms below exact. That is a
        # PhysX setting, so it comes from PhysxSchema rather than UsdPhysics.
        def Cube "Mast" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"])
        {
            double size = 1
            double3 xformOp:translate = (0, 0, 60)
            float3 xformOp:scale = (10, 10, 120)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            float physxRigidBody:linearDamping = 0
            float physxRigidBody:angularDamping = 0
            float physics:mass = 20
            float3 physics:diagonalInertia = (24167, 24167, 333)
        }

        def Cube "Carriage" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"])
        {
            double size = 1
            double3 xformOp:translate = (0, 0, 100)
            float3 xformOp:scale = (30, 30, 10)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            float physxRigidBody:linearDamping = 0
            float physxRigidBody:angularDamping = 0
            float physics:mass = 2
            float3 physics:diagonalInertia = (167, 167, 300)
        }

        # The lift DOF. Its zero is the authored pose, so its position counts centimeters travelled
        # down the mast, and its lower limit is the floor the carriage comes to rest on.
        #
        # A velocity drive lowers it at a fixed rate rather than letting it fall. A constant rate is
        # what makes the descent checkable: the reported velocity is the target exactly, in stage
        # units, so converting it is the whole of the test. The drive target, like every other joint
        # quantity here, is in stage units -- 50 cm/s, which is 0.5 m/s.
        def PhysicsPrismaticJoint "lift" (prepend apiSchemas = ["PhysicsDriveAPI:linear"])
        {
            float drive:linear:physics:targetVelocity = -50
            float drive:linear:physics:stiffness = 0
            float drive:linear:physics:damping = 10000
            # Large enough that the drive reaches its target within a step and holds it against the
            # weight it carries, rather than the rate being a compromise between the two.
            float drive:linear:physics:maxForce = 1000000
            uniform token physics:axis = "Z"
            rel physics:body0 = </World/Robot/Mast>
            rel physics:body1 = </World/Robot/Carriage>
            point3f physics:localPos0 = (0, 0, 40)
            point3f physics:localPos1 = (0, 0, 0)
            float physics:lowerLimit = -80
            float physics:upperLimit = 0
        }

        # A bar centered on the spin axis, so its center of mass sits on the axis and spinning it
        # costs no constraint force.
        def Cube "Turret" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"])
        {
            double size = 1
            double3 xformOp:translate = (0, 0, 110)
            float3 xformOp:scale = (60, 8, 8)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            float physxRigidBody:linearDamping = 0
            float physxRigidBody:angularDamping = 0
            float physics:mass = 1
            float3 physics:diagonalInertia = (11, 305, 305)
        }

        # The spin DOF. Its axis is vertical, so gravity exerts no torque about it and it is free:
        # given an initial rate it keeps that rate, and the joint measures no effort.
        def PhysicsRevoluteJoint "spin"
        {
            uniform token physics:axis = "Z"
            rel physics:body0 = </World/Robot/Carriage>
            rel physics:body1 = </World/Robot/Turret>
            point3f physics:localPos0 = (0, 0, 10)
            point3f physics:localPos1 = (0, 0, 0)
        }
    }
}
"""

ROBOT_PATH = "/World/Robot"
ENGINE = "ovphysx"

# Each DOF moves at a rate that is fixed by construction, so the readings have a closed form to be
# checked against: the lift is driven at a constant rate, and nothing exerts a torque on the spin.
LIFT_DOF = "lift"
SPIN_DOF = "spin"

TIME_STEP = 1.0 / 60.0
STEP_COUNT = 120
# The spin DOF is given a rate directly, in the engine's radians per second. Gravity exerts no
# torque about its vertical axis, so it keeps that rate.
INITIAL_SPIN_RATE = 3.0

# Authored in `SCENE`, restated here in SI so the checks below do not read them back from the
# same place the simulation does.
GRAVITY = 9.81
#: Metres per second the lift drive lowers the carriage at, from `SCENE`'s 50 cm/s target.
LIFT_RATE = 0.50
#: Metres from the lift DOF's zero to its lower limit, where the carriage comes to rest.
LIFT_TRAVEL = 0.80
#: Kilograms the lift joint holds once the carriage is resting on that limit: carriage plus turret.
CARRIED_MASS = 3.0

# Rolling average window, equivalent to the shipped sensor's per-step reading being smoothed.
FILTER_WIDTH = 5

# The carriage covers `LIFT_TRAVEL` at `LIFT_RATE` in 1.6 s, so it reaches the limit around step 96
# and rests for what is left of the run. `DESCENT_STEP` is late enough for the sample buffer to be
# full and early enough that the whole filter window, and the window before it, are still moving.
DESCENT_STEP = 40
DESCENT_STEPS = range(10, 80)

# Tolerances for the reported checks, in m/s^2, rad/s^2, m, rad, m/s, rad/s, and N.
ACCELERATION_TOLERANCE = 0.05
POSITION_TOLERANCE = 0.002
VELOCITY_TOLERANCE = 0.005
EFFORT_TOLERANCE = 0.05


def _read_robot_metadata() -> tuple[list[Dof], float, float]:
    """Read the robot's DOF metadata and stage units, before any simulation exists.

    The simulation stage is built from the same text separately, so this one is opened only to be
    asked about the articulation and closed again.

    Returns:
        The articulation's DOFs in order, and the stage's metres and kilograms per unit.

    """
    stage = Stage("openusd").import_stage_from_string(SCENE)
    try:
        # USD names the two DOF kinds exactly as `joint_state` does, so nothing is translated here.
        # A caller reading `dof_types` from the shipped C++ sensor gets 0 and 1 instead and converts
        # them with `joint_state.dof_type_from_code()`.
        articulation = Articulation(ROBOT_PATH)
        dofs = [Dof(name, dof_type) for name, dof_type in zip(articulation.dof_names, articulation.dof_types)]
        meters_per_unit, kilograms_per_unit = stage.get_units()
        return dofs, meters_per_unit, kilograms_per_unit
    finally:
        stage.close_stage()


def _read_articulation_state(robot: ArticulationEntity) -> ArticulationState:
    """Read the raw DOF state of a single articulation.

    Args:
        robot: Articulation entity wrapping exactly one prim.

    Returns:
        The state, in stage units and articulation order, exactly as the engine reports it.

    """
    return ArticulationState(
        positions=robot.get_dof_positions()[0].list(),
        velocities=robot.get_dof_velocities()[0].list(),
        # `get_dof_projected_joint_forces()`, not `get_dof_efforts()`: the shipped sensor's
        # `efforts` field is the measured joint force projected onto the DOF axis, while the
        # entity's `get_dof_efforts()` is the actuation force a drive commands, which is zero
        # throughout here because neither joint is driven. The OvPhysX backend reports the measured
        # force; the Newton backend does not, and a caller there passes `efforts=None` instead.
        efforts=robot.get_dof_projected_joint_forces()[0].list(),
    )


def _print_reading(label: str, reading: JointStateReading) -> None:
    """Print one reading, one line per DOF.

    Args:
        label: Phase name shown above the reading.
        reading: Reading to print.

    """
    print(f"{label} at t = {reading.time:.3f} s:")
    units = {"rotation": ("rad", "rad/s", "rad/s^2", "N*m"), "translation": ("m", "m/s", "m/s^2", "N")}
    for index, name in enumerate(reading.names):
        position, velocity, acceleration, effort = units[reading.types[index]]
        print(
            f"  {name:<6} ({reading.types[index]:<11})"
            f"  position {reading.positions[index]: .4f} {position:<8}"
            f"  velocity {reading.velocities[index]: .4f} {velocity:<8}"
            f"  acceleration {reading.accelerations[index]: .4f} {acceleration:<8}"
            f"  effort {reading.efforts[index]: .4f} {effort}"
        )


def _check_dof_metadata(sensor: JointStateSensor, names: Sequence[str], types: Sequence[str]) -> None:
    """Check that the sensor reports the articulation's DOFs, in order.

    Args:
        sensor: Sensor reporting every DOF.
        names: The articulation's DOF names, from USD.
        types: The articulation's DOF types, from USD.

    Raises:
        RuntimeError: If the names or types do not match.

    """
    expected_names = tuple(names)
    expected_types = tuple(types)
    if sensor.names != expected_names or sensor.types != expected_types:
        raise RuntimeError(
            f"The sensor reports {sensor.names} of types {sensor.types}, "
            f"expected {expected_names} of types {expected_types}."
        )
    described = ", ".join(f"{name} ({dof_type})" for name, dof_type in zip(expected_names, expected_types))
    print(f"DOF names and types match the articulation: {described}.")


def _check_descent(reading: JointStateReading) -> None:
    """Check the reading taken while the carriage is being lowered, and print the outcome.

    Args:
        reading: Reading captured while the lift drive is lowering the carriage.

    Raises:
        RuntimeError: If the reading does not match the rates the scene fixes.

    """
    lift = reading.names.index(LIFT_DOF)
    spin = reading.names.index(SPIN_DOF)

    # The drive holds the carriage at its target rate, so the translation DOF reads that rate. It
    # only reads -0.5 rather than -50 because the centimetre stage unit was applied.
    if abs(reading.velocities[lift] + LIFT_RATE) > VELOCITY_TOLERANCE:
        raise RuntimeError(f"Lift velocity {reading.velocities[lift]:.4f} m/s is not -{LIFT_RATE}.")
    print("Descending lift velocity is the drive rate, in metres per second.")

    # A constant rate has no acceleration. This is a weak value but not a free one: it is the only
    # place a simulation exercises the finite difference at all, and a difference taken over the
    # wrong interval, or against the wrong window, would not land on zero.
    if abs(reading.accelerations[lift]) > ACCELERATION_TOLERANCE:
        raise RuntimeError(f"Descending lift acceleration {reading.accelerations[lift]:.4f} m/s^2 is not zero.")
    print("Descending lift acceleration is zero at a constant rate.")

    # Gravity has no torque about a vertical axis, so the rotation DOF holds the rate it was given.
    # A rotation DOF is radians in any stage, so this value must not pick up the linear unit.
    if abs(reading.velocities[spin] - INITIAL_SPIN_RATE) > VELOCITY_TOLERANCE:
        raise RuntimeError(f"Spin velocity {reading.velocities[spin]:.4f} rad/s is not {INITIAL_SPIN_RATE}.")
    print("Free spin DOF holds its rate in radians per second.")

    # Positions come from the newest sample rather than the filter window, so this is exact rather
    # than lagging by half a window.
    expected_angle = INITIAL_SPIN_RATE * reading.time
    if abs(reading.positions[spin] - expected_angle) > POSITION_TOLERANCE:
        raise RuntimeError(f"Spin position {reading.positions[spin]:.4f} rad is not {expected_angle:.4f}.")
    print("Free spin DOF position advances at its rate.")

    # Nothing resists the turret, so its joint measures nothing, however fast it turns. The lift
    # beside it is carrying a load the whole time, which is the contrast the effort field is for.
    if abs(reading.efforts[spin]) > EFFORT_TOLERANCE:
        raise RuntimeError(f"Spin effort {reading.efforts[spin]:.4f} N*m is not near zero.")
    print("Free spin DOF measures no effort while the lift beside it carries a load.")


def _check_resting(reading: JointStateReading) -> None:
    """Check the settled reading and print the outcome.

    Args:
        reading: Reading captured after the carriage has landed on the lift joint's lower limit.

    Raises:
        RuntimeError: If the reading does not match rest.

    """
    lift = reading.names.index(LIFT_DOF)

    if abs(reading.positions[lift] + LIFT_TRAVEL) > POSITION_TOLERANCE:
        raise RuntimeError(f"Resting lift position {reading.positions[lift]:.4f} m is not -{LIFT_TRAVEL}.")
    if abs(reading.velocities[lift]) > VELOCITY_TOLERANCE:
        raise RuntimeError(f"Resting lift velocity {reading.velocities[lift]:.4f} m/s is not zero.")
    print("Resting lift position matches the joint limit, in metres.")

    # The joint still holds the carriage and the turret against gravity, now against its limit
    # rather than through its drive. In stage units this is 2943 kg*cm/s^2, which is only newtons
    # after the mass and length units are both applied.
    expected_effort = CARRIED_MASS * GRAVITY
    if abs(reading.efforts[lift] - expected_effort) > EFFORT_TOLERANCE:
        raise RuntimeError(f"Resting lift effort {reading.efforts[lift]:.4f} N is not {expected_effort:.4f}.")
    print(f"Resting lift effort matches the {CARRIED_MASS:.0f} kg it carries: {reading.efforts[lift]:.3f} N.")


def _check_selection(reading: JointStateReading, selected: JointStateReading) -> None:
    """Check that selecting DOFs by name reorders a reading without changing it.

    Args:
        reading: Reading over every DOF, in articulation order.
        selected: Reading over the same DOFs, requested in the reverse order.

    Raises:
        RuntimeError: If the selection did not reorder, or changed a value.

    """
    if selected.names != tuple(reversed(reading.names)):
        raise RuntimeError(f"The selected reading reports {selected.names}, expected {tuple(reversed(reading.names))}.")
    for index, name in enumerate(selected.names):
        source = reading.names.index(name)
        if selected.positions[index] != reading.positions[source]:
            raise RuntimeError(f"Selecting {name!r} changed its position.")
    print(f"Selecting DOFs by name reorders the reading: {', '.join(selected.names)}.")


def _parse_arguments() -> argparse.Namespace:
    """Parse the example's command line.

    Returns:
        Parsed arguments.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the unit tests for joint_state.py and exit, without starting a simulation.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Show a live window with the robot and its joint state side by side. Needs the "
        "optional 'ovui' and 'glfw' packages.",
    )
    return parser.parse_args()


def _run_batch(
    manager: PhysicsManager,
    robot: ArticulationEntity,
    sensor: JointStateSensor,
    selection: JointStateSensor,
    dof_names: Sequence[str],
    dof_types: Sequence[str],
    meters_per_unit: float,
) -> None:
    """Simulate to completion, then report and check the readings.

    Args:
        manager: Physics manager driving the simulation.
        robot: Articulation entity wrapping exactly one prim.
        sensor: Sensor reporting every DOF.
        selection: Sensor reporting the same DOFs in the reverse order.
        dof_names: DOF names in articulation order, which is how the raw buffers are indexed.
        dof_types: DOF types in the same order.
        meters_per_unit: Stage metres per linear unit.

    Raises:
        RuntimeError: If any reported check does not hold.

    """
    print(f"Joint state of the articulation at {ROBOT_PATH}.")
    print("Reporting measured joint effort, not commanded actuation.")
    print(f"The stage is authored in centimetres, {meters_per_unit:.4g} m per unit.")
    _check_dof_metadata(sensor, dof_names, dof_types)

    descent_reading: JointStateReading | None = None
    descent_selection: JointStateReading | None = None
    raw_gap = 0.0
    converted_gap = 0.0
    spin_gap = 0.0
    cross_checked = False
    # Indices into the raw engine buffers, which are always in articulation order whatever order a
    # sensor was asked to report.
    lift = list(dof_names).index(LIFT_DOF)
    spin = list(dof_names).index(SPIN_DOF)

    for step in range(STEP_COUNT + 1):
        manager.step()
        # The buffers hold the state after the step, so stamp it with the time stepped to, which
        # the manager has accumulated for us.
        time = manager.get_simulated_time()
        state = _read_articulation_state(robot)
        sensor.sample(time, state)
        selection.sample(time, state)

        if step in DESCENT_STEPS:
            # Independent of joint_state.py: read straight off the engine buffers, the descending
            # lift DOF must report the drive's rate expressed in stage units, and only the linear
            # unit turns that into SI. Getting the conversion backwards, or applying it twice,
            # fails one of these two.
            raw = state.velocities[lift]
            raw_gap = max(raw_gap, abs(raw + LIFT_RATE / meters_per_unit))
            converted_gap = max(converted_gap, abs(raw * meters_per_unit + LIFT_RATE))
            # The rotation DOF beside it must not pick up the linear unit at all.
            spin_gap = max(spin_gap, abs(state.velocities[spin] - INITIAL_SPIN_RATE))
            cross_checked = True

        if step == DESCENT_STEP:
            descent_reading = sensor.read()
            descent_selection = selection.read()

    resting_reading = sensor.read()
    if descent_reading is None or descent_selection is None or resting_reading is None:
        raise RuntimeError("The sensor did not buffer enough samples to produce a reading.")

    _print_reading("Descending", descent_reading)
    _print_reading("At rest", resting_reading)

    # Each sample is stamped with the time the step advanced to, so a reading carries the time of
    # its newest sample. Nothing else here would notice the whole series shifted by a step.
    for label, reading, step in (
        ("descending", descent_reading, DESCENT_STEP),
        ("resting", resting_reading, STEP_COUNT),
    ):
        expected_time = (step + 1) * TIME_STEP
        if abs(reading.time - expected_time) > 0.5 * TIME_STEP:
            raise RuntimeError(f"The {label} reading is stamped {reading.time:.3f} s, expected {expected_time:.3f} s.")
    print("Readings are stamped with the time their newest sample was taken.")

    if not cross_checked:
        raise RuntimeError("The descending velocity cross-check did not run.")
    print(f"Largest raw descending lift velocity gap: {raw_gap:.3f} stage units/s.")
    if raw_gap > VELOCITY_TOLERANCE / meters_per_unit:
        raise RuntimeError("The raw descending lift velocity does not match the drive rate in stage units.")
    print(f"Largest converted descending lift velocity gap: {converted_gap:.4f} m/s.")
    if converted_gap > VELOCITY_TOLERANCE:
        raise RuntimeError("The converted descending lift velocity does not match the drive rate in SI units.")
    print("Descending lift velocity matches the drive rate in stage units and in SI units.")
    if spin_gap > VELOCITY_TOLERANCE:
        raise RuntimeError(f"The spin rate drifted by {spin_gap:.4f} rad/s, so radians picked up the linear unit.")

    if not descent_reading.efforts_available:
        raise RuntimeError("OvPhysX reports measured joint efforts, but the reading says otherwise.")
    _check_descent(descent_reading)
    _check_resting(resting_reading)
    _check_selection(descent_reading, descent_selection)


def main() -> None:
    """Simulate a two-DOF robot and report the joint state of both of its degrees of freedom."""
    arguments = _parse_arguments()

    if arguments.self_test:
        # The tests need no simulation, so run them before any engine or stage is created.
        import test_joint_state

        raise SystemExit(1 if test_joint_state.run_all() else 0)

    if not activate():
        raise RuntimeError("OvPhysX physics engine is unavailable.")

    ovstage_stage = None
    simulation_initialized = False
    try:
        wp.init()
        dofs, meters_per_unit, kilograms_per_unit = _read_robot_metadata()
        ovstage_stage = Stage("ovstage").import_stage_from_string(SCENE, make_default=False)
        # Everything the simulation needs goes through the manager. The module-level `initialize`,
        # `simulate` and `close` functions do the same job but are on their way out of the public
        # API, so this drives the manager instance directly.
        manager = PhysicsManager.get_instance()
        if not manager.switch_physics_engine(ENGINE):
            raise RuntimeError("OvPhysX physics engine is unavailable.")
        set_suppress_readback(False)

        # The step size is configured once here rather than passed to every step.
        manager.setup(dt=TIME_STEP)
        if not manager.initialize(ovstage_stage.get_stage_ptr(), ovstage_stage.get_stage_id()):
            raise RuntimeError("Physics initialization failed.")
        simulation_initialized = True
        robot = ArticulationEntity(ENGINE, ROBOT_PATH)

        sensor = JointStateSensor(
            dofs,
            meters_per_unit=meters_per_unit,
            kilograms_per_unit=kilograms_per_unit,
            filter_width=FILTER_WIDTH,
        )

        def _set_initial_state() -> None:
            """Return the DOFs to their start state: carriage at the top, turret already spinning.

            Both arrays are in engine units and in articulation order, so the spin rate goes in as
            radians per second and the lift position as centimetres.

            A DOF write has to be a float32 array on the host. `RigidBodyEntity.set_velocities()`
            converts a plain nested list, but the DOF setters do not: a list reaches the backend as
            float64 and the write fails with `ovphysx_write_tensor_binding failed` rather than a
            type error.
            """
            positions = [0.0] * len(dofs)
            velocities = [0.0] * len(dofs)
            velocities[[dof.name for dof in dofs].index(SPIN_DOF)] = INITIAL_SPIN_RATE
            robot.set_dof_positions(wp.array([positions], dtype=wp.float32, device="cpu"))
            robot.set_dof_velocities(wp.array([velocities], dtype=wp.float32, device="cpu"))

        _set_initial_state()

        if arguments.ui:

            def _advance(step: int) -> None:
                """Step the simulation once and sample the sensor.

                Args:
                    step: Current simulation step.
                """
                del step  # The manager keeps the step count and the simulated time.
                manager.step()
                sensor.sample(manager.get_simulated_time(), _read_articulation_state(robot))

            def _restart() -> None:
                """Return the robot to its start state and discard buffered samples.

                The jump makes any difference across it meaningless, so the buffer is cleared.
                """
                _set_initial_state()
                sensor.reset()

            last_reading = viewer.run(
                sensor=sensor,
                advance=_advance,
                restart=_restart,
                robot_path=ROBOT_PATH,
                lift_dof=LIFT_DOF,
                spin_dof=SPIN_DOF,
                lift_travel=LIFT_TRAVEL,
                step_count=STEP_COUNT,
            )
            if last_reading is not None:
                _print_reading("Last reading before exit", last_reading)
        else:
            # The same DOFs, requested in the reverse order. A consumer with a fixed message
            # layout, such as a ROS 2 `sensor_msgs/JointState` publisher, selects like this.
            selection = JointStateSensor(
                dofs,
                meters_per_unit=meters_per_unit,
                kilograms_per_unit=kilograms_per_unit,
                names=[dof.name for dof in reversed(dofs)],
                filter_width=FILTER_WIDTH,
            )
            _run_batch(
                manager,
                robot,
                sensor,
                selection,
                [dof.name for dof in dofs],
                [dof.type for dof in dofs],
                meters_per_unit,
            )
    finally:
        if simulation_initialized:
            PhysicsManager.get_instance().invalidate()
        if ovstage_stage is not None:
            ovstage_stage.close_stage()
        shutdown()


if __name__ == "__main__":
    main()
