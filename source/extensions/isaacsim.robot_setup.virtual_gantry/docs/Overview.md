# Virtual Gantry

The Virtual Gantry suspends a robot from an anchor point during bring-up and
policy testing — like a crane holding the robot so it does not fall over while
you iterate on locomotion or system identification.

It is a *runtime constraint only*: a one-sided spring-damper "rope" applied as a
per-physics-step external wrench to a single articulation body. There is no USD
joint and no kinematic anchor; all six base DOFs stay free. The rope applies no
force until the body drops past the rope length, then it pulls the body back
toward the anchor.

## Usage

1. **Create > Robotics > Virtual Gantry** authors an `IsaacVirtualGantry` prim
   (an `Xform` whose world position is the anchor), placed above the selected
   prim by default. You can also script it with
   `isaacsim.robot_setup.virtual_gantry.create_virtual_gantry(stage, path)`.
2. Set the prim's **Attach Body** relationship to the link the rope pulls on
   (e.g. the torso) and the **Articulation** relationship to the robot's
   articulation root. Move the prim to place the anchor.
3. Tune **Rope Length**, **Stiffness**, **Damping** in the property panel; a
   rope length of `-1` auto-derives from the body's height on the first step.
4. Press **Play**. Toggle the rope with **Enable / Disable** in the property
   panel or the **`G`** hotkey; shorten / lengthen it with **`[`** / **`]`** (the
   viewport must have keyboard focus). Disable it once a policy is balancing.

## Notes

- Backend: PhysX and Newton. The rope reads the attach link's pose and applies
  its wrench through the articulation tensor view (`get_link_transforms` /
  `apply_forces_and_torques_at_position`); that view is engine-agnostic, so no
  backend-specific code is needed. Verified in a local build on both
  `isaac-sim.sh` and `isaac-sim.newton.sh`.
- The anchor uses the prim's position only; orientation is reserved for a future
  6-DOF "soft fixture" mode.
- The force law matches the established MuJoCo virtual-gantry behavior, so a
  policy behaves the same whether suspended in MuJoCo or Isaac Sim.
