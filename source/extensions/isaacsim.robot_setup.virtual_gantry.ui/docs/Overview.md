# Virtual Gantry UI

UI for the [Virtual Gantry](../../isaacsim.robot_setup.virtual_gantry/docs/Overview.md)
extension. Everything here only drives the backend; all rope physics lives in
`isaacsim.robot_setup.virtual_gantry`, which has no UI dependencies and can be
enabled on its own.

Provides:

- **Create > Robotics > Virtual Gantry** — authors an `IsaacVirtualGantry` prim,
  placed above the selected prim by default.
- The **property panel** shown when an `IsaacVirtualGantry` prim is selected:
  the `isaac:gantry:*` attributes, the attach-body / articulation
  relationships, and Enable / Disable buttons.
- Keyboard **hotkeys**: `G` toggles every gantry, `[` / `]` shorten / lengthen
  the rope by 5 mm. The viewport must have keyboard focus.

The hotkeys reach the running ropes through
`isaacsim.robot_setup.virtual_gantry.get_manager()`, resolved on each press so
this extension tolerates starting before the backend. `[` / `]` adjust the live
rope without writing the prim, which is why they need the manager rather than a
USD edit like the Enable / Disable buttons.
