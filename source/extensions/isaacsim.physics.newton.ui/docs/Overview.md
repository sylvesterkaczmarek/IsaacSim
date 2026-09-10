# Overview

The isaacsim.physics.newton.ui extension provides UI integration for Newton and MuJoCo physics schemas within Isaac Sim. It registers schema definitions and property widgets that enable users to configure and interact with Newton and MuJoCo physics parameters through the property panel interface.

## Key Components

### Schema Registration

The extension provides functions to retrieve schema type names for both physics systems:

- {func}`get_newton_schema_names <isaacsim.physics.newton.ui.get_newton_schema_names>` returns prim type names and API schema names for Newton physics.
- {func}`get_mujoco_schema_names <isaacsim.physics.newton.ui.get_mujoco_schema_names>` returns prim type names and API schema names for MuJoCo physics.

These functions enable the property system to recognize and handle Newton and MuJoCo-specific schemas.

### UI Definitions

**{class}`NewtonUiDefinitions <isaacsim.physics.newton.ui.NewtonUiDefinitions>`** and **{class}`MujocoUiDefinitions <isaacsim.physics.newton.ui.MujocoUiDefinitions>`** classes contain comprehensive UI configuration for their respective physics systems. Each definition class includes:

- Property widget specifications for displaying physics parameters.
- Property builders for creating custom UI elements.
- Property ordering configurations for consistent layout.
- Extension mappings and filtering rules.

### Applying schemas

Select a compatible prim and use **Add > Physics > Newton** or **Add > Physics > Mujoco** to apply a physics API schema. The menus filter their entries by the selected prim type and hide schemas that are already applied. The MuJoCo menu also applies required Newton API dependencies.

The generic **Edit API Schema** dialog does not list these privately owned physics schemas. See {ref}`isaac_sim_newton_robot_setup_tips_add_apis` for the complete workflow and supported prim types.

### Creating MuJoCo prims

Some MuJoCo schemas (`MjcActuator`, `MjcKeyframe`, `MjcTendon`) are concrete prim types rather than applied API schemas, so they cannot be layered onto an existing prim. Use **Create > Physics > Mujoco** (from the main **Create** menu, the viewport context menu, or the Stage window context menu) to instantiate them. The new prim is parented under the current selection, or under the stage root when nothing is selected.

### Editing numeric arrays

MuJoCo numeric array attributes (for example the `MjcKeyframe` state vectors and the actuator/tendon parameter vectors) are edited through a pop-up window instead of the default read-only string. The **Edit** button next to the attribute opens a window that shows the attribute name and description and lists every value in a vertical, index-ordered list where entries can be added, removed, and changed.

### Newton Scene Widget

**ExtendedNewtonSceneWidget** provides specialized property handling for Newton physics scenes. The widget automatically detects when Newton physics is the active simulation variant and dynamically adds Newton-specific properties like the "newton:solver" selection to the property panel. This allows users to configure solver settings directly through the UI when Newton physics is enabled.

## Functionality

The extension integrates Newton and MuJoCo physics configuration into Isaac Sim's existing property panel system. When users select physics objects in the scene, the appropriate physics-specific properties become available for editing through the standard property interface. The extension handles the registration of custom widgets and ensures proper display ordering of physics parameters.

For Newton physics specifically, the extension provides dynamic property injection based on the active simulation variant, automatically showing relevant Newton solver options when Newton physics is selected for the scene.
