# Overview

The isaacsim.gui.menu extension provides Isaac Sim-specific menu bar functionality for robotics applications. It creates and manages multiple specialized menus including File, Edit, Create, Window, Tools, Utilities, Layouts, and Help menus, each tailored with robotics-focused operations and Isaac Sim assets.

## Key Components

### Menu Extensions

The Create menu implementation builds and manages robotics asset entries and specialized creation actions. It includes utilities for creating Isaac Sim assets at specified stage paths and generating AprilTag materials with custom textures.

The Edit menu implementation handles robotics-focused prim manipulation, including selection validation, duplication with layer options, parenting operations, grouping/ungrouping, and visibility controls. The extension also includes screenshot capture functionality and live session detection for collaborative workflows.

The File menu implementation manages Isaac Sim-specific file operations. It includes validation methods for stage operations and uses a custom delegate to control text width limits for menu items like "Open Recent".

The FixMe menu implementation manages a specialized placeholder menu using a custom delegate that hides menu entries and provides right-aligned menu positioning.

The Window, Tools, Utilities, Layouts, and Help menu implementations handle their respective menu categories. The Help menu includes functionality to open documentation URLs using the system browser and provides physics documentation URL resolution for version-specific links.

### Menu Infrastructure

The extension uses **omni.kit.menu.utils** for menu item creation and management, integrating with the broader Omniverse Kit menu framework through MenuItemDescription objects and standard menu registration patterns.

### Asset Creation Functions

The Create menu implementation includes specialized asset creation actions for referencing USD assets with optional camera positioning and generating AprilTag materials with custom tag textures.

## Functionality

Each menu extension is responsible for registering its menu items, managing menu state, and handling menu-specific actions. The extensions provide validation methods to determine when menu items should be enabled or disabled based on current stage state, selection, and live session status.

The Edit menu includes advanced prim operations like instancing, duplication across layers, parenting with validation, and screenshot capture with configurable save paths. The Create menu focuses on robotics asset creation and material generation specific to Isaac Sim workflows.

All menu extensions follow a consistent lifecycle pattern with initialization that takes an extension ID and a shutdown method that removes menu layouts and cleans up resources.
