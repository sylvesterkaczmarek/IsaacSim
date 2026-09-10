# Overview

This extension provides UI integration for creating camera and depth sensors in Isaac Sim. It adds menu items to the Create menu and context menus that enable users to create various camera and depth sensor prims from multiple vendors including Orbbec, Leopard Imaging, Luxonis, RealSense, Sensing, SICK, and Stereolabs.

## Functionality

The extension automatically registers sensor creation actions and organizes them into a hierarchical menu structure by vendor. Users can access sensor creation through two main pathways: the main Create menu under "Sensors > Camera and Depth Sensors" and context menus accessible via right-click in the viewport under "Isaac > Sensors".

The full list of supported sensors and their metadata (vendor grouping, display name, depth-sensor flag, default stage prim prefix) is sourced from {data}`isaacsim.sensors.experimental.rtx.SUPPORTED_CAMERA_CONFIGS`, so adding a new vendor sensor is a one-place change in that registry.

**Supported camera sensors include:**
- Leopard Imaging: Hawk, Owl
- Sensing: SG2-AR0233C-5200-G2A-H100F1A, SG2-OX03CC-5200-GMSL2-H60YA, SG3-ISX031C-GMSL2F-H190XA, SG5-IMX490C-5300-GMSL2-H110SA, SG8S-AR0820C-5300-G2A-H120YA, SG8S-AR0820C-5300-G2A-H30YA, SG8S-AR0820C-5300-G2A-H60SA
- SICK: Inspector83x, InspectorP61x

**Supported depth sensors include:**
- Orbbec: Gemini 2, FemtoMega, Gemini 335, Gemini 335L
- Luxonis: OAK4-D, OAK4-D Wide, OAK-D Pro PoE, OAK-D Pro W PoE, OAK-D ToF
- RealSense: D455, D457, D555
- SICK: safeVisionary2, Visionary-T Mini
- Stereolabs: ZED_X

## Key Components

### Sensor Creation Actions

The extension creates specialized actions for each supported sensor type using the action registry. All sensors are loaded via {meth}`isaacsim.sensors.experimental.rtx.RtxCamera.create`. For entries whose registry metadata sets ``is_depth_sensor=True``, every Camera in the loaded asset that has a depth-sensor template render product is additionally wrapped with {class}`isaacsim.sensors.experimental.rtx.SingleViewDepthCameraSensor`, which copies the template's depth-sensor attributes onto the new render product.

### Menu Integration

Menu items are dynamically created and organized by vendor, providing a structured approach to sensor selection. The hierarchical organization helps users locate specific sensor models efficiently within the Isaac Sim interface.

## Integration

The extension uses **omni.kit.actions.core** to register sensor creation actions and **omni.kit.context_menu** to provide right-click access to sensor creation tools. It integrates with `isaacsim.sensors.experimental.rtx` for the underlying sensor implementation ({class}`RtxCamera <isaacsim.sensors.experimental.rtx.RtxCamera>` and {class}`SingleViewDepthCameraSensor <isaacsim.sensors.experimental.rtx.SingleViewDepthCameraSensor>`) and `isaacsim.gui.components` for UI component support.
