```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Settings

## Settings Provided by the Extension

### exts."isaacsim.physics.newton".capture_graph_physics_step
- **Default Value**: true
- **Description**: Enable CUDA graph capture for physics stepping to improve GPU performance.

### exts."isaacsim.physics.newton".auto_switch_on_startup
- **Default Value**: true
- **Description**: Automatically switch to the Newton physics engine when the extension starts.

### exts."isaacsim.physics.newton".load_textures
- **Default Value**: false
- **Description**: Load material textures into Newton. Off by default: they feed Newton's own viewer only -- Kit renders the stage through Hydra -- and a texture that loads also makes Newton build a second UV-expanded render mesh per textured prim. Measured on a four-conveyor scene, world.reset() was 8.5 s off vs 37.5 s on.
