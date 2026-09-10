```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Settings

## app.sensors.nv.lidar.outputBufferOnGPU
   - **Default Value**: false
   - **Description**: Controls whether the renderer keeps the Lidar return buffer on the GPU for post-processing.

## app.sensors.nv.radar.outputBufferOnGPU
   - **Default Value**: false
   - **Description**: Controls whether the renderer keeps the Radar return buffer on the GPU for post-processing.

## rtx.materialDb.nonVisualMaterialCSV.enabled
   - **Default Value**: false
   - **Description**: Enables non-visual materials using USD attributes.

## rtx.materialDb.nonVisualMaterialSemantics.prefix
   - **Default Value**: "omni:simready:nonvisual"
   - **Description**: Specifies the non-visual material USD attribute prefix.

## rtx.rtxsensor.useHydraTimeAlways
   - **Default Value**: true
   - **Description**: Controls whether RTX sensor models use Hydra time (`omni.timeline`) when multi-tick rendering is disabled.
