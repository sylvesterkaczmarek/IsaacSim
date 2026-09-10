Python API
==========

.. warning::

    **The API featured in this extension is experimental and subject to change without deprecation cycles.**
    Although we will try to maintain backward compatibility in the event of a change, it may not always be possible.

.. Summary

The following table summarizes the available classes.

.. currentmodule:: isaacsim.sensors.experimental.rtx

.. rubric:: authoring (USD prim wrappers)
.. autosummary::
    :nosignatures:

    Lidar
    Radar
    Acoustic
    RtxCamera
    StructuredLightCamera

.. rubric:: sensors (runtime)
.. autosummary::
    :nosignatures:

    LidarSensor
    RadarSensor
    AcousticSensor
    CameraSensor
    SingleViewDepthCameraSensor
    TiledCameraSensor

.. rubric:: sensor processing graphs (SPG)
.. autosummary::
    :nosignatures:

    SPGNode

.. rubric:: utils
.. autosummary::
    :nosignatures:

    parse_generic_model_output_data
    parse_stable_id_map_data

.. rubric:: lidar configuration

- :data:`SUPPORTED_LIDAR_CONFIGS`
- :data:`SUPPORTED_LIDAR_VARIANT_SET_NAME`

Authoring
---------

.. autoclass:: isaacsim.sensors.experimental.rtx.Lidar
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: isaacsim.sensors.experimental.rtx.Radar
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: isaacsim.sensors.experimental.rtx.Acoustic
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: isaacsim.sensors.experimental.rtx.RtxCamera
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: isaacsim.sensors.experimental.rtx.StructuredLightCamera
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Sensors
-------

.. autoclass:: isaacsim.sensors.experimental.rtx.LidarSensor
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: isaacsim.sensors.experimental.rtx.RadarSensor
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: isaacsim.sensors.experimental.rtx.AcousticSensor
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: isaacsim.sensors.experimental.rtx.CameraSensor
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: isaacsim.sensors.experimental.rtx.SingleViewDepthCameraSensor
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: isaacsim.sensors.experimental.rtx.TiledCameraSensor
    :members:
    :undoc-members:
    :show-inheritance:

Sensor processing graphs (SPG)
------------------------------

Author custom GPU post-processing passes on RTX sensor outputs with
:meth:`RtxCamera.author_spg <isaacsim.sensors.experimental.rtx.RtxCamera.author_spg>`
(inherited from the authoring base class) and :class:`SPGNode`. See the
``omni.rtx.spg`` documentation for the underlying framework:
https://docs.omniverse.nvidia.com/kit/docs/omni.rtx.spg/0.2.0/Overview.html

.. autoclass:: isaacsim.sensors.experimental.rtx.SPGNode
    :members:
    :undoc-members:

Utils
-----

.. autofunction:: isaacsim.sensors.experimental.rtx.parse_generic_model_output_data

.. autofunction:: isaacsim.sensors.experimental.rtx.parse_stable_id_map_data

Lidar configuration registry
----------------------------

.. py:data:: SUPPORTED_LIDAR_CONFIGS

   Mapping from known Isaac Sim lidar asset paths to optional variant name sets.

.. py:data:: SUPPORTED_LIDAR_VARIANT_SET_NAME

   Variant set name expected on supported lidar prims.
