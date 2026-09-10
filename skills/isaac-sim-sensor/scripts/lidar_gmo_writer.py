"""RTX Lidar GMO (GenericModelOutput) writer pattern for Isaac Sim.

Demonstrates the Writer-based approach for consuming lidar scan data
without frame-drop under multitick.  Run with $ISAAC_SIM_DIR/python.sh.
"""


def create_lidar_with_gmo_writer(
    path: str = "/World/lidar",
    config: str = "OS1",
    variant: str = "OS1_REV6_32ch20hz512res",
    simulation_app=None,
) -> None:
    """Create an RTX lidar and attach a GMO-inspection writer.

    Args:
        path: USD prim path for the new lidar.
        config: Lidar config name from SUPPORTED_LIDAR_CONFIGS.
        variant: Variant selection for the lidar asset.
        simulation_app: Running SimulationApp instance for the update loop.
    """
    import omni.replicator.core as rep
    from isaacsim.sensors.experimental.rtx import (
        Lidar,
        LidarSensor,
        parse_generic_model_output_data,
    )
    from omni.replicator.core import Writer

    lidar = Lidar.create(
        path=path,
        config=config,
        variant=variant,
        aux_output_level="FULL",  # NONE | BASIC | EXTRA | FULL
        tick_rate=20.0,  # Hz; None preserves authored value
        accumulate_outputs=True,  # True = full scan per output
    )

    # Sensor brings no annotators — the writer brings its own GMO annotator.
    sensor = LidarSensor(lidar, annotators=[])

    class GmoInspectWriter(Writer):
        def __init__(self):
            self.data_structure = "renderProduct"
            self.annotators = [rep.annotators.get("GenericModelOutput")]

        def write(self, data):
            if "renderProducts" not in data:
                return
            for _rp, rp_data in data["renderProducts"].items():
                raw = rp_data.get("GenericModelOutput")
                if isinstance(raw, dict):
                    raw = raw.get("data")
                gmo = parse_generic_model_output_data(raw)
                if gmo.numElements > 0:
                    pass  # use gmo.x / gmo.y / gmo.z / gmo.scalar / gmo.matId / ...

    rep.WriterRegistry.register(GmoInspectWriter)
    sensor.attach_writer("GmoInspectWriter")

    import isaacsim.core.experimental.utils.app as app_utils

    app_utils.play(commit=True)
    if simulation_app is not None:
        while simulation_app.is_running():
            simulation_app.update()
