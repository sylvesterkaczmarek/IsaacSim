"""Camera sensor creation and annotator attachment via the Isaac Sim RTX API.

Creates an RTX camera (``RtxCamera``) and wraps it in a ``CameraSensor`` that
builds the Replicator render product and exposes standard perception
annotators (RGB, depth, segmentation, bbox, normals, motion vectors).

Uses ``isaacsim.sensors.experimental.rtx`` throughout — no direct
``UsdGeom.Camera`` authoring or raw ``omni.replicator.core`` render-product
wiring. Run with ``$ISAAC_SIM_DIR/python.sh``.
"""

DEFAULT_ANNOTATORS = [
    "rgb",
    "distance_to_camera",
    "semantic_segmentation",
    "instance_segmentation",
    "motion_vectors",
    "bounding_box_2d_tight",
    "bounding_box_3d",
    "normals",
]


def create_camera_sensor(
    path: str = "/World/Camera",
    focal_length: float = 24.0,
    resolution: tuple = (720, 1280),
    annotators: list | None = None,
):
    """Create an RTX camera and return a ``CameraSensor`` with annotators attached.

    Args:
        path: USD prim path for the new camera (e.g. "/World/Camera").
        focal_length: Camera focal length in mm.
        resolution: Output resolution as ``(height, width)`` — OpenCV/NumPy
            convention used by ``CameraSensor``.
        annotators: Annotator names to attach. Defaults to the standard
            perception set (RGB, depth, segmentation, bbox, normals, motion
            vectors) in :data:`DEFAULT_ANNOTATORS`.

    Returns:
        ``isaacsim.sensors.experimental.rtx.CameraSensor`` handle. Its render
        product and annotators are already created and attached; call
        ``app_utils.play(commit=True)`` (or step the ``SimulationApp`` /
        timeline) before ``sensor.get_data(name)``.
    """
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

    cam = RtxCamera(path, tick_rate=30.0)
    cam.camera.set_focal_lengths(focal_length)
    cam.camera.set_apertures(horizontal_apertures=36.0, vertical_apertures=20.25)
    cam.camera.set_clipping_ranges(0.01, 100.0)

    return CameraSensor(
        cam,
        resolution=resolution,
        annotators=annotators if annotators is not None else DEFAULT_ANNOTATORS,
    )


def attach_annotators(sensor, annotators: list | None = None) -> dict:
    """Attach additional perception annotators to an existing ``CameraSensor``.

    Thin wrapper over ``CameraSensor.attach_annotators`` for parity with the
    skill docs; prefer passing ``annotators=`` to :func:`create_camera_sensor`
    or calling ``sensor.attach_annotators(...)`` directly.

    Args:
        sensor: A ``CameraSensor`` instance.
        annotators: Annotator names to attach. Defaults to
            :data:`DEFAULT_ANNOTATORS`.

    Returns:
        Dict mapping annotator name to attached annotator handle.
    """
    return sensor.attach_annotators(annotators if annotators is not None else DEFAULT_ANNOTATORS)
