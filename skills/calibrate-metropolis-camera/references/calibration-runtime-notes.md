# Calibration: runtime notes

Output schema, ordering rules, and verification for `calibrate-metropolis-camera`.
Parameter semantics live in
[`references/calibration-api.md`](calibration-api.md).

---

## What gets written

Into `camera_calibration_output_folder_path` (must be a local path — Nucleus and
other URIs are rejected). Sizes below are from a real run over 5 cameras on a
warehouse stage:

| File | Size | Contents |
|---|---|---|
| `calibration.json` | ~100 KB | One `sensors[]` entry per camera |
| `Top.png` | ~750 KB | Orthographic plan view at the requested resolution |
| `imageMetadata.json` | ~150 B | Describes `Top.png` |
| `Debug/` | varies | Only when the debug/FOV-polygon options are enabled |

### `calibration.json`

Metropolis calibration schema:

```json
{
  "version": "...",
  "osmURL": "...",
  "calibrationType": "...",
  "sensors": [
    {
      "type": "camera",
      "id": "Camera",
      "origin": {"lng": 0, "lat": 0},
      "coordinates": {"x": -9.86, "y": -7.84},
      "translationToGlobalCoordinates": {"x": 34.52, "y": 18.04},
      "scaleFactor": 1.0,
      "place": "room=warehouse",
      "attributes": [{"name": "fps"}, {"name": "fieldOfView"}, {"name": "direction3d"}],
      "imageCoordinates": [],
      "globalCoordinates": [],
      "intrinsicMatrix": [],
      "extrinsicMatrix": [],
      "cameraMatrix": [],
      "homography": [],
      "tripwires": [],
      "rois": []
    }
  ]
}
```

`id` is the camera prim name, so it is only unique within one parent prim.
Calibrating `/World/Cameras` and `/World/Cameras/AimAt` separately produces two
files that both contain a sensor called `Camera`.

`attributes` carries `fps`, `depth`, `fieldOfView`, `direction`, `direction3d`,
`source`, `frameWidth`, `frameHeight`.

### `imageMetadata.json`

```json
{"images": [{"place": "room=warehouse", "view": "plan-view", "fileName": "Top.png"}]}
```

`place` is the `place_info` string verbatim — which is why an unhelpful
`place_info` propagates into every downstream consumer.

---

## Ordering and async rules

The four stages are strictly sequential and each validates the previous one's
output. The failure mode when you get this wrong is silent: a fire-and-forget
sync call returns immediately, the next stage finds nothing, and the run "works"
while writing an empty or missing file.

```python
from isaacsim.sensors.rtx.calibration import CameraCalibrationManager

manager = CameraCalibrationManager.get_instance()

manager.create_top_view_camera(root_prim_path="/World")     # sync, sets top_view_camera_path
await manager.generate_calibration_dot_prim_async()         # must finish before stage 3
if manager.is_ready_to_calibrate() is False:                # NOT `if ...:` -- returns None on success
    raise RuntimeError("not ready; see the carb log")
await manager.generate_calibration_async()                  # writes calibration.json
await manager.capture_and_process_camera_view_image()       # writes Top.png
```

python_server payloads run inside an async scope, so top-level `await` works as
written. In a `SimulationApp` script, drive the coroutines with the app's event
loop instead.

---

## Verifying the result

The script checks that the files exist and are non-empty; that is necessary, not
sufficient. Two failure modes produce plausible-looking output:

**A black `Top.png`** means the orthographic camera framed empty space — usually
a `scene_root` whose bbox misses the facility. Every homography derives from that
view, so the whole file is meaningless. Check it:

```python
from PIL import Image

image = Image.open("/tmp/calib/Top.png")
extrema = image.convert("RGB").getextrema()
print("size:", image.size, "extrema:", extrema)
if all(low == high == 0 for low, high in extrema):
    raise RuntimeError("Top.png is black; scene_root did not frame the facility")
```

**A sensor count below the camera count** means cameras were dropped. Compare
against the stage:

```python
import json

from pxr import UsdGeom

import omni.usd

data = json.load(open("/tmp/calib/calibration.json"))
stage = omni.usd.get_context().get_stage()
parent = stage.GetPrimAtPath("/World/Cameras")
placed = [c for c in parent.GetChildren() if c.IsA(UsdGeom.Camera)]
print(f"sensors={len(data['sensors'])} cameras={len(placed)}")
assert len(data["sensors"]) == len(placed), "cameras were dropped during calibration"
```

A mismatch nearly always means a camera fell outside the top-view frustum;
`generate_calibration_async()` logs `camera <path> is not covered by top view!`
and stops.

---

## Interaction with placement

Calibration is a snapshot of camera poses at the moment it ran. Anything that
moves a camera invalidates the file:

- re-running `place-camera-max-coverage` or `place-camera-aim-at`
- `CircularCameraPlacement.refresh_camera_prim()`, which deletes the whole camera parent subtree
- editing transforms by hand

Re-run calibration afterwards. The two extensions share
`GeneralSetting.camera_parent_prim_path`, so pointing placement at a new parent
also changes what calibration picks up by default.

To calibrate several camera groups, run once per parent with a distinct
`output_dir` — `id` collides across groups otherwise:

```python
groups = {"/World/Cameras": "/tmp/calib/coverage", "/World/Cameras/AimAt": "/tmp/calib/aimat"}
for parent_path, out in groups.items():
    print(f"calibrate {parent_path} -> {out}")
```

---

## Reading the console

Failures are logged through `carb`, and the abort reasons only appear there —
the API returns normally either way. Raise the channel before a debugging run:

```python
import carb

carb.settings.get_settings().set(
    "/log/channels/isaacsim.sensors.rtx.calibration",
    "info",
)
```

The lines worth grepping for: `is not covered by top view`, `The calibration dot
has not yet been generated`, `Top camera is either not generated or in wrong
projection type`, and `please provide valid camera path and folder path`.
