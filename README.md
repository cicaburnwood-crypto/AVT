# AVT

Any - Video - Trainning - Databuild and process toolkit

AVT is a standalone inverse-video point tracking and WebUI visualization toolkit.
It was extracted so the tracking model, inverse tracking pipeline, and viewer are
separate pieces:

- `avt/tracking/`: point tracker backends.
- `avt/inverse.py`: reversed-video windowing, query seeding, and artifact export.
- `avt/viewer.py`: static WebUI builder that reads generic AVT artifacts.

There are no runtime imports, symlinks, or path assumptions from VENTURA or any
other local project.

## Install

```bash
python -m pip install -e .
```

The default `lk` backend only needs OpenCV and NumPy. The `cotracker` backend
requires PyTorch and a CoTracker hub-capable environment.

## Run Inverse Tracking

Run the whole pipeline on an image folder, Earth Rover recording folder with
`frames.jsonl`, or an OpenLORIS-style folder with `color.txt`:

```bash
avt all \
  --frames-root /path/to/recording_or_images \
  --source-type auto \
  --backend cotracker \
  --cotracker-device cuda \
  --query-mode avt+sift \
  --robot-config configs/virtual_robot.yaml \
  --window-size 250 \
  --window-step 100 \
  --query-stride 10 \
  --fps 10
```

For a quick CPU smoke test without CoTracker:

```bash
avt all \
  --frames-root /path/to/images \
  --source-type image_dir \
  --backend lk \
  --window-size 40 \
  --window-step 20 \
  --query-stride 5 \
  --max-windows 1
```

`--query-mode avt` tracks AVT seed-line points aligned to the virtual robot
footprint. `--query-mode sift` tracks only VENTURA-style SIFT points sampled
from that footprint. `--query-mode avt+sift` is SIFT-first: SIFT fills the
`sift.max_query_points` budget, and AVT seed-line points compensate only when
SIFT captures fewer points than that target. The default remains `avt`.

The AVT+SIFT alignment is calibration-free. It reads the decoded frame width and
height for each window, aligns an approximate virtual robot footprint to the
bottom-center image region, and records the resolved pixel bounds in
`window.json`. It does not ask for camera intrinsics, focal length, pitch, or an
accurate camera height, which keeps it usable for arbitrary internet videos.

`configs/virtual_robot.yaml` contains the default virtual robot parameters:

```yaml
query_mode: avt+sift
virtual_robot:
  # Approximate dimensions are enough.
  width_cm: 40
  length_cm: 60
  # Optional weak prior only; set to null if unknown.
  camera_height_cm: 18
sift:
  enabled: true
  max_query_points: 384
  temporal_stride: 3
```

If footprint ratios are omitted, AVT derives the bottom-image SIFT mask from
the approximate virtual robot geometry and the detected video resolution. The
defaults derive to a roughly VENTURA-like robot footprint region near the bottom
of the image. The fallback AVT seed line uses the same virtual footprint: x
spans the derived robot width and y sits midway inside the derived bottom
footprint. Pass `--seed-y-ratio`,
`--seed-x-min-ratio`, or `--seed-x-max-ratio` only when you want to override
those robot-derived defaults.

By default, each CLI run writes into a fresh child directory under
`/home/wolfie/Project/Cyber_Guider/AVT/outputs`, for example
`/home/wolfie/Project/Cyber_Guider/AVT/outputs/run_20260615_120501_123456`.
Pass `--output-root` to use a different base directory. The command prints the
resolved `output_root` and viewer path when it finishes.

Serve the generated viewer:

```bash
python -m http.server 8780 -d /home/wolfie/Project/Cyber_Guider/AVT/outputs/run_20260615_120501_123456/viewer
```

Open `http://127.0.0.1:8780/`.

## Build Viewer From Existing Tracks

The viewer does not import or call CoTracker. It only reads AVT artifacts:

```bash
avt viewer \
  --frames-root /path/to/recording_or_images \
  --tracking-root /home/wolfie/Project/Cyber_Guider/AVT/outputs/run_20260615_120501_123456
```

The viewer command also writes a fresh child directory under its output base.
By default that base is `/home/wolfie/Project/Cyber_Guider/AVT/outputs/viewer_runs`;
pass `--viewer-dir` to use a different base directory.

## Output Format

Each CLI run creates a unique run directory. Each tracked window is written to:

```text
/home/wolfie/Project/Cyber_Guider/AVT/outputs/
  run_20260615_120501_123456/
    run.json
    windows/
      seq_000_250/
        window.json
        tracks.npz
        path_mask_reference.png
        reverse_video.mp4
```

`tracks.npz` contains:

- `tracks_reverse`: `float32[T, N, 2]`, where time 0 is the last source frame
  in the window.
- `visibility_reverse`: `bool[T, N]`.
- `queries`: `float32[N, 11]` as
  `[id, reverse_time, x, y, side, source_code, response, size, angle, octave, class_id]`.
- `queries_cotracker`: `float32[N, 3]` as `[reverse_time, x, y]`, matching
  the sparse CoTracker/VENTURA query shape.
- `query_sides`: `int8[N]`.
- `query_source_codes`: `int16[N]`, where `0` is fixed AVT seeds and `1` is
  SIFT points from the virtual robot footprint.
- `query_records_json`: rich per-query metadata, including source and SIFT
  keypoint fields when available.

`window.json` also stores `query_capture.image_alignment`, including the input
window resolution, pixel footprint bounds, normalized seed ratios, and the
`image_normalized_no_intrinsics` alignment method.

Any future tracker can plug in by returning a `TrackingBundle` with the same
`tracks` and `visibility` shapes.

## Custom Tracker Backend

```python
from avt.schema import QueryPoint, TrackerInfo
from avt.tracking.base import TrackingBundle


class MyTracker:
    def track(self, frames_rgb, queries: list[QueryPoint]) -> TrackingBundle:
        tracks, visibility = run_my_model(frames_rgb, queries)
        return TrackingBundle(
            tracks=tracks,
            visibility=visibility,
            tracker=TrackerInfo(name="my-tracker"),
        )
```

Pass the backend to `run_inverse_tracking(...)` from Python, or add it to the
CLI registry in `avt/cli.py`.
