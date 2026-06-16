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
  --query-mode ventura \
  --robot-config configs/virtual_robot.yaml \
  --window-size 250 \
  --window-step 100 \
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

`--query-mode ventura` is the default. It mirrors VENTURA's image-process
assumptions: reverse the video window, sample full-frame SIFT anchors for
tracking stability, sample robot-footprint SIFT crumbs from a bottom-center
percentage mask, then build the path mask from the crumb points while excluding
the anchors. `--query-mode sift` keeps only the robot-footprint crumbs, and
`--query-mode avt` remains as a manual deterministic seed-line fallback.
`avt+sift` is accepted as a compatibility alias for the VENTURA anchor+crumb
pipeline.

The VENTURA alignment is calibration-free. It reads the decoded frame width and
height for each window, applies `ROBOT_WIDTH_PCT` and `ROBOT_HEIGHT_PCT` style
percentages to a bottom-center rectangle, and records the resolved pixel bounds
in `window.json`. It does not ask for camera intrinsics, focal length, pitch, or
accurate camera height, which keeps it usable for arbitrary internet videos.

`configs/virtual_robot.yaml` now contains the default VENTURA-style parameters:

```yaml
query_mode: ventura
footprint:
  width_ratio: 0.20
  height_ratio: 0.15
sift:
  enabled: true
  max_query_points: 384
  window_size: 20
  edge_offset_ratio: 0.10
  use_clahe: true
  anchors:
    enabled: true
    max_query_points: 384
```

The SIFT `window_size` is VENTURA's SIFT sampling interval, separate from the
tracking `--window-size`. The default crumb SIFT parameters are
`contrastThreshold=0.02`, `edgeThreshold=18`, `nOctaveLayers=7`, `sigma=1.6`;
anchor SIFT uses `contrastThreshold=0.008`, `edgeThreshold=15`,
`nOctaveLayers=3`, `sigma=1.2`. Both use CLAHE before SIFT by default.

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
- `query_source_codes`: `int16[N]`, where `0` is fixed AVT seeds, `1` is
  VENTURA robot-footprint SIFT crumbs, and `2` is full-frame SIFT anchors.
- `query_records_json`: rich per-query metadata, including source and SIFT
  keypoint fields when available.

`window.json` also stores `query_capture.image_alignment`, including the input
window resolution, pixel footprint bounds, normalized seed ratios, and the
`ventura_pct_bottom_center` alignment method.

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
