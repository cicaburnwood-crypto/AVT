# AVT

Any - Video - Trainning - Navigation

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
  --output-root outputs/example \
  --backend cotracker \
  --cotracker-device cuda \
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
  --output-root outputs/smoke \
  --backend lk \
  --window-size 40 \
  --window-step 20 \
  --query-stride 5 \
  --max-windows 1
```

Serve the generated viewer:

```bash
python -m http.server 8780 -d outputs/example/viewer
```

Open `http://127.0.0.1:8780/`.

## Build Viewer From Existing Tracks

The viewer does not import or call CoTracker. It only reads AVT artifacts:

```bash
avt viewer \
  --frames-root /path/to/recording_or_images \
  --tracking-root outputs/example \
  --viewer-dir outputs/example/viewer
```

## Output Format

Each tracked window is written to:

```text
outputs/example/
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
- `queries`: `float32[N, 5]` as `[id, reverse_time, x, y, side]`.

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
