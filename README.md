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
requires PyTorch and a CoTracker hub-capable environment. The optional
`bootstap` backend is a separate BootsTAPIR adapter using Google DeepMind's
TAPNet PyTorch implementation. The optional `foundationpose` backend is an
adapter for FoundationPose-derived pose outputs.

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

By default, tracking writes only the minimum AVT artifacts needed to rebuild
analysis or visualization later: `run.json`, each window's `window.json`, and
compressed `tracks.npz`. It does not write reverse videos, per-window mask PNGs,
or a WebUI unless explicitly requested.

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

## CoTracker Cache Backend

For dense or repeatedly tuned workflows, precompute CoTracker once and reuse the
saved point IDs in later AVT extraction runs. This avoids re-running CoTracker
for every overlapping window when you only change SIFT, anchor, mask, or window
parameters.

```bash
avt cache \
  --frames-root /path/to/images \
  --source-type image_dir \
  --cache-config configs/cotracker_cache.yaml \
  --cache-frame-start 0 \
  --cache-frame-count 100 \
```

The command writes a reusable cache directory under `outputs/cotracker_caches`
with memory-mappable `.npy` arrays and `metadata.json`. Time is stored in global
reverse-video order: cache time 0 is the last frame in the cached range.

For long videos, build overlapping independent chunks instead of one huge
cache:

```bash
avt cache-chunks \
  --frames-root /path/to/images \
  --source-type image_dir \
  --cache-config configs/cotracker_cache.yaml
```

`cache-chunks` reads chunk size, extraction window size, chunk step, CoTracker
runtime settings, and `bad_track_confidence_threshold` from
`configs/cotracker_cache.yaml`. If `chunk.step` is omitted it uses
`chunk.size - chunk.window_size`, so the default YAML step is 400 frames and
every 80-frame extraction window can fit inside at least one cache chunk.
Chunks are not merged and do not share point identity.

Then extract AVT windows from the cache:

```bash
avt all \
  --frames-root /path/to/images \
  --source-type image_dir \
  --backend cotracker_cache \
  --cotracker-cache outputs/cotracker_caches/run_YYYYMMDD_HHMMSS_xxxxxx \
  --cache-match-distance 12 \
  --query-mode ventura \
  --robot-config configs/virtual_robot.yaml \
  --window-size 80 \
  --window-step 1 \
  --save-path-mask
```

When `--cotracker-cache` points at a chunked cache manifest/root, extraction
selects one chunk that fully contains each window. If multiple chunks cover the
window, AVT evaluates each candidate and uses the one with the highest mean
tracker confidence, with unmatched points counted as zero confidence. In
ID-only artifacts, point identity is chunk-qualified with arrays such as
`cache_chunk_ids` and `cache_unique_point_ids`, for example
`chunk_000400_000879:17`.

By default this cached backend still writes regular `tracks_reverse` arrays for
compatibility. Add `--cache-record-ids-only` to write compact per-window
references instead:

```bash
avt all \
  --frames-root /path/to/images \
  --source-type image_dir \
  --backend cotracker_cache \
  --cotracker-cache /path/to/cache \
  --cache-record-ids-only \
  --query-mode ventura \
  --robot-config configs/virtual_robot.yaml \
  --window-size 80 \
  --window-step 1 \
  --save-path-mask
```

In ID-only mode `tracks.npz` stores arrays such as `cache_point_ids`,
`cache_point_indices`, `cache_chunk_ids`, `cache_unique_point_ids`,
`cache_query_source_frames`, `cache_birth_source_frames`, and
`cache_match_distances` instead of duplicating full tracks per window.
`cache_point_ids` are local to the selected chunk; use
`cache_unique_point_ids` when IDs need to be unique across chunked caches. The
viewer can materialize those tracks from the referenced cache later.

Cache query mode:

- `confidence-refresh`: seed raw dense pixels once on reverse frame 0, check
  confidence/visibility every frame, abort an ID at its first failed check, and
  create one child ID at that abort frame. The cache does not grid-downsample
  and does not create replacement IDs while the parent remains reliable.

For the current top-10 ride cache batch, use the resumable runner:

```bash
python scripts/run_top10_cache_chunks.py \
  --data-root data \
  --prepared-root data/prepared_top10_cache \
  --output-root outputs/top10_cotracker_cache_chunks \
  --cache-config configs/cotracker_cache.yaml \
  --camera front \
  --limit-videos 2
```

The runner prepares front-camera frames from the ride HLS playlists when needed
and reads cache generation parameters from `configs/cotracker_cache.yaml`.
Adjust chunk/window size, CoTracker runtime settings, and
`bad_track_confidence_threshold` in that YAML file rather than adding cache
knobs to the script.
and then writes fixed chunk-cache directories, so reruns reuse completed frames
and completed chunks. It does not run extraction or visualization.

## BootsTAPIR Backend

BootsTAPIR is available as a separate selectable backend, parallel to
CoTracker:

```bash
python -m pip install -e '.[bootstap]'

python -m avt.tracking.bootstap.download

avt all \
  --frames-root /path/to/recording_or_images \
  --source-type auto \
  --backend bootstap \
  --bootstap-config configs/bootstap.yaml \
  --query-mode ventura \
  --robot-config configs/virtual_robot.yaml
```

The backend keeps AVT's query generation and artifact format unchanged. It
converts AVT's `[reverse_time, x, y]` query points into TAPNet's `[t, y, x]`
resized raster convention, runs `tapnet.torch.tapir_model.TAPIR`, and converts
the result back to AVT's `tracks_reverse` and `visibility_reverse` arrays.

BootsTAPIR-specific knobs are namespaced so they do not affect CoTracker:

- `--bootstap-checkpoint`
- `--bootstap-download-checkpoint` / `--no-bootstap-download-checkpoint`
- `--bootstap-resize-height` and `--bootstap-resize-width`
- `--bootstap-query-chunk-size`
- `--bootstap-pyramid-level`
- `--bootstap-visibility-threshold`
- `--bootstap-strict-checkpoint` / `--no-bootstap-strict-checkpoint`

`configs/bootstap.yaml` is an optional backend config. The default checkpoint is
stored outside git under `checkpoints/bootstap/bootstapir_checkpoint_v2.pt`.
The checkpoint URL follows the official TAPNet PyTorch BootsTAPIR notebook:
`https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt`.
The TAPNet README notes that BootsTAPIR typically performs best at `512x512`,
which is why the AVT config uses that resize by default.

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
  width_ratio: 0.25
  height_ratio: 0.20
sift:
  enabled: true
  max_query_points: 384
  window_size: 20
  min_points_per_frame: 8
  max_points_per_frame: 20
  edge_offset_ratio: 0.10
  use_clahe: true
  anchors:
    enabled: true
    max_query_points: 384
    min_points_per_frame: 8
    max_points_per_frame: 20
```

The SIFT `window_size` is VENTURA's SIFT sampling interval, separate from the
tracking `--window-size`. The default crumb SIFT parameters are
`contrastThreshold=0.018`, `edgeThreshold=20`, `nOctaveLayers=5`, `sigma=1.5`;
anchor SIFT uses `contrastThreshold=0.008`, `edgeThreshold=15`,
`nOctaveLayers=3`, `sigma=1.2`. Both use CLAHE before SIFT by default. Like
VENTURA, AVT clamps each selected SIFT frame to 8-20 query points so long
tracking windows do not become under-seeded.

When `--save-path-mask` is enabled, the displayed reference-frame path mask also
uses VENTURA-style support crumbs: extra relaxed SIFT points are sampled on the
bottom robot footprint of the reference frame and used only to draw the mask.
Disable support crumbs with `--no-path-support`, or tune them with
`--path-support-min-points` and `--path-support-fraction`.

By default, each CLI run writes into a fresh child directory under
`/home/wolfie/Project/Cyber_Guider/AVT/outputs`, for example
`/home/wolfie/Project/Cyber_Guider/AVT/outputs/run_20260615_120501_123456`.
Pass `--output-root` to use a different base directory. The command prints the
resolved `output_root` when it finishes. Add `--build-viewer` if you want `avt
all` to build the static WebUI immediately.

Build and serve the viewer only when needed:

```bash
avt viewer \
  --frames-root /path/to/recording_or_images \
  --tracking-root /home/wolfie/Project/Cyber_Guider/AVT/outputs/run_20260615_120501_123456
```

```bash
python -m http.server 8780 -d /path/printed/by/avt/viewer
```

Use the `viewer` path printed by the command as the `-d` directory, then open
`http://127.0.0.1:8780/`.

## FoundationPose Backend

FoundationPose is available as a separate selectable backbone:

```bash
python -m avt.tracking.foundationpose.download

avt all \
  --frames-root /path/to/recording_or_images \
  --backend foundationpose \
  --foundationpose-transforms /path/to/foundationpose_transforms.npz \
  --query-mode ventura \
  --robot-config configs/virtual_robot.yaml
```

This keeps the original CoTracker/LK pipeline intact. AVT still extracts the
same VENTURA-aligned SIFT anchor and robot-footprint query points; the
FoundationPose backend converts pose-derived image transforms into AVT point
tracks.

FoundationPose itself is not an RGB-only point tracker. A real FoundationPose
run needs RGB-D frames, object masks, camera intrinsics, and CAD/reference object
data. The AVT backend expects one of these pose-derived files:

- `.npz` with `tracks_reverse` and optional `visibility_reverse`
- `.npz` or `.json` with `homographies_reverse` or `homographies`, shaped
  `[T, 3, 3]`

For multi-window runs, `--foundationpose-transforms` may point to a directory
containing per-window files such as `seq_0_240.npz`, `seq_60_300.npz`, or nested
files like `seq_0_240/transforms.npz`.

Homographies are absolute transforms from reverse frame 0 to each reverse frame.
For a query inserted at reverse time `q`, AVT applies `H[t] @ inv(H[q])` so the
same query schema works with either CoTracker or FoundationPose-derived motion.

The FoundationPose weights are stored outside git under
`checkpoints/foundationpose/`. The downloader fetches the refiner
`2023-10-28-18-33-37/model_best.pth` and scorer
`2024-01-11-20-02-45/model_best.pth` checkpoints.

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
```

Optional debug outputs are opt-in:

- `--save-reverse-video` writes `reverse_video.mp4` for each window.
- `--save-path-mask` writes `path_mask_reference.png` for each window.
- `--build-viewer` makes `avt all` build the WebUI immediately. Otherwise use
  `avt viewer` later against the saved tracking directory.

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

When `--cache-record-ids-only` is used with `--backend cotracker_cache`,
`tracks.npz` omits `tracks_reverse` and `visibility_reverse` and instead stores
cache ID/frame reference arrays. `window.json` records `artifact_mode:
cache_ids_only` and the cache path needed to materialize tracks.

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
