#!/usr/bin/env bash
set -Eeuo pipefail

# Launch three detached AVT_P jobs on GPUs that stay below both memory and
# compute thresholds during a short monitoring window.

AVT_ROOT="${AVT_ROOT:-/data/disk_14t/diwen/AVT_P}"
DATA_ROOT="${DATA_ROOT:-/data/disk_14t/diwen/data/Frodobots/frodobots-2k-dataset/output_rides_0}"
PYTHON_BIN="${PYTHON_BIN:-/data/disk_14t/diwen/micromamba/envs/avt/bin/python}"

GPU_COUNT="${GPU_COUNT:-3}"
GPU_MEM_LIMIT_PCT="${GPU_MEM_LIMIT_PCT:-20}"
GPU_UTIL_LIMIT_PCT="${GPU_UTIL_LIMIT_PCT:-20}"
GPU_MONITOR_SECONDS="${GPU_MONITOR_SECONDS:-15}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-3}"
GPU_WAIT_RETRIES="${GPU_WAIT_RETRIES:-60}"
GPU_WAIT_SLEEP="${GPU_WAIT_SLEEP:-60}"

FPS="${FPS:-10}"
WINDOW_SIZE="${WINDOW_SIZE:-250}"
WINDOW_STEP="${WINDOW_STEP:-100}"
DETECTOR="${DETECTOR:-sift}"
COTRACKER_BATCH_SIZE="${COTRACKER_BATCH_SIZE:-256}"
EST_SECONDS_PER_WINDOW="${EST_SECONDS_PER_WINDOW:-12}"
DECODE_REALTIME_FACTOR="${DECODE_REALTIME_FACTOR:-8}"
BUILD_VIEWER="${BUILD_VIEWER:-1}"
SAVE_PATH_MASK="${SAVE_PATH_MASK:-1}"
MAX_WINDOWS="${MAX_WINDOWS:-}"
FFMPEG_THREADS="${FFMPEG_THREADS:-2}"
JPEG_QUALITY="${JPEG_QUALITY:-3}"

TORCH_HOME_DIR="${TORCH_HOME_DIR:-$AVT_ROOT/model_cache/torch}"
HF_HOME_DIR="${HF_HOME_DIR:-$AVT_ROOT/model_cache/huggingface}"
HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-$HF_HOME_DIR/hub}"
COTRACKER_REPO="${COTRACKER_REPO:-$TORCH_HOME_DIR/hub/facebookresearch_co-tracker_main}"
COTRACKER_MODEL="${COTRACKER_MODEL:-cotracker3_offline}"

DEFAULT_RIDES=(
  "ride_17620_20240130041859"
  "ride_16577_20240117062216"
  "ride_17102_20240125075115"
)

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_frodobots_3gpu_detached.sh launch [ride_a ride_b ride_c]
  scripts/run_frodobots_3gpu_detached.sh status [run_root]

Default rides are the three long, visually active output_rides_0 records:
  ride_17620_20240130041859
  ride_16577_20240117062216
  ride_17102_20240125075115

Useful overrides:
  MAX_WINDOWS=5                         quick smoke run
  BUILD_VIEWER=0                        skip static WebUI build
  EST_SECONDS_PER_WINDOW=12             rough launch ETA model
  GPU_MEM_LIMIT_PCT=20 GPU_UTIL_LIMIT_PCT=20
  AVT_ROOT=/data/disk_14t/diwen/AVT_P
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

fmt_seconds() {
  local total="${1:-0}"
  total="${total%.*}"
  local h=$(( total / 3600 ))
  local m=$(( (total % 3600) / 60 ))
  local s=$(( total % 60 ))
  if (( h > 0 )); then
    printf "%dh%02dm%02ds" "$h" "$m" "$s"
  else
    printf "%dm%02ds" "$m" "$s"
  fi
}

ride_id_from_name() {
  local ride="$1"
  [[ "$ride" =~ ^ride_([0-9]+)_ ]] || die "Bad ride name: $ride"
  printf "%s" "${BASH_REMATCH[1]}"
}

front_m3u8_for_ride() {
  local ride="$1"
  local ride_id
  ride_id="$(ride_id_from_name "$ride")"
  local ride_dir="$DATA_ROOT/$ride"
  [[ -d "$ride_dir" ]] || die "Ride directory not found: $ride_dir"
  local found
  found="$(find "$ride_dir/recordings" -maxdepth 1 -type f -name "*ride_${ride_id}__uid_s_1000__uid_e_video.m3u8" | sort | head -1)"
  if [[ -z "$found" ]]; then
    found="$(find "$ride_dir/recordings" -maxdepth 1 -type f -name "*ride_${ride_id}__uid_s_*__uid_e_video.m3u8" | sort | head -1)"
  fi
  [[ -n "$found" ]] || die "No front video m3u8 found for $ride"
  printf "%s" "$found"
}

m3u8_duration_seconds() {
  awk -F'[:,]' '/^#EXTINF:/ {sum += $2} END {printf "%.3f", sum + 0}' "$1"
}

ceil_float_to_int() {
  awk -v x="$1" 'BEGIN {printf "%d", (x == int(x) ? x : int(x) + 1)}'
}

window_count_for_frames() {
  local frames="$1"
  if (( frames < 2 )); then
    printf "0"
  elif [[ -n "$MAX_WINDOWS" && "$MAX_WINDOWS" -gt 0 ]]; then
    local all_windows=$(( ((frames - 2) / WINDOW_STEP) + 1 ))
    (( all_windows < MAX_WINDOWS )) && printf "%d" "$all_windows" || printf "%d" "$MAX_WINDOWS"
  else
    printf "%d" $(( ((frames - 2) / WINDOW_STEP) + 1 ))
  fi
}

collect_gpu_samples() {
  local output="$1"
  : > "$output"
  local samples=$(( GPU_MONITOR_SECONDS / GPU_MONITOR_INTERVAL + 1 ))
  (( samples < 2 )) && samples=2
  for ((i = 0; i < samples; i++)); do
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> "$output"
    if (( i + 1 < samples )); then
      sleep "$GPU_MONITOR_INTERVAL"
    fi
  done
}

free_gpu_lines_from_samples() {
  local samples="$1"
  awk -F',' \
    -v mem_limit="$GPU_MEM_LIMIT_PCT" \
    -v util_limit="$GPU_UTIL_LIMIT_PCT" '
    {
      gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); gsub(/ /, "", $4)
      idx = $1
      used = $2 + 0
      total = $3 + 0
      util = $4 + 0
      mem_pct = total > 0 ? used * 100.0 / total : 100.0
      seen[idx] = 1
      if (!(idx in max_mem) || mem_pct > max_mem[idx]) max_mem[idx] = mem_pct
      if (!(idx in max_util) || util > max_util[idx]) max_util[idx] = util
    }
    END {
      for (idx in seen) {
        if (max_mem[idx] < mem_limit && max_util[idx] < util_limit) {
          printf "%s\t%.2f\t%.2f\n", idx, max_mem[idx], max_util[idx]
        }
      }
    }' "$samples" | sort -n
}

write_job_script() {
  local job_script="$1"
  cat > "$job_script" <<'JOB'
#!/usr/bin/env bash
set -Eeuo pipefail

write_stage() {
  printf "%s\n" "$1" > "$JOB_DIR/stage"
  date -Is > "$JOB_DIR/stage_${1}_at"
}

mark_exit() {
  local rc="$?"
  printf "%s\n" "$rc" > "$JOB_DIR/exit_code"
  if [[ "$rc" -eq 0 ]]; then
    write_stage "done"
  else
    write_stage "failed"
  fi
  exit "$rc"
}

gpu_is_free_once() {
  local samples="$JOB_DIR/gpu_recheck_samples.tsv"
  : > "$samples"
  local sample_count=$(( GPU_MONITOR_SECONDS / GPU_MONITOR_INTERVAL + 1 ))
  (( sample_count < 2 )) && sample_count=2
  for ((i = 0; i < sample_count; i++)); do
    nvidia-smi --id="$GPU" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> "$samples"
    if (( i + 1 < sample_count )); then
      sleep "$GPU_MONITOR_INTERVAL"
    fi
  done
  awk -F',' -v mem_limit="$GPU_MEM_LIMIT_PCT" -v util_limit="$GPU_UTIL_LIMIT_PCT" '
    {
      gsub(/ /, "", $2); gsub(/ /, "", $3); gsub(/ /, "", $4)
      used = $2 + 0
      total = $3 + 0
      util = $4 + 0
      mem_pct = total > 0 ? used * 100.0 / total : 100.0
      if (mem_pct > max_mem) max_mem = mem_pct
      if (util > max_util) max_util = util
    }
    END {
      if (max_mem < mem_limit && max_util < util_limit) exit 0
      exit 1
    }' "$samples"
}

wait_for_gpu() {
  write_stage "wait_gpu"
  for ((attempt = 1; attempt <= GPU_WAIT_RETRIES; attempt++)); do
    echo "[$(date -Is)] checking GPU $GPU availability before tracking, attempt $attempt/$GPU_WAIT_RETRIES"
    if gpu_is_free_once; then
      echo "[$(date -Is)] GPU $GPU stayed free for ${GPU_MONITOR_SECONDS}s; starting tracking"
      return 0
    fi
    echo "[$(date -Is)] GPU $GPU is no longer free; sleeping ${GPU_WAIT_SLEEP}s"
    sleep "$GPU_WAIT_SLEEP"
  done
  echo "GPU $GPU did not become free after retries" >&2
  return 1
}

trap mark_exit EXIT

mkdir -p "$JOB_DIR" "$FRAMES_DIR" "$RESULT_ROOT"
printf "%s\n" "$$" > "$JOB_DIR/pid"
date +%s > "$JOB_DIR/start_epoch"
write_stage "decode"

echo "ride=$RIDE"
echo "gpu=$GPU"
echo "m3u8=$M3U8"
echo "frames_dir=$FRAMES_DIR"
echo "result_root=$RESULT_ROOT"
echo "fps=$FPS window_size=$WINDOW_SIZE window_step=$WINDOW_STEP"
echo "python=$PYTHON_BIN"

if [[ ! -f "$FRAMES_DIR/.decode_complete" ]]; then
  ffmpeg \
    -hide_banner \
    -nostdin \
    -loglevel warning \
    -stats \
    -threads "$FFMPEG_THREADS" \
    -i "$M3U8" \
    -vf "fps=$FPS" \
    -q:v "$JPEG_QUALITY" \
    "$FRAMES_DIR/frame_%06d.jpg"
  find "$FRAMES_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l > "$FRAMES_DIR/.frame_count"
  {
    echo "ride=$RIDE"
    echo "m3u8=$M3U8"
    echo "fps=$FPS"
    echo "completed_at=$(date -Is)"
  } > "$FRAMES_DIR/.decode_complete"
else
  echo "decode already complete: $FRAMES_DIR"
fi

wait_for_gpu
write_stage "track"

AVT_ARGS=(
  all
  --frames-root "$FRAMES_DIR"
  --source-type image_dir
  --backend cotracker
  --cotracker-device cuda
  --cotracker-batch-size "$COTRACKER_BATCH_SIZE"
  --torch-home "$TORCH_HOME_DIR"
  --cotracker-hub-repo "$COTRACKER_REPO"
  --cotracker-hub-model "$COTRACKER_MODEL"
  --query-mode anchor_motion
  --query-config "$AVT_ROOT/configs/anchor_motion.yaml"
  --detector "$DETECTOR"
  --window-size "$WINDOW_SIZE"
  --window-step "$WINDOW_STEP"
  --fps "$FPS"
  --output-root "$RESULT_ROOT"
)

if [[ -n "$MAX_WINDOWS" ]]; then
  AVT_ARGS+=(--max-windows "$MAX_WINDOWS")
fi
if [[ "$BUILD_VIEWER" == "1" ]]; then
  AVT_ARGS+=(--build-viewer)
fi
if [[ "$SAVE_PATH_MASK" == "1" ]]; then
  AVT_ARGS+=(--save-path-mask)
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$AVT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$TORCH_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON_BIN" -m avt.cli "${AVT_ARGS[@]}"
JOB
  chmod +x "$job_script"
}

print_status() {
  local run_root="${1:-$AVT_ROOT/outputs/frodobots_3gpu_latest}"
  if [[ -L "$run_root" ]]; then
    run_root="$(readlink -f "$run_root")"
  fi
  [[ -d "$run_root" ]] || die "Run root not found: $run_root"
  local manifest="$run_root/manifest.tsv"
  [[ -f "$manifest" ]] || die "Manifest not found: $manifest"

  local now
  now="$(date +%s)"
  printf "run_root\t%s\n" "$run_root"
  printf "now\t%s\n" "$(date -Is)"
  printf "ride\tgpu\tstage\tpid\talive\tprogress\teta\tlog\n"

  tail -n +2 "$manifest" | while IFS=$'\t' read -r ride gpu duration frames windows eta_s job_dir log frames_dir result_root m3u8 pid; do
    local stage="unknown"
    [[ -f "$job_dir/stage" ]] && stage="$(<"$job_dir/stage")"
    local alive="no"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      alive="yes"
    fi
    local progress="-"
    local eta="$eta_s"
    local start_epoch=""
    [[ -f "$job_dir/start_epoch" ]] && start_epoch="$(<"$job_dir/start_epoch")"
    if [[ "$stage" == "decode" ]]; then
      local decoded=0
      decoded="$(find "$frames_dir" -maxdepth 1 -type f -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')"
      progress="decode ${decoded}/${frames}"
      if [[ -n "$start_epoch" && "$decoded" -gt 100 ]]; then
        local elapsed=$(( now - start_epoch ))
        eta="$(awk -v e="$elapsed" -v d="$decoded" -v f="$frames" 'BEGIN {printf "%d", e * (f - d) / d}')"
      fi
    elif [[ "$stage" == "track" ]]; then
      local line=""
      line="$(grep -E '^\[[0-9]+/[0-9]+\] tracking ' "$log" 2>/dev/null | tail -1 || true)"
      if [[ "$line" =~ ^\[([0-9]+)/([0-9]+)\] ]]; then
        local done="${BASH_REMATCH[1]}"
        local total="${BASH_REMATCH[2]}"
        progress="track ${done}/${total}"
        if [[ -n "$start_epoch" && "$done" -gt 0 ]]; then
          local elapsed=$(( now - start_epoch ))
          eta="$(awk -v e="$elapsed" -v d="$done" -v t="$total" 'BEGIN {printf "%d", e * (t - d) / d}')"
        fi
      else
        progress="track starting"
      fi
    elif [[ "$stage" == "done" || "$stage" == "failed" ]]; then
      local rc=""
      [[ -f "$job_dir/exit_code" ]] && rc="$(<"$job_dir/exit_code")"
      progress="exit ${rc:-unknown}"
      eta=0
    elif [[ "$stage" == "wait_gpu" ]]; then
      progress="waiting for GPU"
      eta="$eta_s"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$ride" "$gpu" "$stage" "$pid" "$alive" "$progress" "$(fmt_seconds "$eta")" "$log"
  done
}

launch() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
  command -v ffmpeg >/dev/null || die "ffmpeg not found"
  [[ -x "$PYTHON_BIN" ]] || die "Python not executable: $PYTHON_BIN"
  [[ -d "$AVT_ROOT" ]] || die "AVT_ROOT not found: $AVT_ROOT"

  local rides=("$@")
  if (( ${#rides[@]} == 0 )); then
    rides=("${DEFAULT_RIDES[@]}")
  fi
  (( ${#rides[@]} == GPU_COUNT )) || die "Expected exactly $GPU_COUNT rides, got ${#rides[@]}"

  local run_id="${RUN_ID:-frodobots_3gpu_$(date +%Y%m%d_%H%M%S)}"
  local run_root="${OUTPUT_ROOT:-$AVT_ROOT/outputs/$run_id}"
  local latest_link="$AVT_ROOT/outputs/frodobots_3gpu_latest"
  local logs_dir="$run_root/logs"
  local jobs_dir="$run_root/jobs"
  local frames_root="$run_root/frames"
  local results_root="$run_root/results"
  mkdir -p "$logs_dir" "$jobs_dir" "$frames_root" "$results_root"

  local samples="$run_root/gpu_samples.tsv"
  echo "Monitoring GPUs for ${GPU_MONITOR_SECONDS}s..."
  collect_gpu_samples "$samples"

  mapfile -t free_lines < <(free_gpu_lines_from_samples "$samples")
  if (( ${#free_lines[@]} < GPU_COUNT )); then
    echo "GPU samples:"
    cat "$samples"
    die "Need $GPU_COUNT free GPUs, found ${#free_lines[@]}"
  fi

  local gpus=()
  for ((i = 0; i < GPU_COUNT; i++)); do
    gpus+=("$(awk '{print $1}' <<<"${free_lines[$i]}")")
  done

  local job_script="$run_root/run_one_ride_job.sh"
  write_job_script "$job_script"

  local manifest="$run_root/manifest.tsv"
  printf "ride\tgpu\tduration_s\texpected_frames\twindows\teta_s\tjob_dir\tlog\tframes_dir\tresult_root\tm3u8\tpid\n" > "$manifest"

  local max_eta=0
  for ((i = 0; i < GPU_COUNT; i++)); do
    local ride="${rides[$i]}"
    local gpu="${gpus[$i]}"
    local m3u8
    m3u8="$(front_m3u8_for_ride "$ride")"
    local duration
    duration="$(m3u8_duration_seconds "$m3u8")"
    local expected_frames
    expected_frames="$(ceil_float_to_int "$(awk -v d="$duration" -v fps="$FPS" 'BEGIN {print d * fps}')")"
    local windows
    windows="$(window_count_for_frames "$expected_frames")"
    local eta_s
    eta_s="$(awk -v d="$duration" -v f="$DECODE_REALTIME_FACTOR" -v w="$windows" -v spw="$EST_SECONDS_PER_WINDOW" 'BEGIN {printf "%d", (d / f) + (w * spw)}')"
    (( eta_s > max_eta )) && max_eta="$eta_s"

    local job_dir="$jobs_dir/$ride"
    local log="$logs_dir/$ride.log"
    local frames_dir="$frames_root/$ride/front_fps${FPS//./p}"
    local result_root="$results_root/$ride"
    mkdir -p "$job_dir" "$result_root"

    setsid env \
      AVT_ROOT="$AVT_ROOT" \
      DATA_ROOT="$DATA_ROOT" \
      PYTHON_BIN="$PYTHON_BIN" \
      RIDE="$ride" \
      GPU="$gpu" \
      M3U8="$m3u8" \
      JOB_DIR="$job_dir" \
      FRAMES_DIR="$frames_dir" \
      RESULT_ROOT="$result_root" \
      FPS="$FPS" \
      WINDOW_SIZE="$WINDOW_SIZE" \
      WINDOW_STEP="$WINDOW_STEP" \
      DETECTOR="$DETECTOR" \
      COTRACKER_BATCH_SIZE="$COTRACKER_BATCH_SIZE" \
      TORCH_HOME_DIR="$TORCH_HOME_DIR" \
      HF_HOME_DIR="$HF_HOME_DIR" \
      HF_HUB_CACHE_DIR="$HF_HUB_CACHE_DIR" \
      COTRACKER_REPO="$COTRACKER_REPO" \
      COTRACKER_MODEL="$COTRACKER_MODEL" \
      MAX_WINDOWS="$MAX_WINDOWS" \
      BUILD_VIEWER="$BUILD_VIEWER" \
      SAVE_PATH_MASK="$SAVE_PATH_MASK" \
      FFMPEG_THREADS="$FFMPEG_THREADS" \
      JPEG_QUALITY="$JPEG_QUALITY" \
      GPU_MONITOR_SECONDS="$GPU_MONITOR_SECONDS" \
      GPU_MONITOR_INTERVAL="$GPU_MONITOR_INTERVAL" \
      GPU_MEM_LIMIT_PCT="$GPU_MEM_LIMIT_PCT" \
      GPU_UTIL_LIMIT_PCT="$GPU_UTIL_LIMIT_PCT" \
      GPU_WAIT_RETRIES="$GPU_WAIT_RETRIES" \
      GPU_WAIT_SLEEP="$GPU_WAIT_SLEEP" \
      bash "$job_script" > "$log" 2>&1 < /dev/null &
    local pid="$!"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$ride" "$gpu" "$duration" "$expected_frames" "$windows" "$eta_s" \
      "$job_dir" "$log" "$frames_dir" "$result_root" "$m3u8" "$pid" >> "$manifest"
  done

  ln -sfn "$run_root" "$latest_link"

  echo "Detached AVT_P run launched."
  echo "run_root: $run_root"
  echo "selected_gpus: ${gpus[*]}"
  echo "rough_parallel_eta: $(fmt_seconds "$max_eta")"
  echo "eta_model: decode=duration/${DECODE_REALTIME_FACTOR} + windows*${EST_SECONDS_PER_WINDOW}s"
  echo "status: $0 status $run_root"
  echo "manifest: $manifest"
}

main() {
  local cmd="${1:-launch}"
  shift || true
  case "$cmd" in
    launch)
      launch "$@"
      ;;
    status)
      print_status "${1:-}"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      die "Unknown command: $cmd"
      ;;
  esac
}

main "$@"
