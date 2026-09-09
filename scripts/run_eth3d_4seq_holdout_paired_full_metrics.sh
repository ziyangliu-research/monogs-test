#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-stereo}"
if [[ "$MODE" != "stereo" && "$MODE" != "mono" ]]; then
    echo "Usage: bash scripts/run_eth3d_4seq_holdout_paired_full_metrics.sh [stereo|mono]"
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

SEQUENCES=(mannequin_face_1 einstein_1 sofa_3 plant_scene_3)

if [[ "$MODE" == "stereo" ]]; then
    ENTRY="run_eth3d_holdout_paired_stereo_entry.py"
    CONFIG="configs/stereo/eth3d/mannequin_face_1.yaml"
    RESULT_PREFIX="eth3d_stereo_"
    METHOD_BASE="MonoGS-Stereo"
else
    ENTRY="run_eth3d_holdout_paired_mono_entry.py"
    CONFIG="configs/mono/eth3d/mannequin_face_1.yaml"
    RESULT_PREFIX="eth3d_mono_"
    METHOD_BASE="MonoGS-Mono"
fi

LOG_DIR="$ROOT_DIR/batch_logs/eth3d_${MODE}_4seq_holdout_paired_full_metrics"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
STATUS_FILE="$LOG_DIR/status_${STAMP}.txt"
RAW_CSV="$LOG_DIR/raw_eth3d_${MODE}_4seq_paired_${STAMP}.csv"
PAPER_CSV="$LOG_DIR/paper_eth3d_${MODE}_4seq_paired_${STAMP}.csv"

RAW_HEADER="Variant,Sequence,TotalFrames,MaxMap,Train PSNR,Train SSIM,Train LPIPS,Test PSNR,Test SSIM,Test LPIPS,ATE(m),Gaussians,Online FPS,Online Time(s),Offline Refine(s),System Total(s)"
echo "$RAW_HEADER" > "$RAW_CSV"
echo "Sequence,Method,MaxMap,RMSE SE(3)↓,Train PSNR/SSIM/LPIPS,Test PSNR/SSIM/LPIPS,FPS,Time(s),# G" > "$PAPER_CSV"

echo "MonoGS ETH3D ${MODE} paired benchmark started: $(date)" | tee "$STATUS_FILE"
echo "Sequences: ${SEQUENCES[*]}" | tee -a "$STATUS_FILE"
echo "Input: pre-rectified ETH3D RGB; no temporal downsampling" | tee -a "$STATUS_FILE"
echo "Split: strict 8:2, every fifth input frame (offset 4) is tracking-only held-out test" | tee -a "$STATUS_FILE"
echo "Protocol: one online run -> exact online-final snapshot -> released 26000-iter color refinement -> paired pre/post evaluation" | tee -a "$STATUS_FILE"
echo "ATE: SE(3), no scale correction" | tee -a "$STATUS_FILE"
echo "Timing: rendering/LPIPS/ATE and snapshot I/O are excluded; w/o CR time=online; +CR time=online+offline refinement" | tee -a "$STATUS_FILE"
echo "Images: all paired metric renders are saved; held-out test renders are additionally linked under test_renderings/" | tee -a "$STATUS_FILE"

find_pair_file() {
    local seq="$1"
    find "$ROOT_DIR/results/${RESULT_PREFIX}${seq}" \
        -type f \
        -path "*/range_000000_end/holdout_5_4_paired_color_refinement/*/paired_benchmark_rows.csv" \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

for SEQ in "${SEQUENCES[@]}"; do
    LOG_FILE="$LOG_DIR/${SEQ}_${MODE}_paired_${STAMP}.log"

    echo "============================================================" | tee -a "$STATUS_FILE"
    echo "START: ETH3D $MODE $SEQ" | tee -a "$STATUS_FILE"
    echo "TIME : $(date)" | tee -a "$STATUS_FILE"
    echo "LOG  : $LOG_FILE" | tee -a "$STATUS_FILE"
    echo "============================================================" | tee -a "$STATUS_FILE"

    python "$ENTRY" \
        --config "$CONFIG" \
        --sequence "$SEQ" \
        --holdout-every 5 \
        --holdout-offset 4 \
        --save-images \
        2>&1 | tee "$LOG_FILE"

    STATUS=${PIPESTATUS[0]}
    echo "$(date)  $SEQ  exit_code=$STATUS" | tee -a "$STATUS_FILE"

    if [[ "$STATUS" -ne 0 ]]; then
        echo "FAILED: $SEQ (continuing to next sequence)" | tee -a "$STATUS_FILE"
        continue
    fi

    PAIR_FILE="$(find_pair_file "$SEQ")"
    if [[ -z "$PAIR_FILE" || ! -f "$PAIR_FILE" ]]; then
        echo "WARNING: $SEQ completed but paired_benchmark_rows.csv was not found" | tee -a "$STATUS_FILE"
        continue
    fi

    tail -n +2 "$PAIR_FILE" >> "$RAW_CSV"

    python - "$PAIR_FILE" "$PAPER_CSV" "$METHOD_BASE" <<'PY'
import csv
import os
import sys
from pathlib import Path

pair_file, paper_file, method_base = sys.argv[1:4]
pair_path = Path(pair_file).resolve()
save_dir = pair_path.parent

with pair_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

with open(paper_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for r in rows:
        variant = r["Variant"]
        total_frames = int(r["TotalFrames"])
        max_map = int(r["MaxMap"])
        max_map_pct = 100.0 * max_map / total_frames if total_frames else 0.0
        train = (
            f'{float(r["Train PSNR"]):.2f}/'
            f'{float(r["Train SSIM"]):.4f}/'
            f'{float(r["Train LPIPS"]):.4f}'
        )
        test = (
            f'{float(r["Test PSNR"]):.2f}/'
            f'{float(r["Test SSIM"]):.4f}/'
            f'{float(r["Test LPIPS"]):.4f}'
        )
        online_fps = float(r["Online FPS"])
        online_t = float(r["Online Time(s)"])
        offline_t = float(r["Offline Refine(s)"])
        total_t = float(r["System Total(s)"])
        gaussians = int(float(r["Gaussians"]))
        if variant == "w/o CR":
            fps_text = f"{online_fps:.2f}"
            time_text = f"{online_t:.1f}"
        else:
            fps_text = "-"
            time_text = f"{online_t:.1f} + {offline_t:.1f} = {total_t:.1f}"
        writer.writerow([
            r["Sequence"],
            f"{method_base} ({variant})",
            f"{max_map_pct:.2f}%",
            f'{float(r["ATE(m)"]):.4f} m',
            train,
            test,
            fps_text,
            time_text,
            f"{gaussians / 1000.0:.1f}k",
        ])

# Convenience views containing only the strict held-out test renderings.
# The canonical all-frame render directories remain untouched.
for stage, label in [("online_final", "w_o_CR"), ("final", "plus_CR")]:
    metrics_path = save_dir / "psnr" / stage / "per_frame_metrics.csv"
    if not metrics_path.exists():
        continue
    with metrics_path.open(newline="", encoding="utf-8") as f:
        metric_rows = list(csv.DictReader(f))
    for kind in ("rendered", "gt"):
        src_dir = save_dir / "psnr" / stage / kind
        dst_dir = save_dir / "test_renderings" / label / kind
        dst_dir.mkdir(parents=True, exist_ok=True)
        for r in metric_rows:
            frame_id = int(r["frame_id"])
            if frame_id % 5 != 4:
                continue
            src = src_dir / f"{frame_id:06d}.png"
            dst = dst_dir / src.name
            if not src.exists():
                continue
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(os.path.relpath(src, start=dst.parent))
PY

    echo "SUCCESS: $SEQ -> $PAIR_FILE" | tee -a "$STATUS_FILE"
done

echo "============================================================" | tee -a "$STATUS_FILE"
echo "Batch finished: $(date)" | tee -a "$STATUS_FILE"
echo "Raw paired CSV : $RAW_CSV" | tee -a "$STATUS_FILE"
echo "Paper table CSV: $PAPER_CSV" | tee -a "$STATUS_FILE"
echo "Status log     : $STATUS_FILE" | tee -a "$STATUS_FILE"
echo
echo "================ PAPER TABLE ================"
cat "$PAPER_CSV"
