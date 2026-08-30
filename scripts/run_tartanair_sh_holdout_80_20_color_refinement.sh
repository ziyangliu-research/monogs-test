#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

SEQUENCES=(SH000 SH001 SH002 SH003)
LOG_DIR="$ROOT_DIR/batch_logs/holdout_80_20_color_refinement"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
STATUS_FILE="$LOG_DIR/status_${STAMP}.txt"
SUMMARY_CSV="$LOG_DIR/summary_SH000_SH003_full_holdout_80_20_color_refinement_${STAMP}.csv"

# Main quality table intentionally omits FPS. OnlineFPS and TotalWallSec are kept
# as auxiliary timing information and should not be interpreted as one common FPS.
echo "Sequence,MaxMap,Train PSNR,Train SSIM,Test PSNR,Test SSIM,ATE(m),Gaussians,OnlineFPS,TotalWallSec" > "$SUMMARY_CSV"

echo "MonoGS stereo + original color refinement batch started: $(date)" | tee "$STATUS_FILE"

auto_find_row() {
    local seq="$1"
    find "$ROOT_DIR/results/tartanair_stereo_${seq}" \
        -type f \
        -path "*/range_000000_end/holdout_5_4_color_refinement/*/benchmark_row.csv" \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

for SEQ in "${SEQUENCES[@]}"; do
    LOG_FILE="$LOG_DIR/${SEQ}_full_holdout_80_20_color_refinement_${STAMP}.log"

    echo "============================================================" | tee -a "$STATUS_FILE"
    echo "START: $SEQ full sequence + original color refinement" | tee -a "$STATUS_FILE"
    echo "TIME : $(date)" | tee -a "$STATUS_FILE"
    echo "LOG  : $LOG_FILE" | tee -a "$STATUS_FILE"
    echo "============================================================" | tee -a "$STATUS_FILE"

    START_SEC=$(date +%s)

    python run_tartanair_holdout_safe_entry.py \
        --config configs/stereo/tartanair/SE000.yaml \
        --sequence "$SEQ" \
        --holdout-every 5 \
        --holdout-offset 4 \
        --color-refinement \
        2>&1 | tee "$LOG_FILE"

    STATUS=${PIPESTATUS[0]}
    END_SEC=$(date +%s)
    TOTAL_WALL_SEC=$((END_SEC - START_SEC))

    echo "$(date)  $SEQ  exit_code=$STATUS  total_wall_sec=$TOTAL_WALL_SEC" | tee -a "$STATUS_FILE"

    if [ "$STATUS" -eq 0 ]; then
        ROW_FILE="$(auto_find_row "$SEQ")"
        if [ -n "$ROW_FILE" ] && [ -f "$ROW_FILE" ]; then
            python - "$ROW_FILE" "$SUMMARY_CSV" "$TOTAL_WALL_SEC" <<'PY'
import csv
import sys

row_file, out_file, total_wall = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(row_file, newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))

out = {
    "Sequence": row["Sequence"],
    "MaxMap": row["MaxMap"],
    "Train PSNR": row["Train PSNR"],
    "Train SSIM": row["Train SSIM"],
    "Test PSNR": row["Test PSNR"],
    "Test SSIM": row["Test SSIM"],
    "ATE(m)": row["ATE(m)"],
    "Gaussians": row["Gaussians"],
    "OnlineFPS": row["FPS"],
    "TotalWallSec": f"{total_wall:.0f}",
}
with open(out_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(out.keys()))
    writer.writerow(out)
PY
            echo "SUCCESS: $SEQ -> $ROW_FILE" | tee -a "$STATUS_FILE"
        else
            echo "WARNING: $SEQ completed but benchmark_row.csv was not found" | tee -a "$STATUS_FILE"
        fi
    else
        echo "FAILED: $SEQ (continuing to next sequence)" | tee -a "$STATUS_FILE"
    fi

done

echo "============================================================" | tee -a "$STATUS_FILE"
echo "Batch finished: $(date)" | tee -a "$STATUS_FILE"
echo "Summary CSV: $SUMMARY_CSV" | tee -a "$STATUS_FILE"
echo "Status log : $STATUS_FILE" | tee -a "$STATUS_FILE"
echo
cat "$SUMMARY_CSV"
