#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

SEQUENCES=(SH000 SH001 SH002 SH003)
LOG_DIR="$ROOT_DIR/batch_logs/holdout_80_20"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
STATUS_FILE="$LOG_DIR/status_${STAMP}.txt"
SUMMARY_CSV="$LOG_DIR/summary_SH000_SH003_full_holdout_80_20_${STAMP}.csv"

echo "Sequence,MaxMap,Train PSNR,Train SSIM,Test PSNR,Test SSIM,ATE(m),FPS,Gaussians" > "$SUMMARY_CSV"

echo "Batch started: $(date)" | tee "$STATUS_FILE"

auto_find_row() {
    local seq="$1"
    find "$ROOT_DIR/results/tartanair_stereo_${seq}" \
        -type f \
        -path "*/range_000000_end/holdout_5_4/*/benchmark_row.csv" \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

for SEQ in "${SEQUENCES[@]}"; do
    LOG_FILE="$LOG_DIR/${SEQ}_full_holdout_80_20_${STAMP}.log"

    echo "============================================================" | tee -a "$STATUS_FILE"
    echo "START: $SEQ full sequence, train:test=4:1" | tee -a "$STATUS_FILE"
    echo "TIME : $(date)" | tee -a "$STATUS_FILE"
    echo "LOG  : $LOG_FILE" | tee -a "$STATUS_FILE"
    echo "============================================================" | tee -a "$STATUS_FILE"

    python run_tartanair_holdout.py \
        --config configs/stereo/tartanair/SE000.yaml \
        --sequence "$SEQ" \
        --holdout-every 5 \
        --holdout-offset 4 \
        2>&1 | tee "$LOG_FILE"

    STATUS=${PIPESTATUS[0]}
    echo "$(date)  $SEQ  exit_code=$STATUS" | tee -a "$STATUS_FILE"

    if [ "$STATUS" -eq 0 ]; then
        ROW_FILE="$(auto_find_row "$SEQ")"
        if [ -n "$ROW_FILE" ] && [ -f "$ROW_FILE" ]; then
            tail -n 1 "$ROW_FILE" >> "$SUMMARY_CSV"
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
