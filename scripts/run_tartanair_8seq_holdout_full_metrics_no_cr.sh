#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-stereo}"
if [[ "$MODE" != "stereo" && "$MODE" != "mono" ]]; then
    echo "Usage: bash scripts/run_tartanair_8seq_holdout_full_metrics_no_cr.sh [stereo|mono]"
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

SEQUENCES=(SE000 SE001 SE002 SE003 SH000 SH001 SH002 SH003)

if [[ "$MODE" == "stereo" ]]; then
    ENTRY="run_tartanair_holdout_safe_entry.py"
    CONFIG="configs/stereo/tartanair/SE000.yaml"
    RESULT_PREFIX="tartanair_stereo_"
else
    ENTRY="run_tartanair_mono_holdout_safe_entry.py"
    CONFIG="configs/mono/tartanair/SE000.yaml"
    RESULT_PREFIX="tartanair_mono_"
fi

LOG_DIR="$ROOT_DIR/batch_logs/${MODE}_8seq_holdout_full_metrics_no_color_refinement"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
STATUS_FILE="$LOG_DIR/status_${STAMP}.txt"
SUMMARY_CSV="$LOG_DIR/summary_${MODE}_SE000_SE003_SH000_SH003_no_cr_${STAMP}.csv"

HEADER="Sequence,MaxMap,Train PSNR,Train SSIM,Train LPIPS,Test PSNR,Test SSIM,Test LPIPS,ATE(m),Gaussians,Online FPS,Online Time(s),Offline Refine(s),System Total(s)"
echo "$HEADER" > "$SUMMARY_CSV"

echo "MonoGS ${MODE} 8-sequence w/o color refinement benchmark started: $(date)" | tee "$STATUS_FILE"
echo "Protocol: 80/20 holdout + no color refinement + SE(3) ATE" | tee -a "$STATUS_FILE"
echo "Timing: Online Time excludes final metric evaluation; Offline Refine=0; System Total=Online Time" | tee -a "$STATUS_FILE"

auto_find_row() {
    local seq="$1"
    find "$ROOT_DIR/results/${RESULT_PREFIX}${seq}" \
        -type f \
        -path "*/range_000000_end/holdout_5_4/*/benchmark_row.csv" \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

for SEQ in "${SEQUENCES[@]}"; do
    LOG_FILE="$LOG_DIR/${SEQ}_${MODE}_full_metrics_no_cr_${STAMP}.log"

    echo "============================================================" | tee -a "$STATUS_FILE"
    echo "START: $MODE $SEQ full sequence, w/o color refinement" | tee -a "$STATUS_FILE"
    echo "TIME : $(date)" | tee -a "$STATUS_FILE"
    echo "LOG  : $LOG_FILE" | tee -a "$STATUS_FILE"
    echo "============================================================" | tee -a "$STATUS_FILE"

    python "$ENTRY" \
        --config "$CONFIG" \
        --sequence "$SEQ" \
        --holdout-every 5 \
        --holdout-offset 4 \
        2>&1 | tee "$LOG_FILE"

    STATUS=${PIPESTATUS[0]}
    echo "$(date)  $SEQ  exit_code=$STATUS" | tee -a "$STATUS_FILE"

    if [[ "$STATUS" -eq 0 ]]; then
        ROW_FILE="$(auto_find_row "$SEQ")"
        if [[ -n "$ROW_FILE" && -f "$ROW_FILE" ]]; then
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
