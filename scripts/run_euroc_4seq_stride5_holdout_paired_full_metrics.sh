#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

SEQUENCES=(MH02 V101 V201 MH05)
ENTRY="run_euroc_holdout_paired_entry.py"
CONFIG="configs/stereo/euroc/benchmark_stride5.yaml"
LOG_DIR="$ROOT_DIR/batch_logs/euroc_4seq_stride5_holdout_paired_full_metrics"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
STATUS_FILE="$LOG_DIR/status_${STAMP}.txt"
RAW_CSV="$LOG_DIR/raw_euroc_MH02_V101_V201_MH05_stride5_paired_${STAMP}.csv"
PAPER_CSV="$LOG_DIR/paper_euroc_MH02_V101_V201_MH05_stride5_paired_${STAMP}.csv"

RAW_HEADER="Variant,Sequence,TotalFrames,MaxMap,Train PSNR,Train SSIM,Train LPIPS,Test PSNR,Test SSIM,Test LPIPS,ATE(m),Gaussians,Online FPS,Online Time(s),Offline Refine(s),System Total(s)"
echo "$RAW_HEADER" > "$RAW_CSV"
echo "Sequence,Method,MaxMap,RMSE SE(3)↓,Train PSNR/SSIM/LPIPS,Test PSNR/SSIM/LPIPS,FPS,Time(s),# G" > "$PAPER_CSV"

declare -A RESULT_DIRS=(
    [MH02]="MH_02_easy"
    [V101]="V1_01_easy"
    [V201]="V2_01_easy"
    [MH05]="MH_05_difficult"
)

echo "MonoGS EuRoC paired paper benchmark started: $(date)" | tee "$STATUS_FILE"
echo "Sequences: ${SEQUENCES[*]}" | tee -a "$STATUS_FILE"
echo "Protocol: raw EuRoC -> stride=5 -> retained-sequence 8:2 holdout -> one online run -> exact online-final snapshot -> released 26000-iter color refinement -> paired pre/post evaluation" | tee -a "$STATUS_FILE"
echo "ATE: SE(3)" | tee -a "$STATUS_FILE"
echo "Timing: metric rendering and snapshot I/O are excluded; w/o CR time=online; +CR time=online+offline refinement" | tee -a "$STATUS_FILE"

find_pair_file() {
    local seq="$1"
    local dirname="${RESULT_DIRS[$seq]}"
    find "$ROOT_DIR/results/euroc_${dirname}" \
        -type f \
        -path "*/range_000000_end/holdout_5_4_paired_color_refinement/*/paired_benchmark_rows.csv" \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

for SEQ in "${SEQUENCES[@]}"; do
    LOG_FILE="$LOG_DIR/${SEQ}_stereo_stride5_paired_${STAMP}.log"

    echo "============================================================" | tee -a "$STATUS_FILE"
    echo "START: EuRoC $SEQ" | tee -a "$STATUS_FILE"
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

    if [[ "$STATUS" -ne 0 ]]; then
        echo "FAILED: $SEQ (continuing to next sequence)" | tee -a "$STATUS_FILE"
        continue
    fi

    PAIR_FILE="$(find_pair_file "$SEQ")"
    if [[ -z "$PAIR_FILE" || ! -f "$PAIR_FILE" ]]; then
        echo "WARNING: $SEQ completed but paired_benchmark_rows.csv was not found" | tee -a "$STATUS_FILE"
        continue
    fi

    python - "$PAIR_FILE" "$RAW_CSV" "$PAPER_CSV" "$SEQ" <<'PY'
import csv
import sys

pair_file, raw_file, paper_file, sequence_label = sys.argv[1:5]

with open(pair_file, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

raw_fields = [
    "Variant", "Sequence", "TotalFrames", "MaxMap",
    "Train PSNR", "Train SSIM", "Train LPIPS",
    "Test PSNR", "Test SSIM", "Test LPIPS",
    "ATE(m)", "Gaussians", "Online FPS", "Online Time(s)",
    "Offline Refine(s)", "System Total(s)",
]

with open(raw_file, "a", newline="", encoding="utf-8") as f_raw, open(
    paper_file, "a", newline="", encoding="utf-8"
) as f_paper:
    raw_writer = csv.DictWriter(f_raw, fieldnames=raw_fields)
    paper_writer = csv.writer(f_paper)

    for r in rows:
        r = dict(r)
        r["Sequence"] = sequence_label
        raw_writer.writerow({k: r[k] for k in raw_fields})

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

        paper_writer.writerow([
            sequence_label,
            f"MonoGS-Stereo ({variant})",
            f"{max_map_pct:.2f}%",
            f'{float(r["ATE(m)"]):.4f} m',
            train,
            test,
            fps_text,
            time_text,
            f"{gaussians / 1000.0:.1f}k",
        ])
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
