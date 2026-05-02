#!/bin/bash
# Backfill MIA metrics on every existing MUSE unlearn run that's missing them.
#
# Why: the unlearn evals were originally run with the slim metric suite (5 fields:
# forget/retain ROUGE + extraction_strength + privleak). The reference target/
# retrain models include 6 extra MIA-style metrics (exact_memorization, mia_loss,
# mia_gradnorm, mia_zlib, mia_min_k, mia_min_k_plus_plus, mia_reference). With
# muse.yaml now enabling all of them by default, this script re-runs eval on
# every {model}/last/ checkpoint that's missing the new fields. The eval cache
# in MUSE_EVAL.json skips already-computed metrics, so only the new ones are
# actually computed.
#
# Usage:
#   bash scripts/backfill_muse_mia.sh
#   MODEL=Llama-2-7b-hf SPLITS="News" bash scripts/backfill_muse_mia.sh
#   DRY_RUN=1 bash scripts/backfill_muse_mia.sh    # list what would run

set -uo pipefail

export HF_HOME=/tank/home/judy/.cache/huggingface
export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi

MODEL="${MODEL:-Llama-2-7b-hf}"
SPLITS="${SPLITS:-Books News}"
EVAL_GPU="${EVAL_GPU:-1}"
PYTHON="${PYTHON:-/home/judy/miniconda3/envs/unlearning/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

# Sentinel field that proves MIA was computed last time. If absent in
# MUSE_SUMMARY.json, we re-run eval to fill in the MIA suite.
SENTINEL='"mia_loss"'

count_run=0
count_skip=0

for split in $SPLITS; do
    retain_logs="saves/eval/muse_${MODEL}_${split}_retrain/MUSE_EVAL.json"
    if [ ! -f "$retain_logs" ]; then
        echo "[warn] missing retain logs for $split: $retain_logs"
    fi

    for task_dir in saves/unlearn/muse_${MODEL}_${split}_*; do
        [ -d "$task_dir" ] || continue
        task_name="$(basename "$task_dir")"
        ckpt_dir="$task_dir/last"
        out_dir="$task_dir/evals"
        summary="$out_dir/MUSE_SUMMARY.json"

        if [ ! -d "$ckpt_dir" ]; then
            echo "[skip] $task_name — no last/ checkpoint"
            count_skip=$((count_skip+1))
            continue
        fi

        if [ -f "$summary" ] && grep -q "$SENTINEL" "$summary"; then
            echo "[done] $task_name — MIA already present"
            count_skip=$((count_skip+1))
            continue
        fi

        echo "[run]  $task_name"
        if [ "$DRY_RUN" = "1" ]; then
            count_run=$((count_run+1))
            continue
        fi

        CUDA_VISIBLE_DEVICES=$EVAL_GPU $PYTHON src/eval.py \
            experiment=eval/muse/default.yaml \
            data_split=${split} \
            task_name=${task_name} \
            model=${MODEL} \
            model.model_args.pretrained_model_name_or_path="$ckpt_dir" \
            paths.output_dir="$out_dir" \
            retain_logs_path="$retain_logs"

        count_run=$((count_run+1))
    done
done

echo "=== Done. ran=$count_run, skipped=$count_skip ==="
