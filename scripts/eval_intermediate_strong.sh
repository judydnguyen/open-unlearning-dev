#!/bin/bash
# Eval intermediate checkpoints (and 'last') for a LatentRMUParallelDDP_strong
# training run. Mirrors the eval block of muse_unlearn_strong.sh but loops over
# every checkpoint-* dir and writes results to evals/<ckpt_name>/.
#
# Usage:
#   bash scripts/eval_intermediate_strong.sh                # use defaults below
#   DATA_SPLITS="News" bash scripts/eval_intermediate_strong.sh
#   TAG="g20sc60a03ce1w100_v1" CKPT_FILTER="checkpoint-816 last" \
#       bash scripts/eval_intermediate_strong.sh

set -uo pipefail   # not -e: a bad checkpoint shouldn't kill the whole sweep

export HF_HOME=/tank/home/judy/.cache/huggingface
export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi

EVAL_GPU="${EVAL_GPU:-1}"
model="${MODEL:-Llama-2-7b-hf}"
trainer="${TRAINER:-LatentRMUParallelDDP_strong}"
TAG="${TAG:-g20sc60a03ce1w100_v1}"

# Space-separated list of splits to eval. Default mirrors the train script.
DATA_SPLITS="${DATA_SPLITS:-News}"

# Optional whitelist: space-separated subset of checkpoint dir names to eval.
# Example: CKPT_FILTER="checkpoint-816 last". Empty = eval everything found.
CKPT_FILTER="${CKPT_FILTER:-}"

for data_split in $DATA_SPLITS; do
    task_name=muse_${model}_${data_split}_${trainer}_${TAG}
    run_dir=saves/unlearn/${task_name}
    retain_logs=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json

    if [ ! -d "$run_dir" ]; then
        echo "[skip] $task_name: $run_dir not found"
        continue
    fi
    if [ ! -f "$retain_logs" ]; then
        echo "[warn] $task_name: retain logs missing at $retain_logs"
    fi

    # Discover candidate checkpoints in numeric order, then append 'last' if it exists.
    candidates=()
    while IFS= read -r d; do
        candidates+=("$(basename "$d")")
    done < <(find "$run_dir" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' \
             | sort -t- -k2 -n)
    [ -d "$run_dir/last" ] && candidates+=("last")

    if [ "${#candidates[@]}" -eq 0 ]; then
        echo "[skip] $task_name: no checkpoints found under $run_dir"
        continue
    fi

    echo "========================================="
    echo "Run: $task_name"
    echo "Found: ${candidates[*]}"
    [ -n "$CKPT_FILTER" ] && echo "Filter: $CKPT_FILTER"
    echo "========================================="

    for ckpt in "${candidates[@]}"; do
        if [ -n "$CKPT_FILTER" ] && ! [[ " $CKPT_FILTER " == *" $ckpt "* ]]; then
            continue
        fi

        ckpt_dir="$run_dir/$ckpt"
        out_dir="$run_dir/evals/$ckpt"

        if [ -f "$out_dir/MUSE_SUMMARY.json" ]; then
            echo "[done] $ckpt already evaluated → $out_dir/MUSE_SUMMARY.json"
            continue
        fi

        echo "----- Eval $ckpt → $out_dir"
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
            experiment=eval/muse/default.yaml \
            data_split=${data_split} \
            task_name=${task_name}_${ckpt} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path="$ckpt_dir" \
            paths.output_dir="$out_dir" \
            retain_logs_path="$retain_logs"
    done
done

echo "=== Done. Compare with: ==="
echo "  for f in saves/unlearn/muse_${model}_*_${trainer}_${TAG}/evals/*/MUSE_SUMMARY.json; do echo \$f; cat \$f; echo; done"
