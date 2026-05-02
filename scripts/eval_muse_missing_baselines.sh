#!/bin/bash
# Eval MUSE baseline runs that don't yet have an MUSE_SUMMARY.json.
# Walks every saves/unlearn/muse_${MODEL}_<split>_<method> directory, picks the
# best/last checkpoint, and runs eval/muse/default. Skips runs that are already
# evaluated and runs with no checkpoint to load.
#
# Usage:
#   bash scripts/eval_muse_missing_baselines.sh
#   DATA_SPLITS="News" bash scripts/eval_muse_missing_baselines.sh
#   EVAL_GPU=0 bash scripts/eval_muse_missing_baselines.sh

set -uo pipefail   # not -e: a bad ckpt shouldn't kill the whole sweep

export HF_HOME=/tank/home/judy/.cache/huggingface
export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi

EVAL_GPU="${EVAL_GPU:-1}"
model="${MODEL:-Llama-2-7b-hf}"
DATA_SPLITS="${DATA_SPLITS:-Books News}"

for data_split in $DATA_SPLITS; do
    retain_logs=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
    if [ ! -f "$retain_logs" ]; then
        echo "[warn] retain logs missing at $retain_logs (privleak will be off)"
    fi

    echo "========================================="
    echo "Split: $data_split"
    echo "========================================="

    for run_dir in saves/unlearn/muse_${model}_${data_split}_*; do
        [ -d "$run_dir" ] || continue
        task_name=$(basename "$run_dir")

        # Pick checkpoint: prefer best/, fall back to last/.
        # Accept either a single model.safetensors or a sharded
        # model.safetensors.index.json (7B+ models split into multiple files).
        has_ckpt() {
            [ -f "$1/model.safetensors" ] || [ -f "$1/model.safetensors.index.json" ] \
                || [ -f "$1/pytorch_model.bin" ] || [ -f "$1/pytorch_model.bin.index.json" ]
        }
        if   has_ckpt "$run_dir/best"; then ckpt_dir="$run_dir/best"
        elif has_ckpt "$run_dir/last"; then ckpt_dir="$run_dir/last"
        else
            echo "[skip] $task_name: no best/ or last/ checkpoint"
            continue
        fi

        out_dir="$run_dir/evals"

        # Already evaluated? — look anywhere under evals/ for MUSE_SUMMARY.json
        if find "$out_dir" -maxdepth 2 -name MUSE_SUMMARY.json 2>/dev/null | grep -q .; then
            echo "[done] $task_name"
            continue
        fi

        echo "----- Eval $task_name from $ckpt_dir → $out_dir"
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
            experiment=eval/muse/default.yaml \
            data_split=${data_split} \
            task_name=${task_name} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path="$ckpt_dir" \
            paths.output_dir="$out_dir" \
            retain_logs_path="$retain_logs"
    done
done

echo "=== Done. Summaries: ==="
echo "  for f in saves/unlearn/muse_${model}_*/evals/**/MUSE_SUMMARY.json; do echo \$f; done"
