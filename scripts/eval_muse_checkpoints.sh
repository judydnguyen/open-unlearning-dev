#!/bin/bash

# Evaluate every checkpoint-* subdir of an unlearn run against MUSE.
# Per-checkpoint evals land in ${save_dir}/evals_checkpoint-{step}/ so the
# final-model eval at ${save_dir}/evals/ is preserved.
#
# Usage:
#   bash scripts/eval_muse_checkpoints.sh <save_dir> <data_split> [model] [eval_gpu]
# Example:
#   bash scripts/eval_muse_checkpoints.sh \
#       saves/unlearn/muse_Llama-3.2-3B-Instruct_News_LatentRMUParallelDDP_DDP_v2 \
#       News

set -e

export HF_HOME=/tank/home/judy/.cache/huggingface
export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAVE_DIR=${1:?save_dir required}
DATA_SPLIT=${2:?data_split required (News|Books)}
MODEL=${3:-Llama-3.2-3B-Instruct}
EVAL_GPU=${4:-1}

RETAIN_LOGS=saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json

if [ ! -f "$RETAIN_LOGS" ]; then
    echo "[warn] retain logs missing at $RETAIN_LOGS (privleak will be off)"
fi

shopt -s nullglob
ckpts=("$SAVE_DIR"/checkpoint-*)
if [ ${#ckpts[@]} -eq 0 ]; then
    echo "[err] no checkpoint-* dirs in $SAVE_DIR"
    exit 1
fi

for ckpt in "${ckpts[@]}"; do
    step=$(basename "$ckpt" | sed 's/checkpoint-//')
    out_dir="$SAVE_DIR/evals_checkpoint-${step}"

    if [ -f "$out_dir/MUSE_SUMMARY.json" ]; then
        echo "[done] checkpoint-${step}: $out_dir/MUSE_SUMMARY.json"
        continue
    fi

    if [ ! -f "$ckpt/model.safetensors" ] && [ ! -f "$ckpt/model.safetensors.index.json" ]; then
        echo "[skip] checkpoint-${step}: no model weights in $ckpt"
        continue
    fi

    task_name=$(basename "$SAVE_DIR")_ckpt${step}
    echo "============================================="
    echo "Eval checkpoint-${step}"
    echo "  ckpt    = $ckpt"
    echo "  out_dir = $out_dir"
    echo "============================================="

    CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${task_name} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${ckpt} \
        paths.output_dir=${out_dir} \
        retain_logs_path=${RETAIN_LOGS}
done

echo "=== All checkpoint evals done. Compare with: ==="
echo "  for f in $SAVE_DIR/evals_checkpoint-*/MUSE_SUMMARY.json $SAVE_DIR/evals/MUSE_SUMMARY.json; do echo \$f; cat \$f; echo; done"
