#!/bin/bash
# Privacy and security superiority evaluation.
# Runs the full MIA suite (mia_loss, mia_min_k, mia_min_k_plus_plus,
# mia_gradnorm, mia_zlib) plus memorization metrics (exact_memorization,
# extraction_strength, privleak) across all unlearning methods.
# Results in saves/privacy/<method>/evals/TOFU_SUMMARY.json.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
else
    echo "WARNING: .env not found — gated models may fail"
fi

export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune

model="Llama-3.2-1B-Instruct"
forget_split="forget01"
holdout_split="holdout01"
retain_split="retain99"
retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

# method_name  unlearned_checkpoint
methods=(
    "LatentRMU_v4.8 saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_LatentRMU_v4.8_sweep/last"
    "RMU saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_RMU"
    "GradAscent saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradAscent"
    "GradDiff saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradDiff"
    "NPO saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_NPO"
    "SimNPO saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_SimNPO"
)

for entry in "${methods[@]}"; do
    method=$(echo "$entry" | cut -d' ' -f1)
    checkpoint=$(echo "$entry" | cut -d' ' -f2-)
    task_name="privacy_${method}_${forget_split}"
    out_dir="saves/privacy/${method}/evals"

    echo "=== Privacy eval: ${method} ==="
    echo "    Checkpoint: ${checkpoint}"

    if [ -f "${out_dir}/TOFU_SUMMARY.json" ]; then
        echo "    Skipping — results already exist at ${out_dir}/TOFU_SUMMARY.json"
        continue
    fi

    CUDA_VISIBLE_DEVICES=1 python src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        eval=tofu_full_privacy \
        model=${model} \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${checkpoint} \
        paths.output_dir=${out_dir} \
        retain_logs_path=${retain_logs_path} \
        eval.tofu.overwrite=true

    echo "    Done. Results: ${out_dir}/TOFU_SUMMARY.json"
done
