#!/bin/bash
# Relearning robustness experiment.
# For each unlearned checkpoint, fine-tune on the forget set for 3 epochs,
# saving every 10 steps. Then evaluate each checkpoint to measure recovery
# of forgotten knowledge. Methods that recover slowest are most robust.

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
    "LatentRMU saves/unlearn/ablation/LatentRMU_full_forget01/best"
    # "RMU saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_RMU"
    # "GradAscent saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradAscent"
    # "GradDiff saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradDiff"
    # "NPO saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_NPO"
    # "SimNPO saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_SimNPO"
)

for entry in "${methods[@]}"; do
    method=$(echo "$entry" | cut -d' ' -f1)
    checkpoint=$(echo "$entry" | cut -d' ' -f2-)
    task_name="relearn_${method}_${forget_split}"

    echo "=== Relearning attack: ${method} ==="
    echo "    Starting from: ${checkpoint}"
    echo "    Task name:     ${task_name}"

    # Phase 1: fine-tune on forget set starting from unlearned checkpoint
    CUDA_VISIBLE_DEVICES=1 python src/train.py \
        experiment=robustness/relearn_tofu.yaml \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=${checkpoint} \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        task_name=${task_name}

    relearn_save_dir="saves/finetune/${task_name}"

    # Phase 2: eval every saved checkpoint (step 0, 10, 20, ...)
    for ckpt_dir in "${relearn_save_dir}"/checkpoint-*; do
        [ -d "$ckpt_dir" ] || continue
        # skip trainer-state-only dirs (no model weights)
        [ -f "${ckpt_dir}/model.safetensors" ] || [ -f "${ckpt_dir}/pytorch_model.bin" ] || continue
        step=$(basename "$ckpt_dir" | sed 's/checkpoint-//')
        echo "  Evaluating checkpoint-${step} ..."
        CUDA_VISIBLE_DEVICES=1 python src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            model=${model} \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=${ckpt_dir} \
            paths.output_dir=${relearn_save_dir}/evals/step_${step} \
            retain_logs_path=${retain_logs_path} \
            eval.tofu.overwrite=true
    done
done
