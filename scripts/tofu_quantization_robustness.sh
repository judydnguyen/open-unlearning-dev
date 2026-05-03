#!/bin/bash
# Quantization robustness experiment.
# For each unlearned checkpoint, evaluate at fp16 (baseline), int8, and int4
# precision. A large jump in forget_Q_A_Prob after quantization means
# the unlearning was not quantization-robust. LatentRMU should show the
# smallest degradation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
else
    echo "WARNING: .env not found — gated models may fail"
fi

PYTHON="${PYTHON:-/home/judy/miniconda3/envs/unlearning/bin/python}"
export MASTER_PORT=$($PYTHON -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
export HF_HOME=/tank/home/judy/.cache/huggingface
export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune


forget_split="forget01"
holdout_split="holdout01"
retain_split="retain99"
base_model="Llama-3.2-1B-Instruct"
retain_logs_path="saves/eval/tofu_${base_model}_${retain_split}/TOFU_EVAL.json"

# method_name  unlearned_checkpoint
methods=(
    "LatentRMU saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_LatentRMU_v4.8_sweep_g0.50/best"
    # "RMU saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_RMU"
    # "GradAscent saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradAscent"
    # "GradDiff saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradDiff"
    # "NPO saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_NPO"
    # "SimNPO saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_SimNPO"
)

# precision_tag  model_config
precisions=(
    "fp16 ${base_model}"
    "int8 ${base_model}-int8"
    "int4 ${base_model}-int4"
)

for entry in "${methods[@]}"; do
    method=$(echo "$entry" | cut -d' ' -f1)
    checkpoint=$(echo "$entry" | cut -d' ' -f2-)

    for prec_entry in "${precisions[@]}"; do
        prec_tag=$(echo "$prec_entry" | cut -d' ' -f1)
        model_cfg=$(echo "$prec_entry" | cut -d' ' -f2)
        task_name="quant_${method}_${prec_tag}_${forget_split}"
        out_dir="saves/robustness/${task_name}/evals"

        echo "=== Quantization eval: ${method} @ ${prec_tag} ==="

        CUDA_VISIBLE_DEVICES=1 python src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            model=${model_cfg} \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=${checkpoint} \
            paths.output_dir=${out_dir} \
            retain_logs_path=${retain_logs_path} \
            eval.tofu.overwrite=true
    done
done
