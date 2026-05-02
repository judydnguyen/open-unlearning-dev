#!/bin/bash
# Overnight hyperparameter sweep for LatentRMUParallelDDP on MUSE-News.
#
# Motivation: with gamma=alpha=1 the retain gradient (~0.7) swamps the forget
# gradient (~sc²/H ≈ 0.09 at sc=20), so forget_loss never moves. This sweep
# explores gamma and steering_coeff to balance the two signals, plus a full-mlp
# variant that gives the model more capacity to steer layer-16 activations.
#
# Sweep grid (6 runs × ~2 h each, all on MUSE-News):
#   tag              gamma  sc   alpha  trainable
#   g5_sc20          5      20   1.0    down_proj
#   g8_sc20          8      20   1.0    down_proj   (isolate gamma effect)
#   g8_sc40          8      40   1.0    down_proj   (new default)
#   g8_sc40_a05      8      40   0.5    down_proj   (reduce retain weight)
#   g10_sc80_a05     10     80   0.5    down_proj   (aggressive)
#   g8_sc40_fmlp     8      40   0.5    fullmlp     (more trainable capacity)
#
# Usage: bash scripts/muse_latent_rmu_sweep.sh 2>&1 | tee /tmp/muse_sweep.log

set -euo pipefail

source /home/judy/miniconda3/etc/profile.d/conda.sh
conda activate /data/judy/conda/envs/unlearning

export HF_HOME=/tank/home/judy/.cache/huggingface
export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi
if [ -z "${HF_TOKEN:-}" ]; then
    HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
    [ -f "$HF_TOKEN_FILE" ] && export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

model=Llama-2-7b-hf
data_split=News
per_device_train_batch_size=2
gradient_accumulation_steps=1

# Format: "tag:trainer_cfg:gamma:steering_coeff:alpha"
sweeps=(
    "g5_sc20:LatentRMUParallelDDP:5:20:1.0"
    "g8_sc20:LatentRMUParallelDDP:8:20:1.0"
    "g8_sc40:LatentRMUParallelDDP:8:40:1.0"
    "g8_sc40_a05:LatentRMUParallelDDP:8:40:0.5"
    "g10_sc80_a05:LatentRMUParallelDDP:10:80:0.5"
    "g8_sc40_fmlp:LatentRMUParallelDDP_fullmlp:8:40:0.5"
)

for entry in "${sweeps[@]}"; do
    IFS=':' read -r tag trainer gamma sc alpha <<< "$entry"

    task_name=muse_sweep_${model}_${data_split}_${tag}
    echo "=== Starting: ${task_name} (gamma=${gamma}, sc=${sc}, alpha=${alpha}, trainer=${trainer}) ==="

    CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
        --config_file configs/accelerate/ddp_2gpu_config.yaml \
        --main_process_port $MASTER_PORT \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default.yaml \
        model=${model} \
        data_split=${data_split} \
        trainer=${trainer} \
        task_name=${task_name} \
        retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.method_args.gamma=${gamma} \
        trainer.method_args.steering_coeff=${sc} \
        trainer.method_args.alpha=${alpha}

    echo "=== Done: ${task_name} ==="
done

echo "=== All sweep runs complete ==="
