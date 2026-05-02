#!/bin/bash
# Smoke test for LatentRMUParallel (rmu_encoder_parallel.py)
# Runs 1 epoch Phase 1 + 1 epoch Phase 2 with max 3 steps to verify the
# ZeRO Stage 3 two-phase training pipeline works end-to-end.
# Usage: bash scripts/smoke_latent_rmu_parallel.sh 2>&1 | tee /tmp/smoke_latent_rmu.log

set -euo pipefail

# source /home/judy/miniconda3/etc/profile.d/conda.sh
# conda activate /data/judy/conda/envs/unlearning

export HF_HOME=/tank/home/judy/.cache/huggingface
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

if [ -z "${HF_TOKEN:-}" ]; then
    HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
    if [ -f "$HF_TOKEN_FILE" ]; then
        export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
    fi
fi

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --config_file configs/accelerate/default_config.yaml \
    --main_process_port $MASTER_PORT \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/muse/default.yaml \
    model=Llama-2-7b-hf \
    data_split=News \
    trainer=LatentRMUParallel \
    task_name=smoke_muse_LatentRMUParallel \
    retain_logs_path=saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.num_train_epochs=2 \
    +trainer.args.max_steps=3 \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=false \
    trainer.method_args.encoder_epochs=1 \
    trainer.method_args.encoder_lr=1e-3 \
    trainer.method_args.steering_coeff=20 \
    trainer.method_args.latent_dim=256 \
    "trainer.method_args.module_regex=model\\.layers\\.7"

echo "=== Smoke test complete ==="
