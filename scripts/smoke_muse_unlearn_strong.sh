#!/bin/bash
# Smoke-test version of muse_unlearn_strong.sh.
# Caps each phase via max_steps so the full pipeline (phase 1 → phase 2 → eval)
# completes in a few minutes per data split. Intended to surface DDP / autograd /
# OOM issues without committing to a full sweep.
#
# Usage: bash scripts/smoke_muse_unlearn_strong.sh 2>&1 | tee /tmp/smoke_strong.log

set -euo pipefail

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
    if [ -f "$HF_TOKEN_FILE" ]; then
        export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
    fi
fi

per_device_train_batch_size=1
gradient_accumulation_steps=2
TRAIN_GPUS="0,1"
EVAL_GPU="1"

model=Llama-2-7b-hf
trainer=LatentRMUParallelDDP_strong
TAG="smoke_v1"

# Smoke knobs:
#   max_steps caps EACH phase (HF Trainer applies args.max_steps per super().train()
#   call, and our trainer calls super().train() once per phase). With encoder_epochs=1
#   and max_steps=3, you get 3 steps phase 1 + 3 steps phase 2.
SMOKE_MAX_STEPS=3
SMOKE_ENCODER_EPOCHS=1
# num_train_epochs must be > encoder_epochs so phase 2 has something to do
# (its epoch count = num_train_epochs - encoder_epochs).
SMOKE_NUM_TRAIN_EPOCHS=2

data_splits=(
    "News"
    "Books"
)

for data_split in "${data_splits[@]}"; do
    task_name=smoke_muse_${model}_${data_split}_${trainer}_${TAG}

    echo "========================================="
    echo "Smoke training: $task_name"
    echo "GPUs: $TRAIN_GPUS  max_steps/phase: $SMOKE_MAX_STEPS"
    echo "========================================="

    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
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
        trainer.args.num_train_epochs=${SMOKE_NUM_TRAIN_EPOCHS} \
        +trainer.args.max_steps=${SMOKE_MAX_STEPS} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.method_args.encoder_epochs=${SMOKE_ENCODER_EPOCHS}

    if [ $? -ne 0 ]; then
        echo "Smoke training failed for $task_name — skipping eval."
        continue
    fi

    echo "========================================="
    echo "Smoke eval: $task_name"
    echo "GPU: $EVAL_GPU"
    echo "========================================="

    CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${data_split} \
        task_name=${task_name} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}/last \
        paths.output_dir=saves/unlearn/${task_name}/evals \
        retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
done

echo "=== Smoke pipeline complete ==="
