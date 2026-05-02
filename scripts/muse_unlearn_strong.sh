#!/bin/bash
# MUSE Books (primary) and News (cross-check) sweep using LatentRMU with
# milder forget pressure than v1 (which over-unlearned: News retain went to 0).
# Strong-v2 setting (baked into trainer=LatentRMUParallelDDP_strong_v2):
#   - Edit surface: layers 10-20 (capped at module_regex)
#   - Steering hook: layer 20
#   - gamma=10, steering_coeff=30, alpha=0.7
# Save every epoch (limit 10 = keep all) so we can pick the best checkpoint.

# source /home/judy/miniconda3/etc/profile.d/conda.sh
# conda activate /data/judy/conda/envs/unlearning

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
trainer=LatentRMUParallelDDP_strong_v2
TAG="g10sc30a07_v2"

data_splits=(
    "Books"
    "News"
)

for data_split in "${data_splits[@]}"; do
    task_name=muse_${model}_${data_split}_${trainer}_${TAG}

    echo "========================================="
    echo "Training: $task_name"
    echo "GPUs: $TRAIN_GPUS"
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
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.save_strategy=epoch \
        +trainer.args.save_total_limit=10 \
        trainer.args.save_only_model=true

    if [ $? -ne 0 ]; then
        echo "Training failed for $task_name — skipping eval."
        continue
    fi

    echo "========================================="
    echo "Evaluating: $task_name"
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