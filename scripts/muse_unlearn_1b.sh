#!/bin/bash

# Run MUSE unlearning on the 1B target produced by scripts/muse_finetune_1b.sh.
# Prereqs (per data_split):
#   saves/finetune/muse_${model}_${split}_target/best     <- target checkpoint
#   saves/finetune/muse_${model}_${split}_retrain/best    <- retrain checkpoint
#   saves/eval/muse_${model}_${split}_retrain/MUSE_EVAL.json <- privleak reference
# These are NOT compatible with the 7B muse-bench reference logs (different tokenizer / capacity);
# always pair a 1B unlearn run with the matching 1B retrain logs.

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
    if [ -f "$HF_TOKEN_FILE" ]; then
        export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
    fi
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set and no cached token found; private model downloads may fail."
fi

model=Llama-3.2-1B-Instruct
per_device_train_batch_size=2
gradient_accumulation_steps=1

data_splits=(
    "News"
    "Books"
)

trainers=(
    # "GradAscent"
    # "GradDiff"
    # "NPO"
    # "SimNPO"
    "LatentRMUParallelDDP"
)

for data_split in "${data_splits[@]}"; do
    target_ckpt=saves/finetune/muse_${model}_${data_split}_target/best
    retrain_logs=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json

    for trainer in "${trainers[@]}"; do
        task_name=muse_${model}_${data_split}_${trainer}_g8sc40a05w100_v2

        CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
            --config_file configs/accelerate/ddp_2gpu_config.yaml \
            --main_process_port $MASTER_PORT \
            src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/default.yaml \
            model=${model} \
            data_split=${data_split} \
            trainer=${trainer} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=${target_ckpt} \
            retain_logs_path=${retrain_logs} \
            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true
    done
done
