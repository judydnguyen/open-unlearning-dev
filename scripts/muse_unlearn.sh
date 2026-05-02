#!/bin/bash

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


per_device_train_batch_size=2
gradient_accumulation_steps=1

model=Llama-2-7b-hf

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

# #########################################################
# #################### MUSE Unlearning ####################
# #########################################################


for data_split in "${data_splits[@]}"; do
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
        retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true

        # CUDA_VISIBLE_DEVICES=1 python src/eval.py \
        # experiment=eval/muse/default.yaml \
        # data_split=${data_split} \
        # task_name=${task_name} \
        # model=${model} \
        # model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}/last \
        # paths.output_dir=saves/unlearn/${task_name}/evals \
        # retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
    done
done
