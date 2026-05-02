#!/bin/bash

# Finetune Llama-3.2-3B-Instruct on MUSE News/Books to produce:
#   1. target  (data_sub_set=full)   -> drop-in replacement for muse-bench/MUSE-${split}_target
#   2. retrain (data_sub_set=retain) -> reference model for privleak / retain-quality metrics
# Then eval the retrain checkpoints so retain_logs_path is populated for unlearning runs.
#
# Why 3B instead of 1B: the 1B target hit a capacity ceiling on KnowMem ROUGE (~0.20
# even with train loss ~0.36). 3B should land closer to the published 7B reference
# (KnowMem retain ~0.55). Same lr=5e-5 / 10 epoch recipe — empirically reasonable for
# this size on the MUSE corpus.

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

model=Llama-3.2-3B-Instruct
# Same recipe as the 1B run that finally fit (loss 2.7→0.36, exact_mem 0.93).
# 3B has ~3x params; effective batch 16, lr 5e-5, 10 epochs is a safe starting point.
# If train loss plateaus above ~0.5 at 10 ep, bump epochs. If it diverges early, drop lr to 3e-5.
per_device_train_batch_size=2     # 3B fits less per device
gradient_accumulation_steps=4     # effective batch = 2 * 4 * 2 GPUs = 16
learning_rate=5e-5
num_train_epochs=10

# DDP setup — bf16 + gradient checkpointing for 3B.
TRAIN_GPUS="0,1"
EVAL_GPU="0"
ACCEL_CFG=configs/accelerate/ddp_2gpu_config.yaml

data_splits=(
    # "News"
    "Books"
)

for data_split in "${data_splits[@]}"; do

    # ---------- 1. target finetune (full) ----------
    target_task=muse_${model}_${data_split}_target
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
        --config_file $ACCEL_CFG \
        --main_process_port $MASTER_PORT \
        src/train.py \
        experiment=finetune/muse/default.yaml \
        model=${model} \
        data_split=${data_split} \
        data_sub_set=full \
        task_name=${target_task} \
        eval.muse.data_split=${data_split} \
        ~eval.muse.metrics.mia_reference \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.learning_rate=${learning_rate} \
        trainer.args.num_train_epochs=${num_train_epochs} \
        trainer.args.gradient_checkpointing=true \
        '+trainer.args.gradient_checkpointing_kwargs={use_reentrant: false}' \
        trainer.args.ddp_find_unused_parameters=true

    # ---------- 2. retrain finetune (retain-only, used as privleak reference) ----------
    retrain_task=muse_${model}_${data_split}_retrain
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
        --config_file $ACCEL_CFG \
        --main_process_port $MASTER_PORT \
        src/train.py \
        experiment=finetune/muse/default.yaml \
        model=${model} \
        data_split=${data_split} \
        data_sub_set=retain \
        task_name=${retrain_task} \
        eval.muse.data_split=${data_split} \
        ~eval.muse.metrics.mia_reference \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.learning_rate=${learning_rate} \
        trainer.args.num_train_epochs=${num_train_epochs} \
        trainer.args.gradient_checkpointing=true \
        '+trainer.args.gradient_checkpointing_kwargs={use_reentrant: false}' \
        trainer.args.ddp_find_unused_parameters=true

    # ---------- 3. eval target -> sanity check the finetune fit ----------
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${data_split} \
        task_name=${target_task} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/${target_task}/last \
        eval.muse.data_split=${data_split} \
        ~eval.muse.metrics.mia_reference

    # ---------- 4. eval retrain -> MUSE_EVAL.json (retain_logs reference for privleak) ----------
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${data_split} \
        task_name=${retrain_task} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/${retrain_task}/last \
        eval.muse.data_split=${data_split} \
        ~eval.muse.metrics.mia_reference

done
