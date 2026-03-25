#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Number of GPUs to use — adjust to match your hardware
NUM_GPUS=4
# GPU IDs — adjust as needed (e.g., "0,1,2,3")
CUDA_IDS="0,1,2,3"

model="Llama-3.1-8B-Instruct"  # Model name (must match config/model_args.yaml)

# Effective batch size = per_device_train_batch_size * gradient_accumulation_steps * NUM_GPUS
# 1 * 8 * 4 = 32
per_device_train_batch_size=1
gradient_accumulation_steps=8

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)


########################################################################################################################
########################################### RETAIN Finetuned TOFU ######################################################
########################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    CUDA_VISIBLE_DEVICES=${CUDA_IDS} accelerate launch \
        --config_file configs/accelerate/default_config.yaml \
        --main_process_port $MASTER_PORT \
        --num_processes ${NUM_GPUS} \
        src/train.py experiment=finetune/tofu/default.yaml \
        task_name=tofu_${model}_${retain_split} \
        model=${model} \
        data/datasets@data.train=TOFU_QA_retain \
        data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true

    CUDA_VISIBLE_DEVICES=${CUDA_IDS} accelerate launch \
        --config_file configs/accelerate/default_config.yaml \
        --main_process_port $MASTER_PORT \
        --num_processes ${NUM_GPUS} \
        src/eval.py experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        task_name=tofu_${model}_${retain_split} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_${retain_split}
done


########################################################################################################################
########################################### FULL Finetuned TOFU ########################################################
########################################################################################################################

CUDA_VISIBLE_DEVICES=${CUDA_IDS} accelerate launch \
    --config_file configs/accelerate/default_config.yaml \
    --main_process_port $MASTER_PORT \
    --num_processes ${NUM_GPUS} \
    src/train.py experiment=finetune/tofu/default.yaml \
    task_name=tofu_${model}_full \
    model=${model} \
    data/datasets@data.train=TOFU_QA_full \
    data.train.TOFU_QA_full.args.hf_args.name=full \
    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true

# Evaluate the full model on each forget split
for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    CUDA_VISIBLE_DEVICES=${CUDA_IDS} accelerate launch \
        --config_file configs/accelerate/default_config.yaml \
        --main_process_port $MASTER_PORT \
        --num_processes ${NUM_GPUS} \
        src/eval.py experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        task_name=tofu_${model}_full_${forget_split} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_full \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
        paths.output_dir=saves/eval/tofu_${model}_full/evals_${forget_split}
done
