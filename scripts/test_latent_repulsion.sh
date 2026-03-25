#!/bin/bash

# Latent Unlearning with per-sample encoder targets (latent_warmup_v1 settings)
# Same as the best run but with fresh training to confirm reproducibility.

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

MODEL="Llama-3.2-1B-Instruct"

splits=(
    # "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

TASK_NAME="steering_v3.1"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
GPU_ID="2"

for split in "${splits[@]}"; do
    read -r FORGET_SPLIT HOLDOUT_SPLIT RETAIN_SPLIT <<< "$split"
    SPLIT_TASK_NAME="${TASK_NAME}_${FORGET_SPLIT}"

    echo "=========================================="
    echo "Latent Warmup Experiment: ${SPLIT_TASK_NAME}"
    echo "=========================================="

    # Step 1: Unlearn
    CUDA_VISIBLE_DEVICES=${GPU_ID} python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=LatentUnlearning \
        task_name=${SPLIT_TASK_NAME} \
        model=${MODEL} \
        forget_split=${FORGET_SPLIT} \
        retain_split=${RETAIN_SPLIT} \
        model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
        retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
        trainer.method_args.forget_loss_type=mse \
        trainer.method_args.steering_coeff=2.0 \
        trainer.method_args.intervention_layer=7 \
        trainer.method_args.lambda_util=1.0 \
        trainer.method_args.forget_warmup_steps=50 \
        trainer.args.learning_rate=1e-5 \
        trainer.args.num_train_epochs=17 \
        trainer.args.per_device_train_batch_size=2 \
        trainer.args.gradient_accumulation_steps=4 \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true

    echo "Unlearning completed: saves/unlearn/${SPLIT_TASK_NAME}"

    # Step 2: Evaluate
    CUDA_VISIBLE_DEVICES=${GPU_ID} python src/eval.py \
        --config-name=eval.yaml \
        experiment=eval/tofu/default \
        forget_split=${FORGET_SPLIT} \
        holdout_split=${HOLDOUT_SPLIT} \
        model=${MODEL} \
        task_name=${SPLIT_TASK_NAME} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${SPLIT_TASK_NAME} \
        paths.output_dir=saves/unlearn/${SPLIT_TASK_NAME}/evals \
        retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

    echo "=========================================="
    echo "Done. Results: saves/unlearn/${SPLIT_TASK_NAME}/evals"
    echo "=========================================="
done
