#!/bin/bash

# Latent Unlearning with per-sample encoder targets (latent_warmup_v1 settings)
# Same as the best run but with fresh training to confirm reproducibility.

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"
TASK_NAME="latent_warmup_v3"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"

echo "=========================================="
echo "Latent Warmup Experiment: ${TASK_NAME}"
echo "=========================================="

# Step 1: Unlearn
CUDA_VISIBLE_DEVICES=3 python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=LatentUnlearning \
    task_name=${TASK_NAME} \
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

echo "Unlearning completed: saves/unlearn/${TASK_NAME}"

# Step 2: Evaluate
CUDA_VISIBLE_DEVICES=3 python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    model=${MODEL} \
    task_name=${TASK_NAME} \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK_NAME} \
    paths.output_dir=saves/unlearn/${TASK_NAME}/evals \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

echo "=========================================="
echo "Done. Results: saves/unlearn/${TASK_NAME}/evals"
echo "=========================================="
