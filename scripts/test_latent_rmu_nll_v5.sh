#!/bin/bash

# LatentRMU with NLL retain loss — v4.10_NLL
#
# v4.9_NLL: forget_nll_weight=1.0, alpha=1.0
#   forget_Q_A_Prob=0.042 (good), forget_quality=0.405 (good)
#   BUT model_utility=0.481 (collapsed), forget_truth_ratio=0.520
#   Cause: forget_nll_ga ~4.0 >> retain_loss ~0.7 (5:1 ratio, GA dominated)
#   Also: warmup was NOT applied to GA term — hit full force from step 0.
#
# v4.10_NLL fixes:
#   - forget_nll_weight: 1.0 → 0.5   (halve GA strength)
#   - alpha: 1.0 → 2.0               (double retain protection)
#   - GA now also ramped by warmup_coeff (same as activation forget loss)

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget01"
RETAIN_SPLIT="retain99"
HOLDOUT_SPLIT="holdout01"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_LatentRMU_v4.10_NLL
GPUS="0"

echo "=========================================="
echo "Running LatentRMU unlearning"
echo "Model: $MODEL"
echo "Task: $TASK_NAME"
echo "=========================================="

# Step 1: Run Unlearning
CUDA_VISIBLE_DEVICES=$GPUS /data/judy/conda/envs/unlearning/bin/python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=LatentRMU \
    task_name=${TASK_NAME} \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.num_train_epochs=12 \
    trainer.args.learning_rate=1e-5 \
    trainer.args.logging_steps=10 \
    trainer.args.eval_strategy=epoch \
    trainer.args.save_strategy=no \
    trainer.method_args.module_regex="model\.layers\.7" \
    trainer.method_args.encoder_epochs=4 \
    trainer.method_args.steering_coeff=10 \
    trainer.method_args.latent_dim=256 \
    trainer.method_args.orth_weight=2.0 \
    trainer.method_args.retain_sep_weight=2.0 \
    trainer.method_args.forget_warmup_steps=30 \
    trainer.method_args.gamma=1.0 \
    trainer.method_args.alpha=0.1 \
    trainer.method_args.retain_loss_type=NLL

echo "=========================================="
echo "Training completed!"
echo "Results saved to: saves/unlearn/${TASK_NAME}"
echo "=========================================="

# Step 2: Evaluate
CUDA_VISIBLE_DEVICES=${GPUS%%,*} /data/judy/conda/envs/unlearning/bin/python src/eval.py \
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
