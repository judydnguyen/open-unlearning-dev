#!/bin/bash
# Test script for RewardUnlearn v3.1 — SFT cold-start + GRPO unlearning.
#
# Pipeline:
#   Step 0: SFT cold-start — teach the model <think>+<answer> format on forget split
#   Step 1: GRPO unlearning — semantic reward + retain NLL from SFT checkpoint
#   Step 2: Evaluate
#
# Phase 2 (self_check_enabled=true): <think> uncertainty shaping active from step 0
#
# Prerequisites:
#   pip install sentence-transformers

set -e

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget01"
RETAIN_SPLIT="retain99"
HOLDOUT_SPLIT="holdout01"
BASE_MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
SFT_DIR="saves/sft/tofu_${MODEL}_${FORGET_SPLIT}_coldstart"
SFT_CHECKPOINT="${SFT_DIR}/final"
TASK_NAME="tofu_${MODEL}_${FORGET_SPLIT}_RewardUnlearn_v3.4"
GPU_ID="1"

echo "=========================================="
echo "RewardUnlearn v3.4: ${TASK_NAME}"
echo "Base model: ${BASE_MODEL_PATH}"
echo "Forget: ${FORGET_SPLIT}  Retain: ${RETAIN_SPLIT}"
echo "=========================================="

# Step 0: SFT cold-start (skip if checkpoint already exists)
if [ -d "${SFT_CHECKPOINT}" ]; then
    echo "SFT checkpoint found at ${SFT_CHECKPOINT} — skipping."
else
    echo "--- Step 0: SFT cold-start ---"
    CUDA_VISIBLE_DEVICES=${GPU_ID} /data/judy/conda/envs/unlearning/bin/python scripts/sft_coldstart.py \
        --model_path ${BASE_MODEL_PATH} \
        --forget_split ${FORGET_SPLIT} \
        --output_dir ${SFT_DIR} \
        --num_epochs 2 \
        --lr 2e-5 \
        --batch_size 4
    echo "SFT cold-start completed: ${SFT_CHECKPOINT}"
fi

# Step 1: GRPO unlearning from SFT checkpoint
echo "--- Step 1: GRPO unlearning ---"
CUDA_VISIBLE_DEVICES=${GPU_ID} /data/judy/conda/envs/unlearning/bin/python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=RewardUnlearn \
    task_name=${TASK_NAME} \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${SFT_CHECKPOINT} \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=4 \
    trainer.args.num_train_epochs=20 \
    trainer.args.logging_steps=1 \
    trainer.args.ddp_find_unused_parameters=false \
    trainer.args.gradient_checkpointing=true \
    trainer.args.report_to=none \
    trainer.method_args.hf_forget_split=${FORGET_SPLIT} \
    trainer.method_args.grpo_num_rollouts=8 \
    trainer.method_args.grpo_beta=1.0 \
    trainer.method_args.forget_ga_weight=0.5 \
    trainer.method_args.self_check_enabled=true

echo "Unlearning completed: saves/unlearn/${TASK_NAME}"

# Step 2: Evaluate
echo "--- Step 2: Evaluate ---"
CUDA_VISIBLE_DEVICES=${GPU_ID} /data/judy/conda/envs/unlearning/bin/python src/eval.py --config-name=eval.yaml \
    experiment=eval/tofu/default \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    model=${MODEL} \
    task_name=${TASK_NAME} \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK_NAME} \
    paths.output_dir=saves/unlearn/${TASK_NAME}/evals \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

echo "=========================================="
echo "Done. Results: saves/unlearn/${TASK_NAME}/evals/TOFU_SUMMARY.json"
echo "v3.4 changes: per-question targeted reward + gradient ascent on forget NLL (forget_ga_weight=0.5)"
echo "=========================================="
