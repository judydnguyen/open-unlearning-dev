#!/bin/bash
# BTPairDPO: DPO unlearning from pre-built BT preference pairs.
#
# Skips Phase 1 entirely — uses a pre-existing bt_pairs.jsonl.
# Much faster than RewardUnlearn since no reward model training is needed.

set -e

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget01"
RETAIN_SPLIT="retain99"
HOLDOUT_SPLIT="holdout01"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TASK_NAME="tofu_${MODEL}_${FORGET_SPLIT}_BTPairDPO_v1"
GPU_ID="0"

# Path to the pre-built BT pairs from RewardUnlearn Phase 1
BT_PAIRS_PATH="saves/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_RewardUnlearn_v1.2/reward_model/bt_pairs.jsonl"

echo "=========================================="
echo "BTPairDPO: ${TASK_NAME}"
echo "Model: ${MODEL_PATH}"
echo "BT pairs: ${BT_PAIRS_PATH}"
echo "Forget: ${FORGET_SPLIT}  Retain: ${RETAIN_SPLIT}"
echo "=========================================="

# Step 1: Unlearn via DPO on BT pairs
CUDA_VISIBLE_DEVICES=${GPU_ID} /data/judy/conda/envs/unlearning/bin/python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=BTPairDPO \
    task_name=${TASK_NAME} \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=4 \
    trainer.args.num_train_epochs=5 \
    trainer.args.logging_steps=1 \
    trainer.args.report_to=none \
    "trainer.method_args.bt_pairs_path=${BT_PAIRS_PATH}" \
    trainer.method_args.dpo_beta=0.1 \
    trainer.method_args.gamma=1.0 \
    trainer.method_args.alpha=1.0

echo "Unlearning completed: saves/unlearn/${TASK_NAME}"

# Step 2: Evaluate
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
echo "=========================================="
