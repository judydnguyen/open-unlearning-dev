#!/bin/bash

# SteerGRPOSimple: minimal two-term GRPO unlearning
#   reward = (1 - answer_reward_weight) * ref_divergence_norm
#          +      answer_reward_weight  * (1 - ROUGE1_recall)
#   + retain_loss_weight * NLL(retain)  — anchors model utility
#
# Removed vs SteerGRPO:
#   - offline buffer, curriculum, resampling, naturalness, LoRA

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget01"
RETAIN_SPLIT="retain99"
HOLDOUT_SPLIT="holdout01"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_SteerGRPOSimple_v1
GPUS="2"

echo "=========================================="
echo "Running SteerGRPOSimple unlearning"
echo "Model: $MODEL"
echo "Task: $TASK_NAME"
echo "=========================================="

# Step 1: Unlearn
CUDA_VISIBLE_DEVICES=$GPUS /data/judy/conda/envs/unlearning/bin/python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SteerGRPOSimple \
    task_name=${TASK_NAME} \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.num_train_epochs=20 \
    trainer.args.learning_rate=1e-4 \
    trainer.args.logging_steps=10 \
    trainer.args.eval_strategy=epoch \
    trainer.args.save_strategy=no \
    trainer.method_args.group_size=8 \
    trainer.method_args.answer_reward_weight=0.75 \
    trainer.method_args.retain_loss_weight=0.5 \
    trainer.method_args.use_lora=true \
    trainer.method_args.lora_r=32 \
    trainer.method_args.lora_alpha=16 \
    trainer.method_args.entropy_beta=0.02

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
