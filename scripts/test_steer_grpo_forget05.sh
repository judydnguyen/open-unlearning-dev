#!/bin/bash

# SteerGRPO on forget05 — v5.9
#
# v5.8 result: ROUGE -74%, extraction -59%, NER +65%, utility -0.048.
# Problem: forget_truth_ratio stuck at ~0.505 → forget_quality stayed ~0.
#
# Loss redesign (no new terms, better balance):
#   - Asymmetric PPO clip: upside unclipped for positive advantages (forgetting),
#     downside still clipped (prevents collapse). Lets GRPO push harder when
#     the forgetting signal is clear.
#   - Removed retain_nll: was unconditionally fighting forget gradient even
#     when utility hadn't degraded. One-sided KL does the job better.
#   - Curriculum normalised by weight-sum not count: stable loss scale when
#     collapsed groups are zeroed.
#   - retain_kl_beta 0.05 → 0.2 (compensates for removing retain_nll)
#
# Script changes vs v5.8:
#   - group_size 4 → 8            (better within-group variance)
#   - per_device_train_batch_size 4 → 2  (compensate)
#   - learning_rate 2e-4 → 1e-4  (smoother with larger group_size)
#   - retain_loss_weight → 0.0, kl_beta → 0.2

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Reduce memory fragmentation over long runs (OOM fix for epoch 9+)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget05"
RETAIN_SPLIT="retain95"
HOLDOUT_SPLIT="holdout05"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_SteerGRPO_v5.9
GPUS="3"

echo "=========================================="
echo "Running SteerGRPO unlearning (forget05) v5.9"
echo "Model: $MODEL"
echo "Task: $TASK_NAME"
echo "=========================================="

# Step 1: Run Unlearning
CUDA_VISIBLE_DEVICES=$GPUS /data/judy/conda/envs/unlearning/bin/python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SteerGRPO \
    task_name=${TASK_NAME} \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
    trainer.args.per_device_train_batch_size=8 \
    trainer.args.gradient_accumulation_steps=4 \
    trainer.args.num_train_epochs=10 \
    trainer.args.learning_rate=1e-4 \
    trainer.args.logging_steps=50 \
    trainer.args.eval_strategy=epoch \
    trainer.args.save_strategy=no \
    trainer.method_args.group_size=8 \
    trainer.method_args.use_lora=true \
    trainer.method_args.lora_r=128 \
    trainer.method_args.lora_alpha=256 \
    trainer.method_args.retain_loss_weight=0.0 \
    trainer.method_args.kl_beta=0.1

echo "=========================================="
echo "Training completed!"
echo "Results saved to: saves/unlearn/${TASK_NAME}"
echo "=========================================="

# Step 2: Evaluate — prefer best/ checkpoint if save-best produced one
EVAL_CKPT=saves/unlearn/${TASK_NAME}
if [ -f "saves/unlearn/${TASK_NAME}/best/best_step.json" ]; then
    EVAL_CKPT=saves/unlearn/${TASK_NAME}/best
    echo "Using best checkpoint: $(cat saves/unlearn/${TASK_NAME}/best/best_step.json)"
fi

CUDA_VISIBLE_DEVICES=${GPUS%%,*} /data/judy/conda/envs/unlearning/bin/python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    model=${MODEL} \
    task_name=${TASK_NAME} \
    model.model_args.pretrained_model_name_or_path=${EVAL_CKPT} \
    model.tokenizer_args.pretrained_model_name_or_path=${EVAL_CKPT} \
    paths.output_dir=${EVAL_CKPT}/evals \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

echo "=========================================="
echo "Done. Results: ${EVAL_CKPT}/evals"
echo "=========================================="