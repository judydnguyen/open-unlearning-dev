#!/bin/bash

# PurgeGRPO on TOFU — v1.0
#
# PURGE baseline: GRPO with binary forget-word reward.
# Reward = 1.0 if completion contains none of the forget-set words (auto-extracted
# from forget dataset answer labels), 0.0 otherwise.
#
# Starting hyperparameters mirror the SteerGRPO v5.7 setting that worked well
# (forget_quality ~0.59 on forget05) with the auxiliary reward terms zeroed out:
#   - answer_reward_weight=0  (PURGE uses word-list reward, not ROUGE anti-answer)
#   - naturalness_reward_weight=0
#   - kl_beta=0
#   - retain_loss_weight=0.2   (NLL on retain keeps utility from collapsing)
#   - use_lora=true, lora_r=64 (memory-efficient; merge on save)
#   - curriculum=false          (faithful PURGE baseline)
#   - entropy_beta=0.0          (no extra diversity signal)

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/judy/huggingface

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget05"
RETAIN_SPLIT="retain95"
HOLDOUT_SPLIT="holdout05"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_PurgeGRPO_v1.0
GPUS="3"

echo "=========================================="
echo "Running PurgeGRPO unlearning (forget05) v1.0"
echo "Model:  $MODEL"
echo "Split:  $FORGET_SPLIT / $RETAIN_SPLIT"
echo "Task:   $TASK_NAME"
echo "GPU(s): $GPUS"
echo "=========================================="

# Step 1: Unlearning
CUDA_VISIBLE_DEVICES=$GPUS /data/judy/conda/envs/unlearning/bin/python src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=PurgeGRPO \
    task_name=${TASK_NAME} \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.num_train_epochs=10 \
    trainer.args.learning_rate=1e-4 \
    trainer.args.logging_steps=10 \
    trainer.args.eval_strategy=epoch \
    trainer.args.save_strategy=no \
    trainer.method_args.group_size=8 \
    trainer.method_args.max_new_tokens=64 \
    trainer.method_args.temperature=1.0 \
    trainer.method_args.epsilon=0.2 \
    trainer.method_args.use_lora=true \
    trainer.method_args.lora_r=64 \
    trainer.method_args.lora_alpha=128 \
    trainer.method_args.retain_loss_weight=0.2 \
    trainer.method_args.resample_low_var=true \
    trainer.method_args.resample_var_threshold=0.02

echo "=========================================="
echo "Training completed!"
echo "Results saved to: saves/unlearn/${TASK_NAME}"
echo "=========================================="

# Step 2: Evaluate — use best/ checkpoint if save-best produced one
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
