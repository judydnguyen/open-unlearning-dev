#!/bin/bash

# Test script for SteerGRPO with offline buffer mixing (NER-reward completions).
#
# Changes vs v5.7_lr2e5_rouge (test_steer_grpo.sh):
#   - offline_fraction=0.25: 2 of 8 completions per group come from a pre-generated
#     offline buffer; decays linearly to 0.05 by end of training
#   - Offline buffer uses 5 system-prompt variants ("You never learned this...", etc.)
#     and rewards via NER entity recall: reward = 1.0 - entity_recall
#   - Buffer is refreshed automatically when mean importance ratio of offline
#     completions drops below 0.3
#   - Task name versioned separately (_offline_v1) to avoid clobbering v5.7 runs

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

MODEL="Llama-3.2-1B-Instruct"
# MODEL="Qwen2.5-7B-Instruct"
FORGET_SPLIT="forget01"
RETAIN_SPLIT="retain99"
HOLDOUT_SPLIT="holdout01"
# MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
MODEL_PATH="saves/finetune/tofu_${MODEL}_full"
# MODEL_PATH="/home/judy/code/open-unlearning-dev/saves/sft/tofu_Llama-3.2-1B-Instruct_forget01_coldstart/final"
TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_SteerGRPO_offline_v2.2
GPUS="2"

echo "=========================================="
echo "Running SteerGRPO unlearning (offline buffer mix)"
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
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.num_train_epochs=20 \
    trainer.args.learning_rate=2e-4 \
    trainer.args.logging_steps=10 \
    trainer.args.eval_strategy=epoch \
    trainer.args.save_strategy=no \
    +trainer.args.save_total_limit=20 \
    trainer.method_args.group_size=4 \
    trainer.method_args.answer_reward_weight=0.6 \
    trainer.method_args.naturalness_tau=0.5 \
    trainer.method_args.naturalness_reward_weight=0 \
    trainer.method_args.use_grad_projection=true \
    trainer.method_args.use_lora=true \
    trainer.method_args.lora_r=32 \
    trainer.method_args.lora_alpha=128 \
    trainer.method_args.ga_warmup_steps=2 \
    trainer.method_args.resample_var_threshold=0.02 \
    trainer.method_args.curriculum_softmax_temp=2.0 \
    trainer.method_args.entropy_beta=0.02 \
    trainer.method_args.skip_mastered=true \
    trainer.method_args.skip_ema_threshold=0.5 \
    trainer.method_args.retain_loss_weight=0.2 \
    trainer.method_args.offline_fraction=0.25

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
    model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/${TASK_NAME} \
    paths.output_dir=saves/unlearn/${TASK_NAME}/evals \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

echo "=========================================="
echo "Done. Results: saves/unlearn/${TASK_NAME}/evals"
echo "=========================================="
