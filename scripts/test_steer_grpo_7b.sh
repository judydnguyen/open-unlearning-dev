#!/bin/bash

# SteerGRPO on Llama-3.1-8B-Instruct (TOFU forget01).
#
# Differences from test_steer_grpo.sh (1B single-GPU):
#   - MODEL: Llama-3.2-1B → Llama-3.1-8B-Instruct
#   - MODEL_PATH: local saves/finetune/ (both finetune + retain99 logs already present)
#   - Multi-GPU: python3 -m torch.distributed.launch --nproc_per_node=2 across GPUs 1,2
#   - per_device_train_batch_size: 4 → 1  (VRAM: ~14 GB per GPU for 8B + LoRA)
#   - gradient_accumulation_steps: 1 → 4  (effective batch = 4, same as 1B run)
#   - group_size: 8 → 4  (generation dominates VRAM; halving keeps peak manageable)
#   - lora_r: 64 → 32, lora_alpha: 128 → 64  (8B is sensitive; smaller adapter to start)
#   - learning_rate: 1e-4 → 5e-5  (larger model, reduce LR)
#   - retain_loss_weight kept at 0.2 (consistent with offline script)

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

MODEL="Qwen2.5-7B-Instruct"
FORGET_SPLIT="forget01"
RETAIN_SPLIT="retain99"
HOLDOUT_SPLIT="holdout01"
MODEL_PATH="saves/finetune/tofu_${MODEL}_full"
TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_SteerGRPO_v1.0_lr5e5
GPUS="1,2"
N_GPUS=2

echo "=========================================="
echo "Running SteerGRPO unlearning (7B/8B)"
echo "Model: $MODEL"
echo "Task: $TASK_NAME"
echo "GPUs: $GPUS"
echo "=========================================="

# Step 1: Run Unlearning
CUDA_VISIBLE_DEVICES=$GPUS python3 -m torch.distributed.launch \
    --nproc_per_node=$N_GPUS \
    --master_port=$MASTER_PORT \
    --use-env \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SteerGRPO \
    task_name=${TASK_NAME} \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.num_train_epochs=20 \
    trainer.args.learning_rate=5e-5 \
    trainer.args.logging_steps=10 \
    trainer.args.eval_strategy=epoch \
    trainer.args.save_strategy=no \
    trainer.method_args.group_size=4 \
    trainer.method_args.answer_reward_weight=0.6 \
    trainer.method_args.naturalness_reward_weight=0 \
    trainer.method_args.use_lora=true \
    trainer.method_args.lora_r=32 \
    trainer.method_args.lora_alpha=64 \
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

# Step 2: Evaluate (single GPU — only rank-0 needed)
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
