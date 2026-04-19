#!/bin/bash

# SteerGRPOMultiGPU on forget01 — v6.0_multigpu
#
# Same hyperparameters as v6.0 but uses SteerGRPOMultiGPU trainer:
#   - policy model pinned to cuda:0 (first listed GPU)
#   - ref model pinned to cuda:1 (second listed GPU, fp16, frozen)
#   - no DDP; single-process with manual device split
#
# v6.0 base params (unchanged):
#   - retain_loss_weight=0.4, answer_reward_weight=0.6
#   - entropy_beta=0.02, lora_r=64, lora_alpha=128
#   - group_size=4, ga_warmup_steps=2

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/judy/huggingface


MODEL="Llama-3.1-8B-Instruct"
FORGET_SPLIT="forget05"
RETAIN_SPLIT="retain95"
HOLDOUT_SPLIT="holdout05"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_SteerGRPOMultiGPU_v6.1
GPUS="0,1"   # cuda:0=GPU3 (policy+training), cuda:1=GPU2 (ref only, fp16)
SKIP_EVAL=${SKIP_EVAL:-0}  # set SKIP_EVAL=1 to skip evaluation

echo "=========================================="
echo "Running SteerGRPOMultiGPU unlearning (forget05) v6.0"
echo "Model: $MODEL"
echo "Task: $TASK_NAME"
echo "=========================================="

# Clean up previous save dir before training
SAVE_DIR=saves/unlearn/${TASK_NAME}
if [ -d "$SAVE_DIR" ]; then
    echo "Removing previous save dir: $SAVE_DIR"
    rm -rf "$SAVE_DIR"
fi

# Step 1: Run Unlearning
CUDA_VISIBLE_DEVICES=$GPUS /data/judy/conda/envs/unlearning/bin/accelerate launch \
    --config_file configs/accelerate/ddp_config.yaml \
    --main_process_port $MASTER_PORT \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SteerGRPOMultiGPU \
    task_name=${TASK_NAME} \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.gradient_checkpointing=true \
    trainer.args.num_train_epochs=10 \
    trainer.args.learning_rate=1e-4 \
    trainer.args.logging_steps=10 \
    trainer.args.eval_strategy=epoch \
    trainer.args.save_strategy=no \
    trainer.method_args.group_size=4 \
    trainer.method_args.answer_reward_weight=0.6 \
    trainer.method_args.naturalness_reward_weight=0.1 \
    trainer.method_args.use_lora=true \
    trainer.method_args.lora_r=32 \
    trainer.method_args.lora_alpha=64 \
    trainer.method_args.ga_warmup_steps=2 \
    trainer.method_args.resample_var_threshold=0.02 \
    trainer.method_args.curriculum_softmax_temp=2.0 \
    trainer.method_args.entropy_beta=0.02 \
    trainer.method_args.retain_loss_weight=0.2

# # Step 1: Run Unlearning
# CUDA_VISIBLE_DEVICES=$GPUS /data/judy/conda/envs/unlearning/bin/accelerate launch \
#     --config_file configs/accelerate/ddp_config.yaml \
#     --main_process_port $MASTER_PORT \
#     src/train.py --config-name=unlearn.yaml \
#     experiment=unlearn/tofu/default \
#     trainer=SteerGRPOMultiGPU \
#     task_name=${TASK_NAME} \
#     model=${MODEL} \
#     forget_split=${FORGET_SPLIT} \
#     retain_split=${RETAIN_SPLIT} \
#     model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
#     retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
#     trainer.args.per_device_train_batch_size=4 \
#     trainer.args.gradient_accumulation_steps=1 \
#     trainer.args.gradient_checkpointing=true \
#     trainer.args.num_train_epochs=10 \
#     trainer.args.learning_rate=2e-4 \
#     trainer.args.logging_steps=10 \
#     trainer.args.eval_strategy=epoch \
#     trainer.args.save_strategy=no \
#     trainer.method_args.group_size=4 \
#     trainer.method_args.answer_reward_weight=0.6 \
#     trainer.method_args.naturalness_reward_weight=0.1 \
#     trainer.method_args.use_lora=true \
#     trainer.method_args.lora_r=64 \
#     trainer.method_args.lora_alpha=128 \
#     trainer.method_args.ga_warmup_steps=0 \
#     trainer.method_args.resample_var_threshold=0.02 \
#     trainer.method_args.curriculum_softmax_temp=2.0 \
#     trainer.method_args.entropy_beta=0.02 \
#     trainer.method_args.retain_loss_weight=0.1

echo "=========================================="
echo "Training completed!"
echo "Results saved to: saves/unlearn/${TASK_NAME}"
echo "=========================================="

if [ "$SKIP_EVAL" = "1" ]; then
    echo "Skipping evaluation (SKIP_EVAL=1)"
    exit 0
fi

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
