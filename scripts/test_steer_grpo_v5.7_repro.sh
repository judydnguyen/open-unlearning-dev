#!/bin/bash

# Reproduction of SteerGRPO_v5.7_lr2e5_rouge config.
#
# That run achieved: fq=0.5894, mu=0.5263, forget_prob=0.2318,
#   privleak=0.0249, ROUGE=0.1452, truth_ratio=-55.97 @ checkpoint-200
#
# Key differences vs v6.0 (test_steer_grpo.sh):
#   - answer_reward_weight 0.2 → 0.6  (primary forget signal, was cut 3x in v6.0)
#   - retain_loss_weight   0.4 → 0.2  (less competing retain gradient)
#   - lora_r 32 → 64, lora_alpha 64 → 128  (more expressive LoRA)
#   - learning_rate 2e-4 → 1e-4
#   - kl_beta removed (restore trainer default)
#
# NOTE: use_grad_projection and skip_mastered were set in the original v5.7 launch
# but are NOT implemented in the current SteerGRPO trainer (use_grad_projection is
# an explicit no-op in both SteerGRPO.py and SteerGRPO copy.py; skip_mastered only
# exists in SteerGRPO copy.py). They are omitted here to avoid silent no-ops.

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget01"
RETAIN_SPLIT="retain99"
HOLDOUT_SPLIT="holdout01"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_SteerGRPO_v5.7_repro
GPUS="0"

echo "=========================================="
echo "Running SteerGRPO unlearning (v5.7 repro)"
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
    trainer.args.learning_rate=1e-4 \
    trainer.args.logging_steps=10 \
    trainer.args.eval_strategy=epoch \
    trainer.args.save_strategy=no \
    trainer.method_args.group_size=8 \
    trainer.method_args.answer_reward_weight=0.6 \
    trainer.method_args.naturalness_tau=0.5 \
    trainer.method_args.naturalness_reward_weight=0 \
    trainer.method_args.use_lora=true \
    trainer.method_args.lora_r=64 \
    trainer.method_args.lora_alpha=128 \
    trainer.method_args.ga_warmup_steps=2 \
    trainer.method_args.resample_var_threshold=0.02 \
    trainer.method_args.curriculum_softmax_temp=2.0 \
    trainer.method_args.entropy_beta=0.02 \
    trainer.method_args.retain_loss_weight=0.2

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
