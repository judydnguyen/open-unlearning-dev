#!/bin/bash

# SteerGRPO on forget01 — v6.0
#
# v5.9 post-mortem: forget_quality stuck at 0.0068 for all 10 epochs.
# Root cause: kl_beta=0.2 uses kl.abs() — bidirectional KL penalty that
# prevented the policy from diverging from ref in EITHER direction, killing
# the forgetting signal entirely.
#
# v6.0 returns to the parameter set from the successful v5.7_repro run
# (forget_quality=0.5894, model_utility=0.5263 @ checkpoint-200):
#   - kl_beta removed (was blocking forgetting)
#   - retain_loss_weight=0.2  (v5.7 value; keeps utility via NLL on retain)
#   - answer_reward_weight=0.6  (v5.7 value; explicit to avoid default drift)
#   - entropy_beta=0.02        (v5.7 value; generation diversity bonus)
#   - lora_r=64, lora_alpha=128  (v5.7 value)
#   - naturalness_reward_weight=0  (disabled, not in v5.7)
#   - ga_warmup_steps=2  (v5.7 value)
#   - num_train_epochs=20  (v5.7 value; best ckpt was at step 200/~270 total)
#
# New vs v5.7_repro:
#   - save-best logic active (saves best forget_quality checkpoint)
#   - GPUS=3

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget01"
RETAIN_SPLIT="retain99"
HOLDOUT_SPLIT="holdout01"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_SteerGRPO_v6.0
GPUS="1"

echo "=========================================="
echo "Running SteerGRPO unlearning (forget01) v6.0"
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
    trainer.method_args.naturalness_reward_weight=0 \
    trainer.method_args.use_lora=true \
    trainer.method_args.lora_r=64 \
    trainer.method_args.lora_alpha=128 \
    trainer.method_args.ga_warmup_steps=2 \
    trainer.method_args.resample_var_threshold=0.02 \
    trainer.method_args.curriculum_softmax_temp=2.0 \
    trainer.method_args.entropy_beta=0.02 \
    trainer.method_args.retain_loss_weight=0.3

echo "=========================================="
echo "Training completed!"
echo "Results saved to: saves/unlearn/${TASK_NAME}"
echo "=========================================="

# Step 2: Evaluate best/ checkpoint if save-best produced one
if [ -f "saves/unlearn/${TASK_NAME}/best/best_step.json" ]; then
    BEST_CKPT=saves/unlearn/${TASK_NAME}/best
    echo "Evaluating best checkpoint: $(cat saves/unlearn/${TASK_NAME}/best/best_step.json)"
    CUDA_VISIBLE_DEVICES=${GPUS%%,*} /data/judy/conda/envs/unlearning/bin/python src/eval.py \
        --config-name=eval.yaml \
        experiment=eval/tofu/default \
        forget_split=${FORGET_SPLIT} \
        holdout_split=${HOLDOUT_SPLIT} \
        model=${MODEL} \
        task_name=${TASK_NAME} \
        model.model_args.pretrained_model_name_or_path=${BEST_CKPT} \
        model.tokenizer_args.pretrained_model_name_or_path=${BEST_CKPT} \
        paths.output_dir=${BEST_CKPT}/evals \
        retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json
    echo "Best evals: ${BEST_CKPT}/evals"
fi

# Step 3: Evaluate last/ checkpoint (always)
LAST_CKPT=saves/unlearn/${TASK_NAME}/last
if [ -f "${LAST_CKPT}/last_step.json" ]; then
    echo "Evaluating last checkpoint: $(cat ${LAST_CKPT}/last_step.json)"
    CUDA_VISIBLE_DEVICES=${GPUS%%,*} /data/judy/conda/envs/unlearning/bin/python src/eval.py \
        --config-name=eval.yaml \
        experiment=eval/tofu/default \
        forget_split=${FORGET_SPLIT} \
        holdout_split=${HOLDOUT_SPLIT} \
        model=${MODEL} \
        task_name=${TASK_NAME} \
        model.model_args.pretrained_model_name_or_path=${LAST_CKPT} \
        model.tokenizer_args.pretrained_model_name_or_path=${LAST_CKPT} \
        paths.output_dir=${LAST_CKPT}/evals \
        retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json
    echo "Last evals: ${LAST_CKPT}/evals"
else
    echo "WARNING: ${LAST_CKPT}/last_step.json not found — skipping last eval"
fi

echo "=========================================="
echo "Done."
echo "=========================================="
