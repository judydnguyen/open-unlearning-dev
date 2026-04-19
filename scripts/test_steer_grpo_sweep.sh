#!/bin/bash

# SteerGRPO sweep across TOFU splits — per-split hyperparameter tuning
#
# Two params varied per split (all others fixed at v6.0_lr2e5_rouge values):
#
#   forget01:  lr=2e-5, retain_loss_weight=0.2
#     - Proven best (fq=0.99, mu=0.59). Small forget set = low LR is stable,
#       light retain guard does not compete with the forgetting signal.
#
#   forget05:  lr=5e-5, retain_loss_weight=0.4
#     - 5x more samples to forget → needs stronger update signal.
#       More retain guard because the forget update sweeps more of the model.
#
#   forget10:  lr=1e-4, retain_loss_weight=0.3
#     - Maximum forgetting pressure needed. retain_loss_weight deliberately
#       kept at 0.3 (not 0.4+) because prior v6.0 runs at 0.4+lr=1e-4 got
#       fq=0.0: the retain gradient cancelled the forget signal entirely.
#       If fq stays at 0, try lowering to 0.2 before increasing LR further.
#
# Fixed params (v6.0_lr2e5_rouge baseline):
#   lora_r=64, lora_alpha=128
#   group_size=8, answer_reward_weight=0.6
#   entropy_beta=0.02, ga_warmup_steps=2
#   naturalness_reward_weight=0
#   num_train_epochs=20

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/judy/huggingface

# Configuration
models=(
    # "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
    "Qwen2.5-7B-Instruct"
)
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)
GPUS="0"
PYTHON=/data/judy/conda/envs/unlearning/bin/python

# Per-split hyperparameters
declare -A SPLIT_LR
declare -A SPLIT_RETAIN_WEIGHT
SPLIT_LR["forget01"]="2e-5"
SPLIT_LR["forget05"]="5e-5"
SPLIT_LR["forget10"]="1e-4"
SPLIT_RETAIN_WEIGHT["forget01"]="0.2"
SPLIT_RETAIN_WEIGHT["forget05"]="0.4"
SPLIT_RETAIN_WEIGHT["forget10"]="0.3"

for MODEL in "${models[@]}"; do
    MODEL_PATH="open-unlearning/tofu_${MODEL}_full"

    for split in "${splits[@]}"; do
        FORGET_SPLIT=$(echo $split | cut -d' ' -f1)
        HOLDOUT_SPLIT=$(echo $split | cut -d' ' -f2)
        RETAIN_SPLIT=$(echo $split | cut -d' ' -f3)

        LR=${SPLIT_LR[$FORGET_SPLIT]}
        RETAIN_W=${SPLIT_RETAIN_WEIGHT[$FORGET_SPLIT]}
        TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_SteerGRPO_sweep_v1

        echo "=========================================="
        echo "Running SteerGRPO sweep"
        echo "  Model:               $MODEL"
        echo "  Split:               $FORGET_SPLIT / $RETAIN_SPLIT"
        echo "  learning_rate:       $LR"
        echo "  retain_loss_weight:  $RETAIN_W"
        echo "  Task:                $TASK_NAME"
        echo "=========================================="

        # Step 1: Run Unlearning
        CUDA_VISIBLE_DEVICES=$GPUS $PYTHON src/train.py --config-name=unlearn.yaml \
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
            trainer.args.learning_rate=${LR} \
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
            trainer.method_args.retain_loss_weight=${RETAIN_W}

        echo "Training completed: saves/unlearn/${TASK_NAME}"

        # Step 2: Evaluate — prefer best/ checkpoint if save-best produced one
        EVAL_CKPT=saves/unlearn/${TASK_NAME}
        if [ -f "saves/unlearn/${TASK_NAME}/best/best_step.json" ]; then
            EVAL_CKPT=saves/unlearn/${TASK_NAME}/best
            echo "Using best checkpoint: $(cat saves/unlearn/${TASK_NAME}/best/best_step.json)"
        fi

        CUDA_VISIBLE_DEVICES=${GPUS%%,*} $PYTHON src/eval.py \
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

        echo "Done. Results: ${EVAL_CKPT}/evals"
        echo "=========================================="
    done
done
