#!/bin/bash

# Sweep script for LatentRMU on Llama-3.2-3B-Instruct across three TOFU forget splits.
# Each scenario uses the per-split hyperparameters from test_latent_rmu_3B.sh.
#
# Scenario 1 (forget01): small forget set — smaller batch
# Scenario 2 (forget05): medium forget set
# Scenario 3 (forget10): large forget set

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

MODEL="Llama-3.2-3B-Instruct"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
GPUS="0"

run_scenario() {
    local FORGET_SPLIT=$1
    local RETAIN_SPLIT=$2
    local HOLDOUT_SPLIT=$3
    local BATCH=$4

    local TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_LatentRMU_v4.8_sweep

    echo "=========================================="
    echo "Running LatentRMU: ${FORGET_SPLIT} / ${RETAIN_SPLIT}"
    echo "Task: $TASK_NAME"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPUS python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=LatentRMU \
        task_name=${TASK_NAME} \
        model=${MODEL} \
        forget_split=${FORGET_SPLIT} \
        retain_split=${RETAIN_SPLIT} \
        model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
        retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
        trainer.args.per_device_train_batch_size=${BATCH} \
        trainer.args.gradient_accumulation_steps=1 \
        trainer.args.num_train_epochs=9 \
        trainer.args.learning_rate=2e-5 \
        trainer.args.logging_steps=10 \
        trainer.args.eval_strategy=epoch \
        trainer.args.save_strategy=no \
        trainer.method_args.module_regex="model\.layers\.7" \
        trainer.method_args.encoder_epochs=4 \
        trainer.method_args.steering_coeff=10 \
        trainer.method_args.latent_dim=256 \
        trainer.method_args.orth_weight=2.0 \
        trainer.method_args.retain_sep_weight=2.0 \
        trainer.method_args.forget_warmup_steps=30 \
        trainer.method_args.gamma=1.0 \
        trainer.method_args.alpha=2.0 \
        trainer.method_args.retain_loss_type=EMBED_DIFF

    echo "=========================================="
    echo "Done: saves/unlearn/${TASK_NAME}"
    echo "=========================================="
}

# Scenario 1: forget01 — batch=4
run_scenario forget01 retain99 holdout01 2

# Scenario 2: forget05 — batch=8
run_scenario forget05 retain95 holdout05 8

# Scenario 3: forget10 — batch=8
run_scenario forget10 retain90 holdout10 8

echo "=========================================="
echo "All three scenarios completed."
echo "=========================================="
