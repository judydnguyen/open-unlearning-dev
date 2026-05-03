#!/bin/bash

# Single-GPU MUSE unlearning sweep on the locally-finetuned Llama-3.2-3B-Instruct targets.
# Mirrors the structure of sweep_latent_rmu_3B.sh (TOFU): one run_scenario function,
# one call per data_split with that split's hyperparameters.
#
# Loads from saves/finetune/muse_Llama-3.2-3B-Instruct_${split}_target/last and uses
# saves/eval/muse_Llama-3.2-3B-Instruct_${split}_retrain/MUSE_EVAL.json as retain_logs.

set -e

export HF_HOME=/tank/home/judy/.cache/huggingface
export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/../.env" ] && { set -a; source "$SCRIPT_DIR/../.env"; set +a; }

MODEL="Llama-3.2-3B-Instruct"
GPUS="${GPUS:-1}"

run_scenario() {
    local DATA_SPLIT=$1
    local BATCH=$2
    local WARMUP=$3
    local STEERING_COEFF=$4

    local TARGET_PATH=saves/finetune/muse_${MODEL}_${DATA_SPLIT}_target/last
    local RETAIN_LOGS=saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json
    local TASK_NAME=muse_${MODEL}_${DATA_SPLIT}_LatentRMU_v4.8_sweep_01_coeff_${STEERING_COEFF}

    echo "=========================================="
    echo "Running LatentRMU: ${DATA_SPLIT}"
    echo "Task: $TASK_NAME"
    echo "Target: $TARGET_PATH"
    echo "=========================================="

    if [ ! -d "$TARGET_PATH" ]; then
        echo "[skip] target checkpoint missing at $TARGET_PATH"
        return
    fi
    if [ ! -f "$RETAIN_LOGS" ]; then
        echo "[warn] retain logs missing at $RETAIN_LOGS (privleak will be off)"
    fi

    CUDA_VISIBLE_DEVICES=$GPUS python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default.yaml \
        trainer=LatentRMU \
        task_name=${TASK_NAME} \
        model=${MODEL} \
        data_split=${DATA_SPLIT} \
        model.model_args.pretrained_model_name_or_path=${TARGET_PATH} \
        retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=${BATCH} \
        trainer.args.gradient_accumulation_steps=1 \
        trainer.args.num_train_epochs=12 \
        trainer.args.learning_rate=2e-5 \
        trainer.args.logging_steps=10 \
        trainer.args.eval_strategy=no \
        trainer.args.save_strategy=no \
        trainer.args.gradient_checkpointing=true \
        '+trainer.args.gradient_checkpointing_kwargs={use_reentrant: false}' \
        trainer.method_args.module_regex="model\.layers\.7" \
        trainer.method_args.encoder_epochs=6 \
        trainer.method_args.steering_coeff=${STEERING_COEFF} \
        trainer.method_args.latent_dim=256 \
        trainer.method_args.orth_weight=2.0 \
        trainer.method_args.forget_warmup_steps=${WARMUP} \
        trainer.method_args.gamma=1.0 \
        trainer.method_args.alpha=2.0 \
        trainer.method_args.retain_loss_type=EMBED_DIFF

    echo "=========================================="
    echo "Done: saves/unlearn/${TASK_NAME}"
    echo "=========================================="

    # Eval against the local retrain reference.
    CUDA_VISIBLE_DEVICES=$GPUS python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${TASK_NAME} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK_NAME}/last \
        paths.output_dir=saves/unlearn/${TASK_NAME}/evals \
        retain_logs_path=${RETAIN_LOGS}
}

# Scenario 1: News  — batch=2, warmup=10, steering_coeff=10
run_scenario News 2 2 30

# Scenario 2: Books — batch=2, warmup=10, steering_coeff=10
# run_scenario Books 2 10 10

echo "=========================================="
echo "All scenarios completed."
echo "=========================================="
