#!/bin/bash

# Steering coefficient sweep for LatentRMU on MUSE 3B targets.
# Trains and evals one model per coeff value so we can plot how representation
# displacement (and the standard MUSE metrics) scale with the coefficient.

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
DATA_SPLIT="${DATA_SPLIT:-News}"

# Coefficients to sweep. 0 = no steering (encoder still trained but Phase 2 ignores it
# in effect — useful as a control). Larger = stronger push away from original activations.
COEFFS="${COEFFS:-0 5 10 20 40 80}"

TARGET_PATH=saves/finetune/muse_${MODEL}_${DATA_SPLIT}_target/last
RETAIN_LOGS=saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json

[ -d "$TARGET_PATH" ] || { echo "ERROR: target missing at $TARGET_PATH"; exit 1; }
[ -f "$RETAIN_LOGS" ] || echo "WARN: retain logs missing at $RETAIN_LOGS"

for COEFF in $COEFFS; do
    TASK_NAME=muse_${MODEL}_${DATA_SPLIT}_LatentRMU_sweep_coeff_${COEFF}
    SAVE_DIR=saves/unlearn/${TASK_NAME}

    echo "============================================="
    echo "coeff=$COEFF → $TASK_NAME"
    echo "============================================="

    # Train (skip if already done).
    if [ -f "$SAVE_DIR/last/model.safetensors" ] || [ -f "$SAVE_DIR/last/model.safetensors.index.json" ]; then
        echo "[done] training already exists for coeff=$COEFF"
    else
        CUDA_VISIBLE_DEVICES=$GPUS python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/default.yaml \
            trainer=LatentRMU \
            task_name=${TASK_NAME} \
            model=${MODEL} \
            data_split=${DATA_SPLIT} \
            model.model_args.pretrained_model_name_or_path=${TARGET_PATH} \
            retain_logs_path=${RETAIN_LOGS} \
            trainer.args.per_device_train_batch_size=2 \
            trainer.args.gradient_accumulation_steps=1 \
            trainer.args.num_train_epochs=12 \
            trainer.args.learning_rate=2e-5 \
            trainer.args.logging_steps=10 \
            trainer.args.eval_strategy=epoch \
            trainer.args.save_strategy=no \
            trainer.args.gradient_checkpointing=true \
            '+trainer.args.gradient_checkpointing_kwargs={use_reentrant: false}' \
            trainer.method_args.module_regex="model\.layers\.7" \
            trainer.method_args.encoder_epochs=6 \
            trainer.method_args.steering_coeff=${COEFF} \
            trainer.method_args.latent_dim=256 \
            trainer.method_args.orth_weight=2.0 \
            trainer.method_args.forget_warmup_steps=5 \
            trainer.method_args.gamma=1.0 \
            trainer.method_args.alpha=2.0 \
            trainer.method_args.retain_loss_type=EMBED_DIFF
    fi

    # Eval.
    if find "$SAVE_DIR/evals" -maxdepth 2 -name MUSE_SUMMARY.json 2>/dev/null | grep -q .; then
        echo "[done] eval already exists for coeff=$COEFF"
    elif [ -f "$SAVE_DIR/last/model.safetensors" ] || [ -f "$SAVE_DIR/last/model.safetensors.index.json" ]; then
        CUDA_VISIBLE_DEVICES=$GPUS python src/eval.py \
            experiment=eval/muse/default.yaml \
            data_split=${DATA_SPLIT} \
            task_name=${TASK_NAME} \
            model=${MODEL} \
            model.model_args.pretrained_model_name_or_path=${SAVE_DIR}/last \
            paths.output_dir=${SAVE_DIR}/evals \
            retain_logs_path=${RETAIN_LOGS}
    fi
done

echo "=========================================="
echo "Sweep complete. Run analysis with:"
echo "  python scripts/analyze_steering_coeff.py --data_split ${DATA_SPLIT} --model ${MODEL} --coeffs \"${COEFFS}\""
echo "=========================================="
