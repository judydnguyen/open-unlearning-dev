#!/bin/bash
# Sweep steering_coeff x gamma for LatentRMU on MUSE 7B (Llama-2-7b-hf).
# Base trainer: LatentRMUParallelDDP_strong_v2 (alpha=0.7, edit layers 10-20,
# steering hook at layer 20, lr=2e-4 → overridden below to 2e-5 to match the
# existing 7B DDP recipe). Per-run overrides: steering_coeff and gamma.
#
# Grid:
#   COEFFS = 10 20 30 40
#   GAMMAS = 2 5 10
# = 12 runs per data split.
#
# Override at call time:
#   COEFFS="20 30" GAMMAS="5" DATA_SPLITS="Books" bash scripts/muse_sweep_coeff_gamma_7b.sh

export HF_HOME=/tank/home/judy/.cache/huggingface
export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/../.env" ] && { set -a; source "$SCRIPT_DIR/../.env"; set +a; }

if [ -z "${HF_TOKEN:-}" ]; then
    HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
    [ -f "$HF_TOKEN_FILE" ] && export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

per_device_train_batch_size=1
gradient_accumulation_steps=2
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
EVAL_GPU="${EVAL_GPU:-1}"

model=Llama-2-7b-hf
trainer=LatentRMUParallelDDP_strong_v2

COEFFS="${COEFFS:-10 20 30 40}"
GAMMAS="${GAMMAS:-2 5 10}"
DATA_SPLITS="${DATA_SPLITS:-Books News}"

for data_split in $DATA_SPLITS; do
    RETAIN_LOGS=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
    [ -f "$RETAIN_LOGS" ] || echo "[warn] retain logs missing at $RETAIN_LOGS (privleak will be off)"

    for coeff in $COEFFS; do
        for gamma in $GAMMAS; do
            TAG="sc${coeff}_g${gamma}_a07"
            task_name=muse_${model}_${data_split}_LatentRMU_sweep_${TAG}
            SAVE_DIR=saves/unlearn/${task_name}

            echo "============================================="
            echo "split=$data_split  coeff=$coeff  gamma=$gamma"
            echo "task: $task_name"
            echo "GPUs: $TRAIN_GPUS"
            echo "============================================="

            # Fresh port per run so a stale port from a prior run can't collide.
            export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

            if [ -f "$SAVE_DIR/last/model.safetensors" ] || [ -f "$SAVE_DIR/last/model.safetensors.index.json" ]; then
                echo "[done] training already exists for $TAG"
            else
                CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
                    --config_file configs/accelerate/ddp_2gpu_config.yaml \
                    --main_process_port $MASTER_PORT \
                    src/train.py --config-name=unlearn.yaml \
                    experiment=unlearn/muse/default.yaml \
                    model=${model} \
                    data_split=${data_split} \
                    trainer=${trainer} \
                    task_name=${task_name} \
                    retain_logs_path=${RETAIN_LOGS} \
                    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                    trainer.args.ddp_find_unused_parameters=true \
                    trainer.args.gradient_checkpointing=true \
                    trainer.args.save_strategy=epoch \
                    +trainer.args.save_total_limit=10 \
                    trainer.args.save_only_model=true \
                    trainer.args.num_train_epochs=12 \
                    trainer.args.learning_rate=2e-5 \
                    trainer.args.logging_steps=10 \
                    trainer.method_args.steering_coeff=${coeff} \
                    trainer.method_args.gamma=${gamma}

                if [ $? -ne 0 ]; then
                    echo "[err] training failed for $TAG — skipping eval."
                    continue
                fi
            fi

            if [ -f "$SAVE_DIR/evals/MUSE_SUMMARY.json" ]; then
                echo "[done] eval already exists for $TAG"
                continue
            fi

            echo "--- Eval $TAG on GPU $EVAL_GPU ---"
            CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
                experiment=eval/muse/default.yaml \
                data_split=${data_split} \
                task_name=${task_name} \
                model=${model} \
                model.model_args.pretrained_model_name_or_path=${SAVE_DIR}/last \
                paths.output_dir=${SAVE_DIR}/evals \
                retain_logs_path=${RETAIN_LOGS}
        done
    done
done

echo "=========================================="
echo "Sweep complete. Compare MUSE summaries with:"
echo '  for f in saves/unlearn/muse_*_LatentRMU_sweep_sc*_g*_a07/evals/MUSE_SUMMARY.json; do echo "$f"; cat "$f"; echo; done'
echo "=========================================="
