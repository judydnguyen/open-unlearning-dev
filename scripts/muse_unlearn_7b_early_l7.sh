#!/bin/bash
# Single-run experiment: test whether an early-layer hook (layer 7) moves
# privleak off the −99 floor seen in every layer-20 run on MUSE-News 7B.
#
# Hypothesis: at layer 20, MIA can still cleanly separate forget vs holdout
# from layers 0–19's untouched representations → privleak ≈ target's −99.81.
# Hooking at layer 7 should disrupt those earlier representations.
#
# Risk: layer-7 disturbance propagates through 24 downstream layers →
# stronger retain damage than layer 20. Compensated here by alpha=1.0
# (vs v1's 0.5). Per-epoch saves so we can pick best ckpt before retain dies.
#
# Compare against:
#   v1 (layer 20, g8 sc40 a05): forget 0.38 / retain 0.41 / privleak −99.6
#   gold retrain target:        forget 0.33 / retain 0.56 / privleak −4.72

export HF_HOME=/tank/home/judy/.cache/huggingface
export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/../.env" ] && { set -a; source "$SCRIPT_DIR/../.env"; set +a; }
if [ -z "${HF_TOKEN:-}" ]; then
    HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
    [ -f "$HF_TOKEN_FILE" ] && export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

model=Llama-2-7b-hf
data_split=News
trainer=LatentRMUParallelDDP_early

per_device_train_batch_size=2
gradient_accumulation_steps=1
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
EVAL_GPU="${EVAL_GPU:-1}"
ACCEL_CFG=configs/accelerate/ddp_2gpu_config.yaml

TAG="l7_g8sc40a10_lr1e-4_ep12_bz_4"
task_name=muse_${model}_${data_split}_LatentRMU_early_${TAG}
save_dir=saves/unlearn/${task_name}

target_path=saves/finetune/muse_${model}_${data_split}_target/last
[ -d "$target_path" ] || target_path=saves/finetune/muse_${model}_${data_split}_target
retain_logs=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json

[ -f "$retain_logs" ] || echo "[warn] retain logs missing at $retain_logs (privleak will be off)"

echo "============================================="
echo "Run: $task_name"
echo "  hook layer = 7 (early)"
echo "  edit surface = layers 5–8 down_proj"
echo "  alpha=1.0 (bumped from v1 0.5 to protect retain)"
echo "============================================="

# ---- 1. unlearn ----
if [ -f "$save_dir/last/model.safetensors" ] || [ -f "$save_dir/last/model.safetensors.index.json" ]; then
    echo "[done] training already exists at $save_dir — delete to redo"
else
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
        --config_file $ACCEL_CFG \
        --main_process_port $MASTER_PORT \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default.yaml \
        model=${model} \
        data_split=${data_split} \
        trainer=${trainer} \
        task_name=${task_name} \
        retain_logs_path=${retain_logs} \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.method_args.module_regex='model\.layers\.12' \
        trainer.args.save_strategy=epoch \
        +trainer.args.save_total_limit=10 \
        trainer.args.save_only_model=true \
        trainer.args.num_train_epochs=12 \
        trainer.args.learning_rate=1e-4 \
        trainer.args.logging_steps=10 \
        trainer.method_args.steering_coeff=40 \
        trainer.method_args.gamma=8.0 \
        trainer.method_args.encoder_epochs=2 \
        trainer.args.eval_strategy=no \
        trainer.method_args.alpha=1.0

    if [ $? -ne 0 ]; then
        echo "[err] training failed — not running eval."
        exit 1
    fi
fi

# ---- 2. eval final checkpoint ----
if [ -f "$save_dir/evals/MUSE_SUMMARY.json" ]; then
    echo "[done] eval already exists at $save_dir/evals/"
else
    echo "--- Eval $task_name (last/) on GPU $EVAL_GPU ---"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${data_split} \
        task_name=${task_name} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=${save_dir}/last \
        paths.output_dir=${save_dir}/evals \
        retain_logs_path=${retain_logs}
fi

echo "=========================================="
echo "Done. Next step — eval every checkpoint to find the best epoch:"
echo "  bash scripts/eval_muse_checkpoints.sh $save_dir News $model $EVAL_GPU"
echo ""
echo "Compare against v1 (layer 20):"
echo "  forget 0.38 / retain 0.41 / privleak −99.60"
echo "Pass criteria: any ckpt with privleak < −80 AND retain > 0.45 AND forget < 0.45"
echo "=========================================="
