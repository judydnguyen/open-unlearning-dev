#!/usr/bin/env bash
# Re-run SimNPO and LatentRMU on forget10, saving model weights at each
# checkpoint so that activation trajectories can be computed.
#
# Usage:
#   conda activate unlearning
#   bash scripts/run_traj_rerun.sh [GPU_ID]
#
# Outputs (model weights at each step):
#   saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_traj/checkpoint-*/
#   saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_LatentRMU_traj/checkpoint-*/

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tank/home/judy/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

GPU="${1:-0}"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export HF_HOME=/tank/home/judy/huggingface
export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune

MODEL=Llama-3.2-1B-Instruct
FORGET=forget10
RETAIN=retain90
HOLDOUT=holdout10
MODEL_PATH=open-unlearning/tofu_${MODEL}_full
RETAIN_LOGS=saves/eval/tofu_${MODEL}_${RETAIN}/TOFU_EVAL.json

echo "============================================================"
echo "Trajectory re-runs with model weight checkpointing"
echo "GPU: $GPU   forget_split: $FORGET"
echo "============================================================"

# ── SimNPO ────────────────────────────────────────────────────────────────
# forget10, 10 epochs, batch 4 × grad_accum 4 = 250 steps total → save every 25
TASK_SIMNPO=tofu_${MODEL}_${FORGET}_SimNPO_traj
echo ""
echo "--- SimNPO ---"
CUDA_VISIBLE_DEVICES=$GPU python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SimNPO \
    task_name=${TASK_SIMNPO} \
    model=${MODEL} \
    forget_split=${FORGET} \
    retain_split=${RETAIN} \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    retain_logs_path=${RETAIN_LOGS} \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=4 \
    trainer.args.save_strategy=steps \
    trainer.args.save_steps=25 \
    trainer.args.save_total_limit=12 \
    trainer.args.save_only_model=true

# ── LatentRMU ─────────────────────────────────────────────────────────────
# forget10, 10 epochs, batch 4 × no grad_accum = 1000 steps total → save every 100
TASK_LRMU=tofu_${MODEL}_${FORGET}_LatentRMU_traj
echo ""
echo "--- LatentRMU ---"
CUDA_VISIBLE_DEVICES=$GPU python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=LatentRMU \
    task_name=${TASK_LRMU} \
    model=${MODEL} \
    forget_split=${FORGET} \
    retain_split=${RETAIN} \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    retain_logs_path=${RETAIN_LOGS} \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.save_strategy=steps \
    trainer.args.save_steps=100 \
    trainer.args.save_total_limit=12 \
    trainer.args.save_only_model=true

echo ""
echo "============================================================"
echo "Done. Checkpoints with model weights saved under saves/unlearn/"
echo "Now run: python scripts/traj_analysis.py"
echo "============================================================"
