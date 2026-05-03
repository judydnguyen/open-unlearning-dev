#!/bin/bash
# Gamma sweep for LatentRMU — lower forget pressure to recover Util on f05/f10.
#
# Hypothesis: with 5–10× more forget data (vs forget01), the same gamma
# produces too much forget gradient → over-shifts and costs utility.
# Lowering gamma should preserve Util while still hitting Mem=1.0 (capped).
#
# Sweeps gamma in {0.5, 0.75, 1.0(baseline)} for forget05 and forget10.
# Each run writes to a distinct task_name so old results aren't overwritten.
#
# Usage:
#   bash scripts/sweep_latent_rmu_gamma.sh           # run all
#   GPUS=1 bash scripts/sweep_latent_rmu_gamma.sh    # pin to GPU 1
#   SPLITS="forget05" bash scripts/sweep_latent_rmu_gamma.sh   # subset

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
export HF_HOME=/tank/home/judy/.cache/huggingface
export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
MODEL="Llama-3.2-1B-Instruct"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
GPUS="1"

MODEL="Llama-3.2-1B-Instruct"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
GPUS="${GPUS:-1}"

# Sweep config
GAMMAS=(0.5 0.75 1.0)
SPLITS="${SPLITS:-forget05 forget10}"   # space-separated

# Per-split static config (carried over from test_latent_rmu.sh)
declare -A BATCH=( [forget01]=4 [forget05]=8  [forget10]=16 )
declare -A WARMUP=( [forget01]=10 [forget05]=30 [forget10]=30 )
declare -A RETAIN=( [forget01]="retain99" [forget05]="retain95" [forget10]="retain90" )
declare -A HOLDOUT=( [forget01]="holdout01" [forget05]="holdout05" [forget10]="holdout10" )

run_one() {
    local FORGET_SPLIT=$1
    local GAMMA=$2

    local RETAIN_SPLIT=${RETAIN[$FORGET_SPLIT]}
    local HOLDOUT_SPLIT=${HOLDOUT[$FORGET_SPLIT]}
    local B=${BATCH[$FORGET_SPLIT]}
    local W=${WARMUP[$FORGET_SPLIT]}

    # Encode gamma in task_name with two decimals (e.g., g0.50, g1.00)
    local GAMMA_TAG=$(printf 'g%.2f' "$GAMMA")
    local TASK_NAME="tofu_${MODEL}_${FORGET_SPLIT}_LatentRMU_v4.8_sweep_${GAMMA_TAG}"

    echo "=========================================="
    echo "  ${TASK_NAME}"
    echo "  gamma=${GAMMA}  forget=${FORGET_SPLIT}  batch=${B}"
    echo "=========================================="

    if [ -d "saves/unlearn/${TASK_NAME}/last" ]; then
        echo "[skip] already exists at saves/unlearn/${TASK_NAME}"
        return
    fi

    CUDA_VISIBLE_DEVICES=$GPUS python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=LatentRMU \
        task_name=${TASK_NAME} \
        model=${MODEL} \
        forget_split=${FORGET_SPLIT} \
        retain_split=${RETAIN_SPLIT} \
        model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
        retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
        trainer.args.per_device_train_batch_size=${B} \
        trainer.args.gradient_accumulation_steps=1 \
        trainer.args.num_train_epochs=12 \
        trainer.args.learning_rate=2e-5 \
        trainer.args.logging_steps=10 \
        trainer.args.eval_strategy=epoch \
        trainer.args.save_strategy=steps \
        +trainer.args.save_steps=25 \
        +trainer.args.save_total_limit=12 \
        trainer.method_args.module_regex="model\.layers\.7" \
        trainer.method_args.encoder_epochs=6 \
        trainer.method_args.steering_coeff=8 \
        trainer.method_args.latent_dim=256 \
        trainer.method_args.orth_weight=2.0 \
        trainer.method_args.forget_warmup_steps=${W} \
        trainer.method_args.gamma=${GAMMA} \
        trainer.method_args.alpha=2.0 \
        trainer.method_args.retain_loss_type=EMBED_DIFF

    # Eval each saved checkpoint with full MIA suite
    for ckpt_dir in saves/unlearn/${TASK_NAME}/checkpoint-*/; do
        [ -d "$ckpt_dir" ] || continue
        ckpt_num=$(basename "$ckpt_dir" | sed 's/checkpoint-//')
        [ "$ckpt_num" = "0" ] && continue
        echo "  [eval] ${TASK_NAME}/checkpoint-${ckpt_num}"
        CUDA_VISIBLE_DEVICES=$GPUS python src/eval.py \
            --config-name=eval.yaml \
            experiment=eval/tofu/default \
            eval=tofu_full_privacy \
            eval.tofu.overwrite=true \
            forget_split=${FORGET_SPLIT} \
            holdout_split=${HOLDOUT_SPLIT} \
            model=${MODEL} \
            task_name=${TASK_NAME}_eval_${ckpt_num} \
            model.model_args.pretrained_model_name_or_path=${ckpt_dir} \
            model.tokenizer_args.pretrained_model_name_or_path=${ckpt_dir} \
            paths.output_dir=${ckpt_dir}evals \
            retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json
    done

    # Also eval best/ if present
    if [ -d "saves/unlearn/${TASK_NAME}/best" ]; then
        echo "  [eval] ${TASK_NAME}/best"
        CUDA_VISIBLE_DEVICES=$GPUS python src/eval.py \
            --config-name=eval.yaml \
            experiment=eval/tofu/default \
            eval=tofu_full_privacy \
            eval.tofu.overwrite=true \
            forget_split=${FORGET_SPLIT} \
            holdout_split=${HOLDOUT_SPLIT} \
            model=${MODEL} \
            task_name=${TASK_NAME}_eval_best \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK_NAME}/best \
            model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/${TASK_NAME}/best \
            paths.output_dir=saves/unlearn/${TASK_NAME}/best/evals \
            retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json
    fi

    echo "Done: saves/unlearn/${TASK_NAME}"
}

for split in $SPLITS; do
    for gamma in "${GAMMAS[@]}"; do
        run_one "$split" "$gamma"
    done
done

echo ""
echo "=========================================="
echo "  Sweep complete."
echo "  After this finishes, run paper_table.py"
echo "  with --our_pattern set to each new task_name"
echo "  to compare composite HM."
echo "=========================================="
