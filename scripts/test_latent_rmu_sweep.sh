#!/bin/bash

# LatentRMU sweep across TOFU splits — per-split hyperparameter tuning
#
# Two params varied per split (all others fixed at v4.8 values):
#
#   forget01:  steering_coeff=10, num_train_epochs=9, alpha=1.0
#     - v4.8 baseline values; small forget set converges quickly.
#       Low alpha avoids utility collapse on a tiny forget set.
#
#   forget05:  steering_coeff=15, num_train_epochs=9, alpha=1.5
#     - Best eval result so far was checkpoint-100 (mid-training), meaning the
#       model over-steers by epoch 9. Stronger coeff lets steering take effect
#       earlier before utility decays. save_strategy=epoch keeps the best ckpt.
#       Alpha bumped to 1.5 to keep forget pressure proportional to set size.
#
#   forget10:  steering_coeff=20, num_train_epochs=12, alpha=2.0
#     - Largest set needs the strongest steering signal and more passes.
#       Peak was checkpoint-200 out of 9 epochs (late), so more epochs gives
#       the steering more room to work before decay sets in.
#       Alpha=2.0 provides maximum forget pressure needed to erase all samples.
#
# Fixed params (v4.8 baseline):
#   lr=1e-5, encoder_epochs=4, latent_dim=256
#   orth_weight=2.0, retain_sep_weight=2.0
#   forget_warmup_steps=30, gamma=1.0, alpha=1.0
#   module_regex="model.layers.7", retain_loss_type=EMBED_DIFF

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
export HF_HOME=/data/judy/huggingface
echo "HF_HOME: $HF_HOME"

# Configuration
models=(
    "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
    # "Qwen2.5-7B-Instruct"
)
splits=(
    "forget01 holdout01 retain99"
    # "forget05 holdout05 retain95"
    # "forget10 holdout10 retain90"
)
GPUS="0,1"
PYTHON=/data/judy/conda/envs/unlearning/bin/python

# Per-split hyperparameters
declare -A SPLIT_STEERING_COEFF
declare -A SPLIT_EPOCHS
declare -A SPLIT_ALPHA
SPLIT_STEERING_COEFF["forget01"]="10"
SPLIT_STEERING_COEFF["forget05"]="15"
SPLIT_STEERING_COEFF["forget10"]="20"
SPLIT_EPOCHS["forget01"]="9"
SPLIT_EPOCHS["forget05"]="9"
SPLIT_EPOCHS["forget10"]="12"
# alpha scales with forget set size: larger sets need more forget pressure to erase
# all samples, but small sets risk utility collapse with high alpha
SPLIT_ALPHA["forget01"]="1.0"
SPLIT_ALPHA["forget05"]="1.5"
SPLIT_ALPHA["forget10"]="2.0"

for MODEL in "${models[@]}"; do
    MODEL_PATH="open-unlearning/tofu_${MODEL}_full"

    for split in "${splits[@]}"; do
        FORGET_SPLIT=$(echo $split | cut -d' ' -f1)
        HOLDOUT_SPLIT=$(echo $split | cut -d' ' -f2)
        RETAIN_SPLIT=$(echo $split | cut -d' ' -f3)

        STEERING=${SPLIT_STEERING_COEFF[$FORGET_SPLIT]}
        EPOCHS=${SPLIT_EPOCHS[$FORGET_SPLIT]}
        ALPHA=${SPLIT_ALPHA[$FORGET_SPLIT]}
        TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_LatentRMU_sweep_v1

        echo "=========================================="
        echo "Running LatentRMU sweep"
        echo "  Model:           $MODEL"
        echo "  Split:           $FORGET_SPLIT / $RETAIN_SPLIT"
        echo "  steering_coeff:  $STEERING"
        echo "  num_epochs:      $EPOCHS"
        echo "  alpha:           $ALPHA"
        echo "  Task:            $TASK_NAME"
        echo "=========================================="

        # Step 1: Run Unlearning
        CUDA_VISIBLE_DEVICES=$GPUS $PYTHON src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            trainer=LatentRMU \
            task_name=${TASK_NAME} \
            model=${MODEL} \
            forget_split=${FORGET_SPLIT} \
            retain_split=${RETAIN_SPLIT} \
            model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
            retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=1 \
            trainer.args.num_train_epochs=${EPOCHS} \
            trainer.args.learning_rate=1e-5 \
            trainer.args.logging_steps=10 \
            trainer.args.eval_strategy=epoch \
            trainer.args.save_strategy=epoch \
            trainer.method_args.module_regex="model\.layers\.7" \
            trainer.method_args.encoder_epochs=4 \
            trainer.method_args.steering_coeff=${STEERING} \
            trainer.method_args.latent_dim=256 \
            trainer.method_args.orth_weight=2.0 \
            trainer.method_args.retain_sep_weight=2.0 \
            trainer.method_args.forget_warmup_steps=30 \
            trainer.method_args.gamma=1.0 \
            trainer.method_args.alpha=${ALPHA} \
            trainer.method_args.retain_loss_type=EMBED_DIFF

        echo "Training completed: saves/unlearn/${TASK_NAME}"

        # Step 2: Evaluate each saved checkpoint and pick best fq
        BEST_FQ=-1
        BEST_CKPT=saves/unlearn/${TASK_NAME}
        for ckpt_dir in saves/unlearn/${TASK_NAME}/checkpoint-*/; do
            [ -d "$ckpt_dir" ] || continue
            # Skip checkpoint dirs that have no model weights (e.g. checkpoint-0 which
            # is created by the trainer for the baseline eval but never saves a model).
            [ -f "${ckpt_dir}config.json" ] || continue
            python3 -c "import json,sys; d=json.load(open('${ckpt_dir}config.json')); sys.exit(0 if 'model_type' in d else 1)" 2>/dev/null || continue
            CUDA_VISIBLE_DEVICES=${GPUS%%,*} $PYTHON src/eval.py \
                --config-name=eval.yaml \
                experiment=eval/tofu/default \
                forget_split=${FORGET_SPLIT} \
                holdout_split=${HOLDOUT_SPLIT} \
                model=${MODEL} \
                task_name=${TASK_NAME} \
                model.model_args.pretrained_model_name_or_path=${ckpt_dir%/} \
                paths.output_dir=${ckpt_dir%/}/evals \
                retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

            FQ=$(python3 -c "
import json
f = '${ckpt_dir%/}/evals/TOFU_EVAL.json'
try:
    d = json.load(open(f))
    print(d['forget_quality']['agg_value'])
except:
    print(-1)
")
            echo "  $(basename $ckpt_dir): fq=$FQ"
            if python3 -c "exit(0 if float('$FQ') > float('$BEST_FQ') else 1)" 2>/dev/null; then
                BEST_FQ=$FQ
                BEST_CKPT=$ckpt_dir
            fi
        done

        echo "Best checkpoint: $BEST_CKPT  (fq=$BEST_FQ)"
        echo "Done. Results: ${BEST_CKPT}/evals"
        echo "=========================================="
    done
done
