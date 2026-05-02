#!/bin/bash

# Run MUSE unlearning on the locally-finetuned Llama-3.2-3B-Instruct targets.
# Loads from saves/finetune/muse_Llama-3.2-3B-Instruct_${split}_target/last (NOT the
# muse-bench public Llama-2-7B target). retain_logs_path points at our local retrain
# eval so privleak is computed against the matching reference.
#
# Trainers list selects which baselines to run; uncomment to enable. LatentRMU at the
# bottom is the proposed method.

export HF_HOME=/tank/home/judy/.cache/huggingface
export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi
if [ -z "${HF_TOKEN:-}" ]; then
    HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
    [ -f "$HF_TOKEN_FILE" ] && export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

model=Llama-3.2-3B-Instruct
per_device_train_batch_size=2
gradient_accumulation_steps=1     # effective batch = 2 * 1 * 2 GPUs = 4

TRAIN_GPUS="0,1"
EVAL_GPU="0"
ACCEL_CFG=configs/accelerate/ddp_2gpu_config.yaml

data_splits=(
    "News"
    # "Books"
)

trainers=(
    # "GradAscent"
    # "GradDiff"
    # "NPO"
    # "SimNPO"
    # "RMU"
    "LatentRMUParallelDDP"
)

for data_split in "${data_splits[@]}"; do
    target_path=saves/finetune/muse_${model}_${data_split}_target/last
    retain_logs=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json

    if [ ! -d "$target_path" ]; then
        echo "[skip] $data_split: target checkpoint missing at $target_path"
        continue
    fi
    if [ ! -f "$retain_logs" ]; then
        echo "[warn] $data_split: retain logs missing at $retain_logs (privleak will be off)"
    fi

    for trainer in "${trainers[@]}"; do
        task_name=muse_${model}_${data_split}_${trainer}
        save_dir=saves/unlearn/${task_name}

        echo "============================================="
        echo "Run: $task_name"
        echo "  target = $target_path"
        echo "  retain_logs = $retain_logs"
        echo "============================================="

        # ---- 1. unlearn ----
        if [ -f "$save_dir/last/model.safetensors" ] || [ -f "$save_dir/last/model.safetensors.index.json" ]; then
            echo "[done] $task_name training already exists, skipping (rm -rf $save_dir to redo)"
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
                model.model_args.pretrained_model_name_or_path=${target_path} \
                retain_logs_path=${retain_logs} \
                trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true
        fi

        # ---- 2. eval ----
        if find "$save_dir/evals" -maxdepth 2 -name MUSE_SUMMARY.json 2>/dev/null | grep -q .; then
            echo "[done] $task_name eval already exists"
        elif [ -f "$save_dir/last/model.safetensors" ] || [ -f "$save_dir/last/model.safetensors.index.json" ]; then
            CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
                experiment=eval/muse/default.yaml \
                data_split=${data_split} \
                task_name=${task_name} \
                model=${model} \
                model.model_args.pretrained_model_name_or_path=${save_dir}/last \
                paths.output_dir=${save_dir}/evals \
                retain_logs_path=${retain_logs}
        else
            echo "[skip eval] no checkpoint at $save_dir/last"
        fi
    done
done

echo "=== All runs complete. Compare with: ==="
echo "  for f in saves/unlearn/muse_${model}_*/evals/**/MUSE_SUMMARY.json; do echo \$f; cat \$f; echo; done"
