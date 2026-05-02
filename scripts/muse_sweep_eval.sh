#!/bin/bash
# Evaluate all LatentRMUParallelDDP sweep checkpoints on MUSE-News.
# Runs on a single GPU (eval doesn't need DDP).
# Usage: bash scripts/muse_sweep_eval.sh 2>&1 | tee /tmp/muse_sweep_eval.log

set -euo pipefail

source /home/judy/miniconda3/etc/profile.d/conda.sh
conda activate /data/judy/conda/envs/unlearning

export HF_HOME=/tank/home/judy/.cache/huggingface
export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune

model=Llama-2-7b-hf
data_split=News
retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json

# All 6 sweep task names
tasks=(
    "muse_sweep_${model}_${data_split}_g5_sc20"
    "muse_sweep_${model}_${data_split}_g8_sc20"
    "muse_sweep_${model}_${data_split}_g8_sc40"
    "muse_sweep_${model}_${data_split}_g8_sc40_a05"
    "muse_sweep_${model}_${data_split}_g10_sc80_a05"
    "muse_sweep_${model}_${data_split}_g8_sc40_fmlp"
)

for task_name in "${tasks[@]}"; do
    ckpt=saves/unlearn/${task_name}/last
    output_dir=saves/unlearn/${task_name}/evals

    if [ ! -d "$ckpt" ]; then
        echo "SKIP: checkpoint not found at ${ckpt}"
        continue
    fi

    echo "=== Evaluating: ${task_name} ==="

    CUDA_VISIBLE_DEVICES=2 python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${data_split} \
        task_name=${task_name} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=${ckpt} \
        paths.output_dir=${output_dir} \
        retain_logs_path=${retain_logs_path}

    echo "=== Done: ${task_name} → ${output_dir} ==="
done

echo ""
echo "=== All evals complete. Results: ==="
for task_name in "${tasks[@]}"; do
    result=saves/unlearn/${task_name}/evals/MUSE_EVAL.json
    if [ -f "$result" ]; then
        echo "--- ${task_name} ---"
        python -c "
import json, sys
d = json.load(open('${result}'))
# Print top-level scalar metrics
for k, v in d.items():
    if isinstance(v, (int, float)):
        print(f'  {k}: {v:.4f}')
    elif isinstance(v, dict):
        for kk, vv in v.items():
            if isinstance(vv, (int, float)):
                print(f'  {k}/{kk}: {vv:.4f}')
" 2>/dev/null || echo "  (could not parse ${result})"
    fi
done
