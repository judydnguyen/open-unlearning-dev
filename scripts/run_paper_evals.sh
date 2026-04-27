#!/bin/bash
# Pipeline: run all missing evals for paper experiments, then print the report table.
#
# Methods:  Baseline (full model), Retrained, GradAscent, GradDiff, NPO, RMU, SimNPO, Ours (LatentRMU)
# After all evals are present, calls paper_table.py to print the composite-score table.
#
# Edit the variables below before running.

set -e

# ── Configuration ───────────────────────────────────────────────────────────────
MODEL="Llama-3.2-1B-Instruct"

# "Ours" task name pattern — {split} is replaced per-loop with forget01/05/10
OUR_TASK_PATTERN="tofu_${MODEL}_{split}_LatentRMU_v4.8_sweep"

GPU="0"
# PYTHON="/home/judy/conda/envs/unlearning/bin/python"
PYTHON="/home/judy/miniconda3/envs/unlearning/bin/python"

SPLITS=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

BASELINE_METHODS=(GradAscent GradDiff NPO RMU SimNPO)

SAVES_UNLEARN="saves/unlearn"
SAVES_EVAL="saves/eval"
SAVES_FINETUNE="saves/finetune"
# ────────────────────────────────────────────────────────────────────────────────

# Run eval for a single-checkpoint unlearn method (flat evals/ output).
run_eval_flat() {
    local task_name=$1
    local forget_split=$2
    local holdout_split=$3
    local retain_split=$4
    local model_path=$5
    local eval_json="${SAVES_UNLEARN}/${task_name}/evals/TOFU_EVAL.json"

    if [ -f "$eval_json" ]; then
        echo "[skip] ${task_name}"
        return
    fi

    echo "[eval] ${task_name}"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} src/eval.py \
        --config-name=eval.yaml \
        experiment=eval/tofu/default \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        model=${MODEL} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        model.tokenizer_args.pretrained_model_name_or_path=${model_path} \
        paths.output_dir=${SAVES_UNLEARN}/${task_name}/evals \
        retain_logs_path=${SAVES_EVAL}/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json
}

# Run eval for a reference model (Baseline / Retrained) into saves/eval/.
run_eval_ref() {
    local task_name=$1
    local forget_split=$2
    local holdout_split=$3
    local retain_split=$4
    local model_path=$5
    local output_dir=$6
    local eval_json="${output_dir}/TOFU_EVAL.json"

    if [ -f "$eval_json" ]; then
        echo "[skip] ${task_name} (${forget_split}) — ref eval exists"
        return
    fi

    echo "[eval] ${task_name} ref (${forget_split})"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        task_name=${task_name} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        paths.output_dir=${output_dir} \
        retain_logs_path=${SAVES_EVAL}/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json
}

# Eval every unevaluated checkpoint inside a LatentRMU-style run dir.
run_eval_checkpoints() {
    local task_name=$1
    local forget_split=$2
    local holdout_split=$3
    local retain_split=$4
    local run_dir="${SAVES_UNLEARN}/${task_name}"

    if [ ! -d "${run_dir}" ]; then
        echo "[warn] ${task_name} — run dir not found, skipping"
        return
    fi

    local any_ckpt=0
    for ckpt_dir in ${run_dir}/checkpoint-*/; do
        [ -d "$ckpt_dir" ] || continue
        any_ckpt=1
        local ckpt_num
        ckpt_num=$(basename "${ckpt_dir}" | sed 's/checkpoint-//')
        local eval_json="${ckpt_dir}evals/TOFU_EVAL.json"

        if [ -f "$eval_json" ]; then
            echo "[skip] ${task_name}/checkpoint-${ckpt_num}"
            continue
        fi

        echo "[eval] ${task_name}/checkpoint-${ckpt_num}"
        CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} src/eval.py \
            --config-name=eval.yaml \
            experiment=eval/tofu/default \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            model=${MODEL} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=${ckpt_dir} \
            model.tokenizer_args.pretrained_model_name_or_path=${ckpt_dir} \
            paths.output_dir=${ckpt_dir}evals \
            retain_logs_path=${SAVES_EVAL}/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json
    done

    if [ $any_ckpt -eq 0 ]; then
        echo "[warn] ${task_name} — no checkpoints found"
    fi
}

echo "========================================"
echo "Paper eval pipeline: ${MODEL}"
echo "========================================"

for split in "${SPLITS[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    echo ""
    echo "── ${forget_split} ──────────────────────────"

    # Baseline: full finetuned model evaluated on this forget split
    run_eval_ref \
        tofu_${MODEL}_full_${forget_split} \
        ${forget_split} ${holdout_split} ${retain_split} \
        ${SAVES_FINETUNE}/tofu_${MODEL}_full/best \
        ${SAVES_EVAL}/tofu_${MODEL}_full/evals_${forget_split}

    # Retrained: retain-split model (eval already done by tofu_finetune.sh)
    retain_eval="${SAVES_EVAL}/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json"
    if [ -f "$retain_eval" ]; then
        echo "[skip] Retrained (${retain_split}) — eval exists"
    else
        echo "[warn] Retrained eval missing: ${retain_eval}"
        echo "       Run tofu_finetune.sh first to produce the retain-model eval."
    fi

    # Baseline unlearn methods (single checkpoint, flat evals/)
    for method in "${BASELINE_METHODS[@]}"; do
        task_name="tofu_${MODEL}_${forget_split}_${method}"
        model_path="${SAVES_UNLEARN}/${task_name}"

        if [ ! -d "${model_path}" ]; then
            echo "[warn] ${task_name} — no save dir, skipping"
            continue
        fi
        # Check that the dir actually has model weights (not just evals/logs)
        if [ -z "$(ls "${model_path}" 2>/dev/null | grep -vE '^(evals|logs)$')" ]; then
            echo "[warn] ${task_name} — checkpoint weights missing, skipping eval"
            continue
        fi

        run_eval_flat \
            ${task_name} ${forget_split} ${holdout_split} ${retain_split} \
            ${model_path}
    done

    # Ours: LatentRMU (checkpoint-based, eval all checkpoints)
    our_task="${OUR_TASK_PATTERN//\{split\}/${forget_split}}"
    run_eval_checkpoints \
        ${our_task} ${forget_split} ${holdout_split} ${retain_split}
done

echo ""
echo "========================================"
echo "Generating paper table (console)..."
echo "========================================"
${PYTHON} scripts/paper_table.py --model "${MODEL}" \
    --our_pattern "${OUR_TASK_PATTERN}"

echo ""
echo "========================================"
echo "Writing LaTeX table to experiment.tex..."
echo "========================================"
${PYTHON} scripts/update_paper_table.py \
    --model "${MODEL}" \
    --our_pattern "${OUR_TASK_PATTERN}"
