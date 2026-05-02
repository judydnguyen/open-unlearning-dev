#!/usr/bin/env bash
# Re-eval all unlearn methods with the full MIA suite (mia_loss, mia_zlib,
# mia_min_k, mia_min_k_plus_plus) so we can compute the canonical
# Privacy = HM(sLOSS, sZlib, sMin-k, sMink++) score.
#
# Each existing TOFU_EVAL.json is updated in place via eval.tofu.overwrite=true;
# already-populated evals are skipped.
#
# Coverage:
#   * Flat baselines (GradAscent/GradDiff/NPO/RMU/SimNPO × 3 splits) — re-eval
#     the saves/unlearn/<task>/ model into <task>/evals/.
#   * FLOUR _sweep_01_coeff_5 — only best/ has saved weights (per-checkpoint
#     dirs contain only evals, not model snapshots). The script re-evals best/
#     and writes to best/evals/. The step represented is in best_step.json.
#
# Usage:
#   bash scripts/reeval_full_privacy.sh                 # run on GPU 0
#   GPU=1 bash scripts/reeval_full_privacy.sh           # pin to GPU 1
#   DRY_RUN=1 bash scripts/reeval_full_privacy.sh       # just list what would run
#   FLOUR_ONLY=1 bash scripts/reeval_full_privacy.sh    # skip flat baselines
#   BASELINES_ONLY=1 bash scripts/reeval_full_privacy.sh # skip FLOUR

# Note: no `set -e` — a single eval failure (Hydra error, OOM) shouldn't abort
# the whole sweep. Each eval is logged with its exit status instead.

# ── Config ──────────────────────────────────────────────────────────────────
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/home/judy/miniconda3/envs/unlearning/bin/python}"
MODEL="${MODEL:-Llama-3.2-1B-Instruct}"
DRY_RUN="${DRY_RUN:-0}"
FLOUR_ONLY="${FLOUR_ONLY:-0}"
BASELINES_ONLY="${BASELINES_ONLY:-0}"

FLOUR_TASK_PATTERN="tofu_${MODEL}_{split}_LatentRMU_v4.8_sweep_01_coeff_5"

SAVES_UNLEARN="saves/unlearn"
SAVES_EVAL="saves/eval"

SPLITS=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

BASELINE_METHODS=(GradAscent GradDiff NPO RMU SimNPO)

# 4 MIA metrics that must all be present for the eval to be considered complete.
REQUIRED_MIAS=(mia_loss mia_zlib mia_min_k mia_min_k_plus_plus)

# ── Helpers ─────────────────────────────────────────────────────────────────
has_all_mias() {
    # Returns 0 if eval JSON exists and has all 4 MIA metrics with non-null agg_value.
    local eval_json="$1"
    [ -f "$eval_json" ] || return 1
    "$PYTHON" -c "
import json, sys
required = sys.argv[2:]
d = json.load(open(sys.argv[1]))
def g(k):
    e = d.get(k)
    if e is None: return None
    return e.get('agg_value') if isinstance(e, dict) else e
for k in required:
    if g(k) is None: sys.exit(1)
sys.exit(0)
" "$eval_json" "${REQUIRED_MIAS[@]}" 2>/dev/null
}

run_eval() {
    # Run a single full-privacy eval. Skips if all 4 MIAs already present.
    #
    # Args: label, model_path, output_dir, forget_split, holdout_split, retain_split
    local label="$1" model_path="$2" output_dir="$3"
    local forget_split="$4" holdout_split="$5" retain_split="$6"

    local eval_json="${output_dir}/TOFU_EVAL.json"
    local retain_logs="${SAVES_EVAL}/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json"

    if [ ! -d "$model_path" ]; then
        echo "[warn] ${label} — model path not found: ${model_path}"
        return
    fi
    # Find weights either in the dir itself or in best/ (some runs only save weights there)
    local weight_match
    weight_match=$(ls "${model_path}" 2>/dev/null | grep -E '^(model\.safetensors|model-.*\.safetensors|pytorch_model\.bin|model\.safetensors\.index\.json)$' | head -1)
    if [ -z "$weight_match" ] && [ -d "${model_path}/best" ]; then
        local best_weights
        best_weights=$(ls "${model_path}/best" 2>/dev/null | grep -E '^(model\.safetensors|model-.*\.safetensors|pytorch_model\.bin|model\.safetensors\.index\.json)$' | head -1)
        if [ -n "$best_weights" ]; then
            echo "[info] ${label} — using ${model_path}/best (no flat weights)"
            model_path="${model_path}/best"
        fi
    fi
    if [ -z "$(ls "${model_path}" 2>/dev/null | grep -E '^(model\.safetensors|model-.*\.safetensors|pytorch_model\.bin|model\.safetensors\.index\.json)$')" ]; then
        echo "[warn] ${label} — no model weights in ${model_path}, skipping"
        return
    fi
    if has_all_mias "$eval_json"; then
        echo "[skip] ${label} — all 4 MIAs already present"
        return
    fi
    if [ ! -f "$retain_logs" ]; then
        echo "[error] ${label} — retain logs missing: ${retain_logs}"
        return
    fi

    echo "[eval] ${label}"
    if [ "$DRY_RUN" = "1" ]; then
        echo "       model:       ${model_path}"
        echo "       output:      ${output_dir}"
        echo "       retain_logs: ${retain_logs}"
        return
    fi

    # Hydra is sensitive to special characters (parens, slashes, etc.) in
    # override values. Strip them from the task_name we pass to Hydra; the
    # human-readable label is still printed to the log above.
    local hydra_task
    hydra_task=$(printf '%s' "$label" | tr -c 'A-Za-z0-9_.-' '_')

    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" src/eval.py \
        --config-name=eval.yaml \
        experiment=eval/tofu/default \
        eval=tofu_full_privacy \
        eval.tofu.overwrite=true \
        forget_split="${forget_split}" \
        holdout_split="${holdout_split}" \
        model="${MODEL}" \
        task_name="${hydra_task}" \
        model.model_args.pretrained_model_name_or_path="${model_path}" \
        model.tokenizer_args.pretrained_model_name_or_path="${model_path}" \
        paths.output_dir="${output_dir}" \
        retain_logs_path="${retain_logs}"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[fail] ${label} — eval exited ${rc} (continuing)"
    fi
}

# ── Main ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Full-privacy re-eval — ${MODEL}"
echo "  GPU=${GPU}   DRY_RUN=${DRY_RUN}"
echo "================================================================"

for split_triplet in "${SPLITS[@]}"; do
    forget_split=$(echo "$split_triplet" | cut -d' ' -f1)
    holdout_split=$(echo "$split_triplet" | cut -d' ' -f2)
    retain_split=$(echo "$split_triplet" | cut -d' ' -f3)

    echo ""
    echo "── ${forget_split} ──"

    # 1) Flat baselines: weights live directly in saves/unlearn/<task>/
    if [ "$FLOUR_ONLY" != "1" ]; then
        for method in "${BASELINE_METHODS[@]}"; do
            task="tofu_${MODEL}_${forget_split}_${method}"
            run_eval \
                "${task}" \
                "${SAVES_UNLEARN}/${task}" \
                "${SAVES_UNLEARN}/${task}/evals" \
                "${forget_split}" "${holdout_split}" "${retain_split}"
        done
    fi

    # 2) FLOUR (LatentRMU sweep_01_coeff_5): only best/ has saved weights —
    #    per-checkpoint dirs only contain evals, not model snapshots. Re-eval
    #    the best/ model and write to best/evals/. The step it represents is
    #    recorded in best/best_step.json.
    if [ "$BASELINES_ONLY" != "1" ]; then
        flour_task="${FLOUR_TASK_PATTERN//\{split\}/${forget_split}}"
        flour_best="${SAVES_UNLEARN}/${flour_task}/best"
        if [ -d "$flour_best" ]; then
            best_step=""
            if [ -f "${flour_best}/best_step.json" ]; then
                best_step=$("$PYTHON" -c "import json; print(json.load(open('${flour_best}/best_step.json')).get('step',''))" 2>/dev/null)
            fi
            label="${flour_task}/best"
            [ -n "$best_step" ] && label="${label}(step=${best_step})"
            run_eval \
                "${label}" \
                "${flour_best}" \
                "${flour_best}/evals" \
                "${forget_split}" "${holdout_split}" "${retain_split}"
        else
            echo "[warn] ${flour_task}/best — not found"
        fi
    fi
done

echo ""
echo "================================================================"
if [ "$DRY_RUN" = "1" ]; then
    echo "  Dry run complete. Re-run without DRY_RUN=1 to actually evaluate."
else
    echo "  Full-privacy re-eval complete."
    echo "  Run scripts/paper_table.py to see updated Privacy scores."
fi
echo "================================================================"
